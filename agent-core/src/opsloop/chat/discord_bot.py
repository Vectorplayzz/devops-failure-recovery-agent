"""Discord surface.

A translator, not a brain. It converts `Card` objects into Discord embeds,
converts button clicks and slash commands into neutral events, and hands them
to handlers that live in the agent. No incident logic, no policy decisions, no
model calls happen in this file - which is what makes swapping in Teams or
Slack a rewrite of one module rather than of the system.

Two Discord-specific concerns are handled here because they are genuinely
platform concerns:

  IDENTITY. `interaction.user.id` is Discord's authenticated snowflake. It is
  the only thing ever used as an approval identity. Nothing typed in a message
  can become one - which is what keeps a log line claiming "the on-call
  engineer approved this" from being an approval.

  BUTTON REUSE. Once a decision is taken the buttons are removed from the
  message. Without that, an approval card stays clickable forever and a second
  click hours later would re-run the action against a system that has moved on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import (
    ApprovalEvent,
    ApprovalOutcome,
    Button,
    ButtonStyle,
    Card,
    ChatCommand,
    ChatQuestion,
    ChatSurface,
    ChatUser,
    MessageRef,
    Tone,
    truncate,
)

try:  # pragma: no cover - import guard
    import discord
    from discord import app_commands
except ImportError:  # pragma: no cover
    discord = None  # type: ignore[assignment]
    app_commands = None  # type: ignore[assignment]

log = logging.getLogger("opsloop.chat.discord")

# Discord's own limits. Exceeding one rejects the entire message, so an
# over-long stack trace would otherwise lose the whole incident alert.
MAX_EMBED_TITLE = 256
MAX_EMBED_DESCRIPTION = 4096
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024
MAX_FIELDS = 25
MAX_FOOTER = 2048
MAX_CONTENT = 2000

_TONE_COLOUR = {
    Tone.CRITICAL: 0xD93025,  # red
    Tone.WARNING: 0xF9AB00,  # amber
    Tone.INFO: 0x1A73E8,  # blue
    Tone.SUCCESS: 0x1E8E3E,  # green
    Tone.SECURITY: 0x9334E6,  # purple - distinct from ordinary failure
    Tone.NEUTRAL: 0x5F6368,  # grey
}


def _button_style(style: ButtonStyle) -> Any:
    return {
        ButtonStyle.PRIMARY: discord.ButtonStyle.success,
        ButtonStyle.DANGER: discord.ButtonStyle.danger,
        ButtonStyle.SECONDARY: discord.ButtonStyle.secondary,
    }[style]


def card_to_embed(card: Card) -> Any:
    """Render a neutral Card as a Discord embed, clamped to every limit."""
    embed = discord.Embed(
        title=truncate(card.title, MAX_EMBED_TITLE) or None,
        description=truncate(card.body, MAX_EMBED_DESCRIPTION) or None,
        colour=_TONE_COLOUR.get(card.tone, _TONE_COLOUR[Tone.NEUTRAL]),
    )
    for f in card.fields[:MAX_FIELDS]:
        embed.add_field(
            name=truncate(f.name, MAX_FIELD_NAME) or "​",
            value=truncate(f.value, MAX_FIELD_VALUE) or "​",
            inline=f.inline,
        )
    if len(card.fields) > MAX_FIELDS:
        embed.add_field(
            name="​",
            value=f"_...and {len(card.fields) - MAX_FIELDS} more field(s) omitted_",
            inline=False,
        )
    if card.footer:
        embed.set_footer(text=truncate(card.footer, MAX_FOOTER))
    return embed


class CardView(discord.ui.View if discord else object):  # type: ignore[misc]
    """Buttons for one card.

    `timeout=None` because an approval's lifetime is a policy decision, not a
    UI one: `PolicyEngine.is_expired` owns it, and it refuses a stale approval
    even if the button still renders. Letting the View expire independently
    would produce two different, silently disagreeing deadlines.
    """

    def __init__(self, surface: DiscordSurface, buttons: list[Button]) -> None:
        super().__init__(timeout=None)
        self.surface = surface
        for b in buttons:
            self.add_item(_CardButton(surface, b))


class _CardButton(discord.ui.Button if discord else object):  # type: ignore[misc]
    def __init__(self, surface: DiscordSurface, spec: Button) -> None:
        super().__init__(
            label=spec.label,
            style=_button_style(spec.style),
            emoji=spec.emoji or None,
            custom_id=spec.action_id,
        )
        self.surface = surface
        self.spec = spec

    async def callback(self, interaction: Any) -> None:  # pragma: no cover - needs a live gateway
        await self.surface._handle_button(interaction, self.spec.action_id)


class DiscordSurface(ChatSurface):
    name = "discord"

    def __init__(
        self,
        token: str,
        *,
        default_channel_id: int | str = 0,
        guild_id: int | str = 0,
        admin_role: str = "",
        enable_message_chat: bool = True,
        scenario_choices: list[str] | None = None,
    ) -> None:
        super().__init__()
        if discord is None:
            raise RuntimeError(
                "discord.py is not installed. Run: pip install 'discord.py>=2.4'"
            )
        self.token = token
        self.default_channel_id = int(default_channel_id or 0)
        self.guild_id = int(guild_id or 0)
        self.admin_role = admin_role
        self.enable_message_chat = enable_message_chat
        self.scenario_choices = list(scenario_choices or [])

        intents = discord.Intents.default()
        # message_content is a PRIVILEGED intent. Without it the bot receives
        # empty message bodies and free-text chat silently does nothing, which
        # looks like a broken bot rather than a missing setting - so it is
        # checked explicitly at startup and reported.
        intents.message_content = enable_message_chat
        self.client = discord.Client(intents=intents)
        self.tree = app_commands.CommandTree(self.client)
        self._ready = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None
        self._register_events()
        self._register_commands()

    # -- lifecycle ---------------------------------------------------------

    def _register_events(self) -> None:
        @self.client.event
        async def on_ready() -> None:  # pragma: no cover
            guild = discord.Object(id=self.guild_id) if self.guild_id else None
            try:
                if guild is not None:
                    # Guild-scoped commands appear immediately; global ones can
                    # take up to an hour to propagate, which is unusable while
                    # iterating.
                    self.tree.copy_global_to(guild=guild)
                    await self.tree.sync(guild=guild)
                else:
                    await self.tree.sync()
            except discord.HTTPException:
                log.exception("slash command sync failed")
            log.info("discord surface ready as %s", self.client.user)
            self._ready.set()

        @self.client.event
        async def on_message(message: Any) -> None:  # pragma: no cover
            if message.author.bot or self._on_question is None:
                return
            mentioned = self.client.user in getattr(message, "mentions", [])
            is_dm = isinstance(message.channel, discord.DMChannel)
            if not (mentioned or is_dm):
                return

            text = message.content
            for mention in (f"<@{self.client.user.id}>", f"<@!{self.client.user.id}>"):
                text = text.replace(mention, "")
            text = text.strip()
            if not text:
                return

            async with message.channel.typing():
                try:
                    reply = await self._on_question(
                        ChatQuestion(
                            text=text,
                            user=self._user_of(message.author),
                            channel_id=str(message.channel.id),
                            surface=self.name,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("question handler failed")
                    reply = f"Something went wrong handling that: `{exc}`"
            await self._reply_to_message(message, reply)

    def _register_commands(self) -> None:
        def simple(name: str, description: str) -> None:
            @self.tree.command(name=name, description=description)
            async def _cmd(interaction: Any) -> None:  # pragma: no cover
                await self._handle_command(interaction, name, {})

        simple("status", "Current incidents and telemetry adapter health")
        simple("incidents", "List recent incidents")
        simple("discover", "Inventory the configured hosts")
        simple("settings", "Show the active LLM provider and adapters")

        @self.tree.command(name="incident", description="Show one incident in detail")
        @app_commands.describe(incident_id="Incident id, e.g. inc_a1b2c3d4")
        async def _incident(interaction: Any, incident_id: str) -> None:  # pragma: no cover
            await self._handle_command(interaction, "incident", {"incident_id": incident_id})

        @self.tree.command(name="diagnose", description="Investigate an incident now")
        @app_commands.describe(incident_id="Incident id to investigate")
        async def _diagnose(interaction: Any, incident_id: str) -> None:  # pragma: no cover
            await self._handle_command(interaction, "diagnose", {"incident_id": incident_id})

        if self.scenario_choices:
            # Demo-only commands, registered only when the app supplies
            # scenario names - the surface itself knows nothing about them.
            simple("clear", "Demo: clear every injected fault and restore health")

            @self.tree.command(name="inject", description="Demo: break production on purpose")
            @app_commands.describe(scenario="Which failure to inject")
            async def _inject(interaction: Any, scenario: str) -> None:  # pragma: no cover
                await self._handle_command(interaction, "inject", {"scenario": scenario})

            @_inject.autocomplete("scenario")
            async def _inject_choices(interaction: Any, current: str) -> list[Any]:  # pragma: no cover
                return [
                    app_commands.Choice(name=s, value=s)
                    for s in self.scenario_choices
                    if current.lower() in s.lower()
                ][:25]

        @self.tree.command(name="ask", description="Ask the agent about production")
        @app_commands.describe(question="What do you want to know?")
        async def _ask(interaction: Any, question: str) -> None:  # pragma: no cover
            if self._on_question is None:
                await interaction.response.send_message("No handler configured.", ephemeral=True)
                return
            await interaction.response.defer(thinking=True)
            try:
                reply = await self._on_question(
                    ChatQuestion(
                        text=question,
                        user=self._user_of(interaction.user),
                        channel_id=str(interaction.channel_id),
                        surface=self.name,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("ask handler failed")
                reply = f"Something went wrong: `{exc}`"
            await self._send_followup(interaction, reply)

    async def start(self, *, timeout: float = 60.0) -> None:
        """Connect, and fail loudly if Discord refuses.

        Waiting on the ready event alone is a trap: if the gateway rejects the
        connection - a bad token, or the MESSAGE CONTENT intent requested but
        not enabled in the developer portal - the connect task dies, the event
        is never set, and startup hangs forever with no error. Racing the two
        turns that into an immediate, readable failure.
        """
        self._task = asyncio.create_task(self.client.start(self.token))
        ready = asyncio.create_task(self._ready.wait())
        done, _ = await asyncio.wait(
            {self._task, ready}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )

        if ready in done:
            return

        ready.cancel()
        # Release the HTTP session before raising, or the process exits with
        # an "Unclosed client session" error stacked on top of the real one.
        await self.client.close()
        if self._task in done:
            exc = self._task.exception()
            if isinstance(exc, discord.PrivilegedIntentsRequired):
                raise RuntimeError(
                    "Discord refused the MESSAGE CONTENT intent. Enable it at "
                    "https://discord.com/developers/applications -> your app -> Bot "
                    "-> Privileged Gateway Intents, or set "
                    "OPSLOOP_DISCORD_MESSAGE_CHAT=false to run with slash commands only."
                ) from exc
            if isinstance(exc, discord.LoginFailure):
                raise RuntimeError(
                    "Discord rejected the bot token. Reset it in the developer portal "
                    "and update OPSLOOP_DISCORD_TOKEN."
                ) from exc
            raise RuntimeError(f"Discord connection failed: {exc!r}") from exc

        raise TimeoutError(f"Discord did not become ready within {timeout:.0f}s")

    async def stop(self) -> None:
        await self.client.close()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # -- identity ----------------------------------------------------------

    def _user_of(self, author: Any) -> ChatUser:
        """Build an identity from Discord's authenticated data only."""
        is_admin = False
        if self.admin_role:
            roles = getattr(author, "roles", []) or []
            is_admin = any(getattr(r, "name", "") == self.admin_role for r in roles)
        else:
            perms = getattr(author, "guild_permissions", None)
            is_admin = bool(perms and perms.administrator)
        return ChatUser(
            id=str(author.id),
            display_name=getattr(author, "display_name", None) or str(author),
            is_admin=is_admin,
        )

    # -- inbound -----------------------------------------------------------

    async def _handle_button(self, interaction: Any, action_id: str) -> None:  # pragma: no cover
        if self._on_approval is None:
            await interaction.response.send_message(
                "No approval handler is configured.", ephemeral=True
            )
            return

        try:
            verb, incident_id, act_id = action_id.split(":", 2)
        except ValueError:
            await interaction.response.send_message(
                f"Malformed button id `{action_id}`.", ephemeral=True
            )
            return

        await interaction.response.defer()
        event = ApprovalEvent(
            incident_id=incident_id,
            action_id=act_id,
            approved=(verb == "approve"),
            user=self._user_of(interaction.user),  # authenticated, not typed
            surface=self.name,
        )
        try:
            outcome = await self._on_approval(event)
        except Exception as exc:  # noqa: BLE001
            log.exception("approval handler failed")
            # An internal error is not a decision. Keep the card live.
            outcome = ApprovalOutcome(
                message=f"The approval could not be processed: `{exc}`", retire_card=False
            )

        if not outcome.retire_card:
            # A transient refusal - fix already in flight, clicker not an
            # approver. Tell only the person who clicked, and leave the buttons
            # for whoever should press them, when they should.
            try:
                await interaction.followup.send(
                    truncate(outcome.message, MAX_CONTENT), ephemeral=True
                )
            except discord.HTTPException:
                log.exception("could not send approval refusal")
            return

        # A recorded decision, or a card that can never work again: retire the
        # buttons so it cannot be replayed against a system that has moved on.
        label = outcome.label or ("APPROVED" if event.approved else "REJECTED")
        try:
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed is not None:
                embed.add_field(
                    name=f"Decision: {label}",
                    value=truncate(
                        f"by {event.user.display_name} - {outcome.message}", MAX_FIELD_VALUE
                    ),
                    inline=False,
                )
                embed.colour = _TONE_COLOUR[
                    Tone.INFO if label == "APPROVED" else Tone.NEUTRAL
                ]
                await interaction.message.edit(embed=embed, view=None)
            else:
                await interaction.message.edit(view=None)
        except discord.HTTPException:
            log.exception("could not retire approval buttons")

    async def _handle_command(
        self, interaction: Any, name: str, args: dict[str, Any]
    ) -> None:  # pragma: no cover
        if self._on_command is None:
            await interaction.response.send_message(
                "No command handler is configured.", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            result = await self._on_command(
                ChatCommand(
                    name=name,
                    args=args,
                    user=self._user_of(interaction.user),
                    channel_id=str(interaction.channel_id),
                    surface=self.name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("command handler failed: %s", name)
            result = f"`/{name}` failed: `{exc}`"
        await self._send_followup(interaction, result)

    # -- outbound ----------------------------------------------------------

    async def _send_followup(self, interaction: Any, result: Card | str) -> None:  # pragma: no cover
        if isinstance(result, Card):
            view = CardView(self, result.buttons) if result.buttons else None
            await interaction.followup.send(embed=card_to_embed(result), view=view)
        else:
            await interaction.followup.send(truncate(str(result), MAX_CONTENT))

    async def _reply_to_message(self, message: Any, result: Card | str) -> None:  # pragma: no cover
        if isinstance(result, Card):
            view = CardView(self, result.buttons) if result.buttons else None
            await message.reply(embed=card_to_embed(result), view=view)
        else:
            await message.reply(truncate(str(result), MAX_CONTENT))

    async def post(self, card: Card, *, channel_id: str = "") -> MessageRef:
        target = int(channel_id) if channel_id else self.default_channel_id
        if not target:
            raise RuntimeError(
                "No channel to post to. Set OPSLOOP_DISCORD_CHANNEL_ID or pass "
                "channel_id explicitly."
            )
        channel = self.client.get_channel(target) or await self.client.fetch_channel(target)
        view = CardView(self, card.buttons) if card.buttons else None
        message = await channel.send(embed=card_to_embed(card), view=view)
        return MessageRef(
            surface=self.name, channel_id=str(target), message_id=str(message.id)
        )

    async def update(self, ref: MessageRef, card: Card) -> None:
        channel = self.client.get_channel(int(ref.channel_id)) or await self.client.fetch_channel(
            int(ref.channel_id)
        )
        message = await channel.fetch_message(int(ref.message_id))
        view = CardView(self, card.buttons) if card.buttons else None
        await message.edit(embed=card_to_embed(card), view=view)
