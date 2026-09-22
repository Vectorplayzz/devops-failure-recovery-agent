"""Console surface - the same agent, driven from a terminal.

Two jobs. It lets the agent be developed and demonstrated without any chat
platform at all, and it is the cheapest possible proof that the surface
abstraction is real: if `Card` only ever rendered to Discord embeds, "surface
neutral" would be a claim rather than a fact.

Approvals here still carry an identity - the OS username - because the rule
that an approval comes from an authenticated principal and never from text
must hold on every surface, including this one.
"""

from __future__ import annotations

import asyncio
import getpass
import itertools
import sys

from .base import (
    ApprovalEvent,
    Card,
    ChatQuestion,
    ChatSurface,
    ChatUser,
    MessageRef,
    Tone,
)

# ASCII only: a Windows console defaults to cp1252 and raises on anything else.
_TONE_MARK = {
    Tone.CRITICAL: "[!!]",
    Tone.WARNING: "[! ]",
    Tone.INFO: "[i ]",
    Tone.SUCCESS: "[ok]",
    Tone.SECURITY: "[SEC]",
    Tone.NEUTRAL: "[  ]",
}

_WIDTH = 78


def render(card: Card) -> str:
    mark = _TONE_MARK.get(card.tone, "[  ]")
    lines = ["", "=" * _WIDTH, f"{mark} {card.title}".rstrip(), "=" * _WIDTH]
    if card.body:
        lines += [card.body, ""]
    for f in card.fields:
        lines.append(f"  {f.name}")
        for line in f.value.splitlines() or [""]:
            lines.append(f"      {line}")
        lines.append("")
    if card.footer:
        lines += ["-" * _WIDTH, f"  {card.footer}"]
    if card.buttons:
        lines.append("  actions: " + "  ".join(f"[{b.label}]" for b in card.buttons))
    return "\n".join(lines)


class ConsoleSurface(ChatSurface):
    name = "console"

    def __init__(self, *, stream: object = None, auto_approve: bool = False) -> None:
        super().__init__()
        self.stream = stream or sys.stdout
        # Unattended runs (the closed-loop proof, CI) need a decision without a
        # human at the keyboard. It is named honestly rather than disguised as
        # a policy setting, and it never affects the Discord surface.
        self.auto_approve = auto_approve
        self._ids = itertools.count(1)
        self.posted: list[Card] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def _user(self) -> ChatUser:
        try:
            name = getpass.getuser()
        except Exception:  # noqa: BLE001
            name = "local"
        return ChatUser(id=f"local:{name}", display_name=name, is_admin=True)

    async def post(self, card: Card, *, channel_id: str = "") -> MessageRef:
        self.posted.append(card)
        print(render(card), file=self.stream, flush=True)

        if card.buttons and self._on_approval is not None:
            approve = next(
                (b for b in card.buttons if b.action_id.startswith("approve:")), None
            )
            if approve is not None:
                decision = self.auto_approve or await self._prompt()
                verb, incident_id, action_id = approve.action_id.split(":", 2)
                outcome = await self._on_approval(
                    ApprovalEvent(
                        incident_id=incident_id,
                        action_id=action_id,
                        approved=decision,
                        user=self._user(),
                        surface=self.name,
                    )
                )
                label = outcome.label or ("APPROVED" if decision else "REJECTED")
                shown = label if outcome.retire_card else "NOT ACCEPTED (card stays live)"
                print(f"  -> {shown}: {outcome.message}", file=self.stream, flush=True)

        return MessageRef(
            surface=self.name, channel_id="console", message_id=str(next(self._ids))
        )

    async def _prompt(self) -> bool:
        answer = await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("  Approve? [y/N] ").strip().lower()
        )
        return answer in {"y", "yes"}

    async def update(self, ref: MessageRef, card: Card) -> None:
        print(render(card), file=self.stream, flush=True)

    async def ask(self, text: str) -> None:
        """Drive the agent from the terminal, as a chat message would."""
        if self._on_question is None:
            print("  (no question handler configured)", file=self.stream)
            return
        reply = await self._on_question(
            ChatQuestion(
                text=text, user=self._user(), channel_id="console", surface=self.name
            )
        )
        if isinstance(reply, Card):
            print(render(reply), file=self.stream, flush=True)
        else:
            print(f"\n{reply}\n", file=self.stream, flush=True)
