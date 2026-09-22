# Discord bot setup

Ten minutes, no tenant, no admin approval, no cost.

## 1. Create the application

1. Go to <https://discord.com/developers/applications> → **New Application**.
   Name it `OpsLoop`.
2. **Bot** tab → **Reset Token** → copy it. This is the only time it is shown.
   It is a credential: put it in `.env`, never in the repo.
3. On the same tab, under **Privileged Gateway Intents**, enable
   **MESSAGE CONTENT INTENT**.

> **Do not skip the intent.** Without it the bot connects, slash commands
> work, and `@OpsLoop what's broken?` silently does nothing — because message
> bodies arrive empty. It looks like a broken bot rather than a missing
> checkbox, and it is the single most common setup mistake. Set
> `OPSLOOP_DISCORD_MESSAGE_CHAT=false` if you deliberately want slash commands
> only.

## 2. Invite it to a server

**OAuth2 → URL Generator**:

- Scopes: `bot`, `applications.commands`
- Bot permissions: `Send Messages`, `Embed Links`, `Read Message History`,
  `Use Slash Commands`

Open the generated URL and add it to a server you own. Create a server first if
you need one — it takes seconds and you get admin on it automatically.

## 3. Get the IDs

Enable **User Settings → Advanced → Developer Mode**, then right-click →
**Copy ID** on:

- the **server** (guild) — commands sync there instantly; global commands can
  take up to an hour to appear, which is unusable while iterating
- the **channel** OpsLoop should post incidents into

## 4. Configure

`agent-core/.env`:

```
OPSLOOP_DISCORD_TOKEN=your-bot-token
OPSLOOP_DISCORD_GUILD_ID=123456789012345678
OPSLOOP_DISCORD_CHANNEL_ID=123456789012345678
OPSLOOP_DISCORD_ADMIN_ROLE=SRE        # optional; defaults to server admins
OPSLOOP_DISCORD_MESSAGE_CHAT=true
```

`.env` is already in `.gitignore`. **A leaked bot token lets anyone drive your
agent** — including approving its actions. If it ever lands in a commit,
screenshot or paste, reset it in the developer portal immediately.

## 5. Commands

| Command | Does |
|---|---|
| `/status` | current incidents and telemetry adapter health |
| `/incidents` | recent incidents |
| `/incident <id>` | one incident: diagnosis, evidence, timeline |
| `/diagnose <id>` | investigate now |
| `/discover` | inventory the configured hosts |
| `/settings` | active LLM provider and adapters |
| `/llm` | **settings menu**: choose the LLM provider, model, base URL and key (admins) |
| `/llm_test` | check the active LLM provider responds |
| `/ask <question>` | ask in natural language |
| `@OpsLoop ...` | same, conversationally (needs the message-content intent) |
| `/inject <scenario>` | demo only: break the demo stack on purpose |
| `/clear` | demo only: remove every injected fault and close open incidents |

Approvals are **buttons on the incident card**, not typed commands. That is a
security property, not a UI preference — see below.

**Who can approve.** Only members with `OPSLOOP_DISCORD_ADMIN_ROLE`, or server
administrators when it is unset. Being able to read the incident channel is not
the same as being allowed to change production, so anyone else who clicks
Approve gets a refusal and nothing runs. `/inject` and `/clear` follow the same
rule.

## 6. Run it

```bash
cd agent-core
.venv/Scripts/python.exe -m opsloop
```

It posts **"OpsLoop is online"** with a status card to your channel, then
scans every 15 seconds. With `OPSLOOP_DEMO_ENABLED=true` and the demo stack up
(`docker compose up -d` in `demo-stack/`), try:

```
/inject scenario:oom
```

Within about half a minute an incident card appears with two approval cards:
**Restart** (LOW risk, and it says plainly it will not fix a leak) and **Ship
the corrected build** (MEDIUM). Approve the restart first to watch verification
catch the relapse and roll it back; then approve the real fix and watch it hold.

With no `OPSLOOP_DISCORD_TOKEN` set, the same agent runs in the terminal.

## 7. Connect an LLM

`/llm` opens a private form - not slash-command options, because those are
visible to everyone in the channel and one field is an API key. Fill in:

- **Provider**: a preset (`groq`, `openai`, `anthropic`, `openrouter`,
  `ollama`, `google`, `deepseek`, ...) or any label you like
- **Base URL**: blank for a preset, or **any OpenAI-compatible endpoint**
  (vLLM, LM Studio, a gateway, your own server)
- **API key**: blank keeps the current one
- **Model**: blank for the preset's default

The new provider is **connection-tested before it replaces the old one**, so a
typo cannot silently cut the agent's reasoning off. The choice is saved to
`agent-core/opsloop-settings.json` (gitignored) and survives restarts.
`.env` (`OPSLOOP_LLM_PROVIDER`, `_BASE_URL`, `_API_KEY`, `_MODEL`) is only the
starting point.

With a model connected, `/ask` and `@OpsLoop` answer from live telemetry and
cite the evidence they used. Rule-based triage still handles the failures it
recognises - instantly and for free - and the model takes the ones it
abstains on.

## 8. Connect a remote host (VPS)

In `agent-core/.env`:

```
OPSLOOP_SSH_HOST=your.host
OPSLOOP_SSH_USER=user
OPSLOOP_SSH_KEY=~/.ssh/opsloop_ed25519
```

Use a **dedicated key without a passphrase** - an unattended agent cannot type
one - and install it on the host once:

```bash
ssh-keygen -t ed25519 -N "" -C opsloop-agent -f ~/.ssh/opsloop_ed25519
ssh-copy-id -i ~/.ssh/opsloop_ed25519.pub user@your.host
```

Revoke it any time by deleting its line from `~/.ssh/authorized_keys` on the
host. The agent then watches for a full disk, failed systemd units and memory
exhaustion, and the model gains read-only tools for journald, log files and
disk usage.

If the host refuses the connection the agent **backs off** - 15 seconds,
doubling to 10 minutes - rather than retrying every probe. Retrying a failing
login in a loop looks exactly like brute force and gets your own address
banned by fail2ban.

## Why Discord is fine here (and where it is not)

**Fine:** buttons carry `interaction.user.id`, an identity Discord
authenticates. Nothing typed in a message can forge one. Buttons are removed
once a decision is taken, so a card cannot be clicked twice against a system
that has since moved on. Threads keep an incident together. Embeds render the
same `Card` objects every other surface renders.

**Not fine, and worth stating plainly:** Discord is not an enterprise
incident-response surface. No SRE team runs production approvals there. It
ships first because it needs no tenant, no app registration and no admin
approval — not because it is the right production choice.

The architecture is what carries the claim instead: `ChatSurface` in
[`chat/base.py`](../agent-core/src/opsloop/chat/base.py) is the same kind of
boundary as the telemetry adapters and the LLM providers. Discord ships,
console ships, and Teams is a third implementation of a settled interface
rather than a rewrite.

**The framing that holds up:**

> Neutral on three axes — telemetry, model, and chat surface. Discord is what
> ships and what gets demonstrated. Teams is a third implementation of an
> interface that already has two.

That is a stronger architectural position than picking one vendor, and it can
be demonstrated today.

## Adding Teams later

Implement `ChatSurface` — `start`, `stop`, `post`, `update` — mapping `Card` to
an Adaptive Card and the AAD object id to `ChatUser.id`. No incident logic,
policy code, or model code changes. `chat/console.py` is ~130 lines and is the
reference for how small a surface should be.
