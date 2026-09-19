# OpsLoop

**A conversational SRE agent that verifies its own fixes.**

Talk to it in **Discord**. It watches production, diagnoses failures
from cited evidence, proposes a fix, waits for a human to approve it, executes it —
and then **verifies that the system actually recovered**, rolling back automatically
if it did not.

That last step is the point. Across eleven commercial products surveyed in August 2026
— PagerDuty, Datadog, Azure SRE Agent, incident.io, Resolve.ai, Cleric, Traversal,
Parity, AWS DevOps Agent, New Relic and Dynatrace — **not one documents a guarantee
that the fix actually worked.** Every "self-healing" claim in the market stops at
*action taken*, not *recovery confirmed*.

---

## Design goals, and where each one lives in the code

| # | Goal | Enforced by |
|---|---|---|
| 1 | **Closed loop** — verify recovery, auto-rollback on failure | `verify/`, plus `ProposedAction` requiring an inverse |
| 2 | **Surface-neutral** | `chat/` — one `Card` model, rendered by Discord and console; Teams is a third implementation, not a rewrite |
| 3 | **Vendor-neutral** | `telemetry/` adapter protocol; `llm/` provider abstraction |
| 4 | **Evidence-grounded, attack-aware** | `domain/models.py` grounding rules; `security/sanitizer.py` |

None of these are prompt instructions. They are invariants in code — see
`agent-core/tests/test_foundations.py`.

---

## Architecture

```
Discord  ·  console  ·  (Teams, later)
      │  one Card model, rendered natively by each
      ▼
agent-core/               Python — one runtime, all reasoning and control
      ├── chat/           ChatSurface protocol + discord + console renderers
      ├── domain/         Incident, Evidence, Hypothesis, Action, Verification
      ├── llm/            any provider; custom OpenAI-compatible base URL
      ├── telemetry/      adapter protocol: docker | ssh | loki | elasticsearch
      ├── security/       telemetry sanitiser (AIOpsDoom mitigation)
      ├── policy/         risk tiers, the human approval gate
      ├── remediate/      executors; every mutation carries its inverse
      ├── verify/         post-fix health loop + automatic rollback
      └── store/          incidents, settings, audit log
      │
      ▼
demo-stack/               injectable "production": app + Loki/Prometheus + fault injector
```

**Three axes of neutrality.** The system is not locked to one telemetry vendor
(`telemetry/`), one model vendor (`llm/`), or one chat surface (`chat/`). A
surface renders `Card` objects and translates events; it holds no incident
logic, no policy decisions and no model calls. `chat/console.py` is ~130 lines
and is the reference for how small a surface should be.

**On Discord vs Teams.** Teams is the better enterprise surface — no SRE team
runs production approvals in Discord. Discord ships first because it needs no
tenant, no admin approval and no app registration, so the agent is usable in
ten minutes. The neutrality is in the interface, not the vendor. See
[docs/DISCORD_SETUP.md](docs/DISCORD_SETUP.md).

---

## Incident lifecycle

```
DETECTED → TRIAGING → DIAGNOSED → AWAITING_APPROVAL → REMEDIATING → VERIFYING ─┬→ RESOLVED
                          │                                                     │
                          └→ ESCALATED (abstained)                              └→ ROLLED_BACK
```

Two edges most systems do not have:

- **DIAGNOSED → ESCALATED.** If no hypothesis cites evidence we actually
  collected, or the model reached a conclusion without calling a single tool,
  the agent abstains and escalates. Abstaining is a correct answer.
- **VERIFYING → ROLLED_BACK.** If health does not hold for the full window, the
  inverse action runs without waiting to be asked.

---

## Safety invariants

**1. Evidence grounding.** A hypothesis with no citations is dropped. A hypothesis
citing an evidence id we never collected is also dropped — a fabricated citation
buys no credibility. Zero tool calls forces abstention outright.

> Cloud-OpsBench (2026): a diagnosis was asserted with **zero tool calls in 32%**
> of cases. ORCA-bench (2026): **~40%** implausible-cause rate on weaker models.

**2. Reversibility.** `ProposedAction` refuses to construct if it mutates state
and declares no rollback. Irreversible actions cannot reach the approval card.

**3. Injection resistance.** Telemetry never enters a system message. It is
sanitised, fenced, and marked untrusted. Role and fence tokens are neutralised so
a log line cannot forge a turn boundary. Suspicious content is **tainted, not
deleted** — an attacker's payload is itself evidence of an attack. Credentials are
redacted before anything leaves the machine.

> AIOpsDoom (RSAC 2025): **90% success over 180 trials**, defeating both
> PromptShields and PromptGuard-2. The claim here is not better classification than
> the systems it beat — it is structural defences that hold when detection fails,
> backed by a human approval gate that no telemetry can bypass.

---

## Status

| Component | State |
|---|---|
| Domain contracts + invariants | ✅ done |
| Telemetry sanitiser | ✅ done, verified against live attacker telemetry |
| LLM provider layer (OpenAI-compatible / Anthropic / Gemini) | ✅ done |
| Provider registry + presets | ✅ done |
| Demo stack (7 scenarios + ground truth) | ✅ done, real OOM kill verified |
| Telemetry adapter protocol | ✅ done |
| Docker / SSH / Loki adapters | ✅ done (SSH includes host discovery) |
| Policy + approval gate | ✅ done |
| Remediation executors | ✅ done |
| **Verification loop + auto-rollback** | ✅ **done — proven end to end** |
| Chat surface protocol (`ChatSurface`, `Card`) | ✅ done |
| Discord surface (slash commands + button approvals) | ✅ done |
| Console surface | ✅ done |
| Agent reasoning loop (LLM ↔ tools) | ⬜ next |
| Settings UI | ⬜ next |
| Teams surface | ⬜ optional — a third implementation of a settled interface |

**98 tests passing** in 2.8s, plus an end-to-end proof against the live stack.

## The closed loop, proven

`agent-core/scripts/closed_loop_demo.py` runs the same incident twice against
the real demo stack:

```
ROUND A — DECOY FIX: restart the container
  poll  1  [XX.]  fail
  poll  2  [...]  ALL PASS        <-- a one-shot checker declares victory HERE
  poll  3  [XX.]  fail
  ...
  VERDICT   NOT_RECOVERED
  SUMMARY   the system became healthy and then relapsed within the 70s window.
            This is the signature of a fix that suppresses the symptom without
            addressing the cause.
  INCIDENT  state=rolled_back  rollback_triggered=True

ROUND B — CORRECT FIX: remove the defect at source
  poll  1..10  [...]  ALL PASS
  VERDICT   RECOVERED
  SUMMARY   health held continuously for 63s to the end of the 70s window.
  INCIDENT  state=resolved
```

Poll 2 of Round A is the entire argument. Restarting an OOM-killed container
clears every error instantly. Every surveyed vendor stops there and reports
success. Requiring health to **hold continuously to the end of a window** is
what separates a fix from a symptom suppressant.

The diagnosis in that script is hand-constructed on purpose — the guarantee
comes from the verification machinery, not from the model being clever.

### The four verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `RECOVERED` | health achieved and held to the end of the window | resolve |
| `NOT_RECOVERED` | never healthy, or relapsed after being healthy | roll back |
| `REGRESSED` | a check that passed **before** the fix now fails | roll back **immediately**, without waiting out the window |
| `INCONCLUSIVE` | nothing could be measured | escalate — an unmeasurable fix is not a fix |

`REGRESSED` is why a baseline is captured *before* remediation. Without a
pre-fix reading there is no way to distinguish "still broken" from "broken
differently because of what we just did".

---

## Running what exists

```bash
cd agent-core
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

Then the end-to-end proof (needs the demo stack up — see
[demo-stack/README.md](demo-stack/README.md)):

```bash
.venv/Scripts/python.exe scripts/closed_loop_demo.py
```

## LLM configuration

Any endpoint, set at runtime in Settings — no code change and no restart.
Presets ship for Anthropic, OpenAI, Azure OpenAI, Gemini, Ollama, LM Studio,
vLLM, OpenRouter, Groq, Together and DeepSeek; **Custom** accepts any base URL
that serves `POST /chat/completions`.

Several providers can be configured at once, which makes comparison cheap: run
the same injected incidents across several models and report honest numbers.
