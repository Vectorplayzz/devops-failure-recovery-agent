"""Chat surface tests.

Two things are being defended here:

  1. Surface neutrality is real - the same Card renders on two independent
     surfaces with no incident logic in either.
  2. The approval identity rule survives the surface boundary. An approval
     carries a platform-authenticated principal, never text.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from opsloop.chat.base import (
    ApprovalEvent,
    Card,
    ChatUser,
    Field,
    Tone,
    approval_card,
    diagnosis_card,
    incident_card,
    truncate,
    verification_card,
)
from opsloop.chat.console import ConsoleSurface, render
from opsloop.chat.discord_bot import (
    MAX_EMBED_TITLE,
    MAX_FIELD_VALUE,
    MAX_FIELDS,
    card_to_embed,
)
from opsloop.domain.models import (
    ActionSpec,
    ApprovalDecision,
    CheckOutcome,
    Diagnosis,
    EvidenceRef,
    Hypothesis,
    Incident,
    IncidentState,
    ProposedAction,
    RiskLevel,
    Severity,
    VerificationPlan,
    VerificationReport,
    VerificationVerdict,
)
from opsloop.policy.engine import PolicyEngine

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def ev(ev_id: str = "ev_1", *, tainted: bool = False) -> EvidenceRef:
    return EvidenceRef(
        id=ev_id,
        source_kind="docker",
        source_name="demo",
        query="docker inspect demo-orders-api",
        observed_at=NOW,
        excerpt="OOMKilled=true",
        raw_sha256="0" * 64,
        tainted=tainted,
        taint_reasons=["fake_approval"] if tainted else [],
    )


def incident_with_diagnosis(*, tainted: bool = False, abstained: bool = False) -> Incident:
    inc = Incident(title="orders-api is down", severity=Severity.SEV1, service="orders-api")
    e = ev(tainted=tainted)
    diag = Diagnosis(
        incident_id=inc.id,
        model_id="test-model",
        evidence=[e],
        tool_call_count=4,
        hypotheses=[
            Hypothesis(
                statement="orders-api OOM-killed at its 256MB limit",
                mechanism="heap grows past the cgroup limit under load",
                confidence=0.86,
                evidence_ids=[e.id],
            ),
            Hypothesis(
                statement="DNS is broken",  # uncited: must be shown as discarded
                mechanism="speculation",
                confidence=0.9,
                evidence_ids=[],
            ),
        ],
    )
    if abstained:
        diag.abstained = True
        diag.abstain_reason = "evidence too thin"
    inc.diagnosis = diag
    return inc


def sample_action() -> ProposedAction:
    return ProposedAction(
        intent="Restart the OOM-killed orders-api container",
        forward=ActionSpec(
            executor="docker.start", params={"container": "demo-orders-api"}, target="demo"
        ),
        rollback=ActionSpec(
            executor="docker.stop", params={"container": "demo-orders-api"}, target="demo"
        ),
        risk=RiskLevel.LOW,
        rationale="The container is exited.",
        evidence_ids=["ev_1"],
        expected_effect="Service returns.",
        blast_radius="orders-api only.",
    )


class TestTruncation:
    def test_short_text_untouched(self) -> None:
        assert truncate("hello", 100) == "hello"

    def test_long_text_clamped_within_limit(self) -> None:
        out = truncate("x" * 5000, 100)
        assert len(out) <= 100
        assert out.endswith("[truncated]")

    def test_every_field_fits_discord_limits(self) -> None:
        """A long stack trace must not cost us the whole incident alert."""
        inc = incident_with_diagnosis()
        inc.diagnosis.hypotheses[0].mechanism = "y" * 9000
        card = incident_card(inc)
        assert all(len(f.value) <= MAX_FIELD_VALUE for f in card.fields)
        assert len(card.title) <= MAX_EMBED_TITLE


class TestIncidentCard:
    def test_leading_hypothesis_and_confidence_shown(self) -> None:
        card = incident_card(incident_with_diagnosis())
        text = " ".join(f"{f.name} {f.value}" for f in card.fields)
        assert "OOM-killed" in text
        assert "86%" in text

    def test_cited_evidence_ids_are_shown(self) -> None:
        card = incident_card(incident_with_diagnosis())
        assert any("ev_1" in f.value for f in card.fields)

    def test_abstention_is_surfaced_not_hidden(self) -> None:
        card = incident_card(incident_with_diagnosis(abstained=True))
        text = " ".join(f.value for f in card.fields)
        assert "Abstained" in text and "correct answer" in text

    def test_tainted_evidence_raises_a_security_field(self) -> None:
        card = incident_card(incident_with_diagnosis(tainted=True))
        names = " ".join(f.name for f in card.fields)
        values = " ".join(f.value for f in card.fields)
        assert "Security" in names
        assert "not as instructions" in values

    def test_resolved_incident_uses_success_tone(self) -> None:
        inc = incident_with_diagnosis()
        inc.state = IncidentState.RESOLVED
        assert incident_card(inc).tone is Tone.SUCCESS


class TestApprovalCard:
    def test_rollback_is_visible_before_approving(self) -> None:
        """A rollback discovered after the fact is not one anyone consented to."""
        card = approval_card(incident_with_diagnosis(), sample_action())
        rollback = next(f for f in card.fields if "Rollback" in f.name)
        assert "docker.stop" in rollback.value

    def test_blast_radius_and_effect_are_shown(self) -> None:
        card = approval_card(incident_with_diagnosis(), sample_action())
        names = " ".join(f.name for f in card.fields)
        assert "Blast radius" in names and "Expected effect" in names

    def test_buttons_carry_incident_and_action_ids(self) -> None:
        inc = incident_with_diagnosis()
        action = sample_action()
        card = approval_card(inc, action)
        assert [b.label for b in card.buttons] == ["Approve", "Reject"]
        assert card.buttons[0].action_id == f"approve:{inc.id}:{action.id}"
        assert card.buttons[1].action_id == f"reject:{inc.id}:{action.id}"

    def test_policy_warnings_switch_the_card_to_security_tone(self) -> None:
        inc = incident_with_diagnosis(tainted=True)
        action = sample_action()
        decision = PolicyEngine().evaluate(action, incident=inc)
        card = approval_card(inc, action, decision)
        assert card.tone is Tone.SECURITY
        assert any("SECURITY" in f.value for f in card.fields)


class TestDiagnosisCard:
    def test_discarded_hypotheses_are_shown(self) -> None:
        """Making the grounding rule visible is the point."""
        inc = incident_with_diagnosis()
        card = diagnosis_card(inc, inc.diagnosis)
        text = " ".join(f"{f.name} {f.value}" for f in card.fields)
        assert "Discarded" in text
        assert "DNS is broken" in text
        assert "no citation" in text


class TestVerificationCard:
    def test_rollback_is_reported(self) -> None:
        inc = incident_with_diagnosis()
        report = VerificationReport(
            plan=VerificationPlan(window_seconds=70),
            verdict=VerificationVerdict.NOT_RECOVERED,
            summary="relapsed within the window",
            rollback_triggered=True,
            outcomes=[CheckOutcome(check_id="c1", passed=False, observed="x")],
        )
        card = verification_card(inc, report)
        assert card.tone is Tone.CRITICAL
        assert any("reverted automatically" in f.value for f in card.fields)

    def test_recovered_uses_success_tone(self) -> None:
        report = VerificationReport(
            plan=VerificationPlan(), verdict=VerificationVerdict.RECOVERED, summary="ok"
        )
        assert verification_card(incident_with_diagnosis(), report).tone is Tone.SUCCESS


class TestDiscordRendering:
    def test_card_renders_as_an_embed(self) -> None:
        embed = card_to_embed(incident_card(incident_with_diagnosis()))
        assert embed.title == "orders-api is down"
        assert embed.colour.value == 0xD93025  # SEV1 -> critical red

    def test_field_overflow_is_summarised_not_dropped_silently(self) -> None:
        card = Card(
            title="many", fields=[Field(f"f{i}", f"v{i}") for i in range(40)]
        )
        embed = card_to_embed(card)
        assert len(embed.fields) == MAX_FIELDS + 1
        assert "more field(s) omitted" in embed.fields[-1].value

    def test_empty_field_value_does_not_break_the_embed(self) -> None:
        """Discord rejects an empty field value; a blank must be substituted."""
        embed = card_to_embed(Card(title="t", fields=[Field("name", "")]))
        assert embed.fields[0].value == "​"


class TestConsoleRendering:
    def test_same_card_renders_on_a_second_surface(self) -> None:
        """Surface neutrality, demonstrated rather than asserted."""
        card = incident_card(incident_with_diagnosis())
        text = render(card)
        assert "orders-api is down" in text
        assert "OOM-killed" in text

    def test_every_builder_renders_on_a_windows_console(self) -> None:
        """A Windows console is cp1252 and raises on anything outside it.

        This covers every card builder, with taint switched on, because the
        first version of this test used a bare Card and therefore missed a
        real crash: the security field carried a glyph cp1252 cannot encode.
        Emoji belong to a surface that supports them, not to the neutral
        layer that the console also has to render.
        """
        inc = incident_with_diagnosis(tainted=True)
        action = sample_action()
        decision = PolicyEngine().evaluate(action, incident=inc)
        report = VerificationReport(
            plan=VerificationPlan(),
            verdict=VerificationVerdict.REGRESSED,
            summary="made things worse",
            rollback_triggered=True,
        )

        cards = [
            incident_card(inc),
            incident_card(incident_with_diagnosis(abstained=True)),
            approval_card(inc, action, decision),
            diagnosis_card(inc, inc.diagnosis),
            verification_card(inc, report),
        ]
        for card in cards:
            render(card).encode("cp1252")  # raises on failure

    def test_neutral_card_text_carries_no_emoji(self) -> None:
        """Glyphs are a surface concern; the shared layer stays portable."""
        inc = incident_with_diagnosis(tainted=True)
        card = incident_card(inc)
        blob = card.title + card.body + card.footer
        blob += "".join(f.name + f.value for f in card.fields)
        assert all(ord(c) < 0x2000 for c in blob), (
            "a card builder emitted a symbol outside the portable range"
        )

    async def test_console_approval_carries_an_identity(self) -> None:
        captured: list[ApprovalEvent] = []

        async def handler(event: ApprovalEvent) -> str:
            captured.append(event)
            return "recorded"

        surface = ConsoleSurface(auto_approve=True)
        surface.on_approval(handler)
        inc = incident_with_diagnosis()
        action = sample_action()
        await surface.post(approval_card(inc, action))

        assert len(captured) == 1
        assert captured[0].approved is True
        assert captured[0].incident_id == inc.id
        assert captured[0].action_id == action.id
        # An identity is present on every surface - never blank, never typed.
        assert captured[0].user.id.startswith("local:")


class TestApprovalIdentityRule:
    def test_policy_refuses_an_approval_with_no_principal(self) -> None:
        """The rule is enforced twice: at the surface, and again at the gate."""
        engine = PolicyEngine()
        inc = incident_with_diagnosis()
        with pytest.raises(ValueError, match="authenticated user id"):
            engine.record_decision(
                inc, "act_1", decision=ApprovalDecision.APPROVED, user_id=""
            )

    def test_surface_identity_flows_into_the_approval_record(self) -> None:
        engine = PolicyEngine()
        inc = incident_with_diagnosis()
        action = sample_action()
        inc.proposed_actions.append(action)
        user = ChatUser(id="discord:249081", display_name="oncall-engineer")

        engine.record_decision(
            inc,
            action.id,
            decision=ApprovalDecision.APPROVED,
            user_id=user.id,
            user_name=user.display_name,
            channel="discord",
        )
        recorded = inc.approval_for(action.id)
        assert recorded.decided_by == "discord:249081"
        assert recorded.channel == "discord"
        assert inc.timeline[-1].actor == "user:oncall-engineer"
