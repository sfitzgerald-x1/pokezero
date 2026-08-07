"""The denominator rule, shared by every differential harness.

A differential harness answers "how many boundaries diverged". That number is meaningless
without the denominator it came from, and every harness in this repo has at some point
reported a pass it had no ability to withhold:

- ``leaf_vs_reality.py`` read the observation schema off a key the corpus rows do not carry,
  so the encoder's guard rejected **100% of boundaries** as ``skip:encode_error``. Skips print
  in a separate column from the exit-code-gating class, so the run still printed
  ``DEFECT-CLASS divergent boundaries: 0`` and read as a pass (C112).
- ``leaf_root_parity.py`` exits on ``all(divergent == 0)`` and ``prior_mapping_assert.py`` on
  ``all(mismatch == 0)``. Both are **vacuously true on zero rows**.
- ``fidelity_gate_events.py`` ended ``return 0`` unconditionally — a gate that could not fail.

So the rule this module mechanizes:

1. publish ``boundaries_measured`` — the count the harness actually compared, not the count the
   corpus contains;
2. **hard-fail when it is zero**, because a run that measured nothing is not a pass; and
3. assert ``matched + diverged == measured``, so every attempt is accounted for.

**Why (3) is not a tautology, which is the trap this module exists to avoid.** In
``leaf_vs_reality`` the pre-existing identity ``compared == exact + divergent`` held *by
definition* — ``compared`` was assigned that sum — so it carried no information and was
nonetheless cited as evidence. Here ``measured`` must be supplied INDEPENDENTLY: the harness
increments it once per boundary it attempts to compare, before it knows the outcome. The
identity then genuinely catches a boundary that was attempted and never classified, or
classified twice. A caller that passes ``measured=matched+diverged`` gets the tautology back
and :func:`check_denominator` says so rather than passing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenominatorReport:
    """One harness run's denominator, ready to publish and to gate on."""

    label: str
    measured: int
    matched: int
    diverged: int
    contained: int | None = None
    skipped: int | None = None

    @property
    def failures(self) -> tuple[str, ...]:
        """Every rule violated, in the order the rules are numbered. Empty means clean."""
        out: list[str] = []
        if self.measured == 0:
            out.append(
                f"{self.label}: boundaries_measured == 0 — the run compared nothing, so its "
                f"result is not a pass. "
                + (
                    f"The corpus contains {self.contained} boundaries"
                    + (f" and {self.skipped} were skipped." if self.skipped is not None else ".")
                    if self.contained is not None
                    else "Check the skip reasons."
                )
            )
        total = self.matched + self.diverged
        if total != self.measured:
            out.append(
                f"{self.label}: matched + diverged ({self.matched} + {self.diverged} = {total}) "
                f"!= boundaries_measured ({self.measured}) — "
                f"{abs(self.measured - total)} attempted boundaries were "
                + ("classified twice" if total > self.measured else "never classified")
                + "."
            )
        return tuple(out)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def inert(self) -> bool:
        """True when the run cannot fail on divergence because it measured nothing.

        Distinct from ``ok``: an inert run is reported AND gated, because silence about
        inertness is the failure mode this module exists to remove.
        """
        return self.measured == 0

    def render(self) -> str:
        """The one line every harness prints, so the denominator is never implicit."""
        parts = [f"boundaries_measured {self.measured}"]
        if self.contained is not None:
            parts.append(f"of {self.contained} contained")
        if self.skipped is not None:
            parts.append(f"{self.skipped} skipped")
        parts.append(f"matched {self.matched}")
        parts.append(f"diverged {self.diverged}")
        return f"   [denominator] {self.label}: " + ", ".join(parts)


def check_denominator(
    label: str,
    *,
    measured: int,
    matched: int,
    diverged: int,
    contained: int | None = None,
    skipped: int | None = None,
) -> DenominatorReport:
    """Build a report. `measured` must be counted independently of `matched`/`diverged`.

    Passing ``measured=matched+diverged`` makes rule 3 unfalsifiable. That is not detectable
    from the values alone — ``5 == 2 + 3`` looks identical either way — so it cannot be
    asserted here; it is a contract on the caller, stated in the module docstring and in each
    adoption site's comment. What IS enforced is that a zero denominator fails, which is the
    half no caller can fake.
    """
    return DenominatorReport(
        label=label,
        measured=int(measured),
        matched=int(matched),
        diverged=int(diverged),
        contained=None if contained is None else int(contained),
        skipped=None if skipped is None else int(skipped),
    )


def gate(reports: list[DenominatorReport]) -> int:
    """Print every report, then return the exit code the denominator rule alone implies.

    Callers OR this with their own defect gates; it never lowers an exit code. An empty list
    is itself a failure — a harness that produced no reports measured nothing.
    """
    if not reports:
        print("   [denominator] FAIL: the run produced no corpora reports at all.")
        return 1
    failed = False
    for report in reports:
        print(report.render())
        for reason in report.failures:
            print(f"   [denominator] FAIL: {reason}")
            failed = True
    return 1 if failed else 0
