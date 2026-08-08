"""The denominator rule, adopted by four differential harnesses.

A differential harness answers "how many boundaries diverged". That number is meaningless
without the denominator it came from, and each of the four harnesses adopted here could at
some point report a pass it had no ability to withhold:

- ``leaf_vs_reality.py`` read the observation schema off a key the corpus rows do not carry,
  so the encoder's guard rejected **100% of boundaries** as ``skip:encode_error``. Skips print
  in a separate column from the exit-code-gating class, so the run still printed
  ``DEFECT-CLASS divergent boundaries: 0`` and read as a pass (C112).
- ``leaf_root_parity.py`` exits on ``all(divergent == 0)`` and ``prior_mapping_assert.py`` on
  ``all(mismatch == 0)``. Both are **vacuously true on zero rows**.
- ``fidelity_gate_events.py`` ended ``return 0`` unconditionally — a gate that could not fail.

``scripts/`` holds roughly a dozen other ``*_differential.py`` and ``*_gate.py`` scripts. The four
above are the ones adopted here; the claim is about them, not about every harness in the repo.

So the rule this module mechanizes:

1. publish ``boundaries_measured`` — the count the harness actually compared, not the count the
   harness's own universe holds -- see the note on ``contained`` below;
2. **hard-fail when it is zero**, because a run that measured nothing is not a pass; and
3. assert ``matched + diverged == measured``, so every attempt is accounted for; and
4. assert ``measured + skipped == contained`` whenever both are supplied — the rule that
   actually bites.

**What ``contained`` means, precisely, because the label overstates it.** It is whatever the
caller counts as its universe, and that is NOT the same unit at all four sites:
``leaf_vs_reality`` and ``fidelity_gate_events`` pass same-seat BOUNDARIES (1271 on golden-v4),
while ``leaf_root_parity`` and ``prior_mapping_assert`` pass decision ROWS (1295). Both are right
for their own harness — rule 4 only requires ``measured`` and ``skipped`` to be counted in the
same unit as ``contained`` — and the failure message no longer says "boundaries contained", which was
wrong for two of the four, and it is the HARNESS's universe, not the corpus's.

**What rule 3 actually does, stated narrowly after review.** In ``leaf_vs_reality`` the
pre-existing identity ``compared == exact + divergent`` held *by definition* — ``compared`` was
assigned that sum — so it carried no information and was nonetheless cited as evidence. This
module asks callers to supply ``measured`` independently, and they do.

But **as adopted, rule 3 is currently unfalsifiable at all four sites**: in each one the
increment is followed by exactly one classification with no intervening early return, so
``matched + diverged == measured`` holds structurally. It is retained as a guard against a
future refactor that introduces an unclassified path, **not** as evidence that the present
accounting is sound — and "the partition closes on all four" is therefore a statement about
the code's shape, not a measurement. Reading it as verification would be the
instrument-that-cannot-move error this repo has made repeatedly.

**Rule 4 is the falsifiable one, and it exists because rule 3 is not.** Rule 3 sits entirely
on one side of the skip decision, so an increment adjacent to its classification can never
violate it. Rule 4 SPANS that decision: every unit in the harness's universe was either compared
or skipped, exactly once. It mechanically detects both bugs review found in this PR's first
round, neither of which needed a forced-skip driver or a fixture to catch:

- the counter placed above the skip decision (``fidelity_gate_events``): 1271 + 315 = 1586
  against 1271 contained → FAIL;
- a boundary counted into BOTH ``attempted`` and ``skipped`` (``leaf_vs_reality``'s
  ``no_golden_row`` path): 956 + 316 = 1272 against 1271 → FAIL.

It is skipped when either figure is absent, so callers that cannot supply them still work —
but supplying them turns ``contained`` and ``skipped`` from display strings into load-bearing
arguments, which is the point.

Rules 2 and 4 are the ones doing work. Rule 2 also cannot be faked by deriving ``measured``
from the sum: if nothing was measured, the sum is zero too.
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
                    f"The harness counted {self.contained} in its universe"
                    + (f" and {self.skipped} were skipped." if self.skipped is not None else ".")
                    if self.contained is not None
                    else "Check the skip reasons."
                )
            )
        if self.contained is not None and self.skipped is not None:
            accounted = self.measured + self.skipped
            if accounted != self.contained:
                out.append(
                    f"{self.label}: boundaries_measured + skipped ({self.measured} + "
                    f"{self.skipped} = {accounted}) != the harness's universe "
                    f"({self.contained}) — {abs(self.contained - accounted)} boundaries were "
                    + (
                        "counted twice (compared AND skipped)"
                        if accounted > self.contained
                        else "neither compared nor skipped"
                    )
                    + "."
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
        if self.contained is None or self.skipped is None:
            # Say so in the output. A downgraded run should not be inferable only from the
            # ABSENCE of a phrase -- that is the silent-inertness shape this module exists
            # to remove.
            parts.append("rule 4 NOT CHECKED (contained/skipped not supplied)")
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
    """Build a report. `measured` should be counted independently of `matched`/`diverged`.

    Passing ``measured=matched+diverged`` makes rule 3 unfalsifiable. That is not detectable
    from the values alone — ``5 == 2 + 3`` looks identical either way — so it cannot be
    asserted here; it is a contract on the caller, stated in the module docstring and in each
    adoption site's comment. What IS enforced is that a zero denominator fails.

    Even that half can be defeated by a caller that counts in the wrong place rather than by
    deriving the value: ``fidelity_gate_events`` first incremented before its skip decision,
    making ``measured`` identically the corpus's boundary count, and a 100%-skipped run still
    exited 0. The counter's PLACEMENT is as load-bearing as its independence, which is why
    each adoption site carries a comment naming where it sits and why.
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
