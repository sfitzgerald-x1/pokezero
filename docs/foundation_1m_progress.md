# Foundation 1M continuation progress

Status: live progress record for the recipe-fidelity continuation from the completed
**500,800-game** foundation checkpoint toward **1,000,000 self-play games**.

This note records evaluation evidence only. It intentionally omits private operational details.

## Evaluation cadence

- The 500k anchor row comes from [`foundation_500k_results.md`](foundation_500k_results.md).
- Continuation rows use the active 1M run's scheduled readouts.
- Continuation non-foul-play opponents are mirrored **400-game** aggregate reads.
- Continuation foul-play is a **100-game** async read at the same scheduled milestone.
- Larger independent 1,000-game reads are reserved for 50k milestones and should be added below as
  they complete.

## Standard 10k Progress

| Total self-play games | Checkpoint | Random-legal | Simple-legal | Max-damage | Foul-play |
|---:|---|---:|---:|---:|---:|
| 500,800 | 500k anchor, iteration 313 | 592 / 600 (98.7%) | 554 / 600 (92.3%) | 309 / 600 (51.5%) | 37 / 1000 (3.7%) |
| 502,400 | continuation iteration 1 | 396 / 400 (99.0%) | 352 / 400 (88.0%) | 213 / 400 (53.2%) | 5 / 100 (5.0%) |
| 510,400 | continuation iteration 6 | 390 / 400 (97.5%) | 364 / 400 (91.0%) | 220 / 400 (55.0%) | 2 / 100 (2.0%) |
| 520,000 | continuation iteration 12 | 397 / 400 (99.2%) | 370 / 400 (92.5%) | 200 / 400 (50.0%) | 3 / 100 (3.0%) |
| 531,200 | continuation iteration 19 | 397 / 400 (99.2%) | 374 / 400 (93.5%) | 210 / 400 (52.5%) | 5 / 100 (5.0%) |
| 540,800 | continuation iteration 25 | 393 / 400 (98.2%) | 367 / 400 (91.8%) | 212 / 400 (53.0%) | 6 / 100 (6.0%) |
| 550,400 | continuation iteration 31 | 395 / 400 (98.8%) | 371 / 400 (92.8%) | 221 / 400 (55.2%) | 3 / 100 (3.0%) |
| 560,000 | continuation iteration 37 | 396 / 400 (99.0%) | 363 / 400 (90.8%) | 204 / 400 (51.0%) | 3 / 100 (3.0%) |
| 571,200 | continuation iteration 44 | 393 / 400 (98.2%) | 369 / 400 (92.2%) | 215 / 400 (53.8%) | 5 / 100 (5.0%) |
| 580,800 | continuation iteration 50 | 393 / 400 (98.2%) | 367 / 400 (91.8%) | 215 / 400 (53.8%) | 5 / 100 (5.0%) |
| 590,400 | continuation iteration 56 | 396 / 400 (99.0%) | 379 / 400 (94.8%) | 209 / 400 (52.2%) | 1 / 100 (1.0%) |
| 600,000 | continuation iteration 62 | 397 / 400 (99.2%) | 365 / 400 (91.2%) | 239 / 400 (59.8%) | 3 / 100 (3.0%) |
| 611,200 | continuation iteration 69 | 396 / 400 (99.0%) | 370 / 400 (92.5%) | 223 / 400 (55.8%) | 2 / 100 (2.0%) |
| 620,800 | continuation iteration 75 | 398 / 400 (99.5%) | 368 / 400 (92.0%) | 210 / 400 (52.5%) | 3 / 100 (3.0%) |
| 630,400 | continuation iteration 81 | 395 / 400 (98.8%) | 375 / 400 (93.8%) | 230 / 400 (57.5%) | 4 / 100 (4.0%) |
| 640,000 | continuation iteration 87 | 400 / 400 (100.0%) | 372 / 400 (93.0%) | 239 / 400 (59.8%) | 6 / 100 (6.0%) |
| 651,200 | continuation iteration 94 | 394 / 400 (98.5%) | 380 / 400 (95.0%) | 233 / 400 (58.2%) | 3 / 100 (3.0%) |
| 660,800 | continuation iteration 100 | 397 / 400 (99.2%) | 381 / 400 (95.2%) | 232 / 400 (58.0%) | 5 / 100 (5.0%) |
| 670,400 | continuation iteration 106 | 398 / 400 (99.5%) | 380 / 400 (95.0%) | 239 / 400 (59.8%) | 1 / 100 (1.0%) |
| 680,000 | continuation iteration 112 | 398 / 400 (99.5%) | 386 / 400 (96.5%) | 233 / 400 (58.2%) | 5 / 100 (5.0%) |
| 691,200 | continuation iteration 119 | 396 / 400 (99.0%) | 368 / 400 (92.0%) | 221 / 400 (55.2%) | 6 / 100 (6.0%) |
| 700,800 | continuation iteration 125 | 398 / 400 (99.5%) | 376 / 400 (94.0%) | 241 / 400 (60.2%) | 9 / 100 (9.0%) |
| 710,400 | continuation iteration 131 | 395 / 400 (98.8%) | 376 / 400 (94.0%) | 242 / 400 (60.5%) | 5 / 100 (5.0%) |
| 720,000 | continuation iteration 137 | 392 / 400 (98.0%) | 375 / 400 (93.8%) | 245 / 400 (61.2%) | 10 / 100 (10.0%) |
| 731,200 | continuation iteration 144 | 395 / 400 (98.8%) | 379 / 400 (94.8%) | 245 / 400 (61.2%) | 4 / 100 (4.0%) |
| 740,800 | continuation iteration 150 | 396 / 400 (99.0%) | 378 / 400 (94.5%) | 232 / 400 (58.0%) | 1 / 100 (1.0%) |
| 750,400 | continuation iteration 156 | 394 / 400 (98.5%) | 377 / 400 (94.2%) | 231 / 400 (57.8%) | 6 / 100 (6.0%) |
| 760,000 | continuation iteration 162 | 398 / 400 (99.5%) | 373 / 400 (93.2%) | 241 / 400 (60.2%) | 4 / 100 (4.0%) |
| 771,200 | continuation iteration 169 | 395 / 400 (98.8%) | 372 / 400 (93.0%) | 239 / 400 (59.8%) | 7 / 100 (7.0%) |
| 780,800 | continuation iteration 175 | 398 / 400 (99.5%) | 377 / 400 (94.2%) | 250 / 400 (62.5%) | 3 / 100 (3.0%) |
| 790,400 | continuation iteration 181 | 395 / 400 (98.8%) | 379 / 400 (94.8%) | 235 / 400 (58.8%) | 5 / 100 (5.0%) |
| 800,000 | continuation iteration 187 | 394 / 400 (98.5%) | 376 / 400 (94.0%) | 237 / 400 (59.2%) | 1 / 100 (1.0%) |
| 811,200 | continuation iteration 194 | 395 / 400 (98.8%) | 380 / 400 (95.0%) | 251 / 400 (62.7%) | 4 / 100 (4.0%) |
| 820,800 | continuation iteration 200 | 398 / 400 (99.5%) | 382 / 400 (95.5%) | 256 / 400 (64.0%) | 3 / 100 (3.0%) |
| 830,400 | continuation iteration 206 | 393 / 400 (98.2%) | 379 / 400 (94.8%) | 242 / 400 (60.5%) | 6 / 100 (6.0%) |
| 840,000 | continuation iteration 212 | 397 / 400 (99.2%) | 384 / 400 (96.0%) | 256 / 400 (64.0%) | 7 / 100 (7.0%) |

The 502,400 row is the first checkpoint after resuming from the 500,800-game model. It is useful as
an initial continuation baseline, but it is closer to a startup read than a regular 10k interval.
The foul-play anchor is a higher-fidelity 1,000-game read; continuation foul-play rows are the
scheduled 100-game milestone reads.

## High-Fidelity 50k Progress

Continuation high-fidelity rows use independent **1,000-game-per-seat mirrored** reads for
random/simple/max-damage, for 2,000 aggregate games per matchup, and an independent **1,000-game**
foul-play read. Rows are added only once all four opponents have complete results, so partial
milestones are not mixed into the trend table.

| Total self-play games | Checkpoint | Random-legal | Simple-legal | Max-damage | Foul-play |
|---:|---|---:|---:|---:|---:|
| 550,400 | continuation iteration 31 | 1,981 / 2,000 (99.1%) | 1,872 / 2,000 (93.6%) | 1,091 / 2,000 (54.5%) | 37 / 1,000 (3.7%) |
| 600,000 | continuation iteration 62 | 1,975 / 2,000 (98.8%) | 1,868 / 2,000 (93.4%) | 1,116 / 2,000 (55.8%) | 59 / 1,000 (5.9%) |
| 750,400 | continuation iteration 156 | 1,974 / 2,000 (98.7%) | 1,899 / 2,000 (95.0%) | 1,193 / 2,000 (59.7%) | 42 / 1,000 (4.2%) |
| 800,000 | continuation iteration 187 | 1,979 / 2,000 (99.0%) | 1,887 / 2,000 (94.3%) | 1,229 / 2,000 (61.5%) | 47 / 1,000 (4.7%) |

## Current Readout

The continuation rows so far are constructive for the current recipe. Max-damage remains noisy, but
it is not collapsed, and the scheduled readouts have moved from the low/mid-50s around the 500k
anchor into a high-50s/low-60s band. From 630k through 840k, scheduled max-damage reads mostly held
in that band. The 690k scheduled row dipped back to 55.2%, similar to the earlier 610k row at 55.8%,
but the 700k scheduled row rebounded to 60.2%, the 710k row held at 60.5%, and the 720k and 730k rows
both held at 61.2%. The 740k row dipped to 58.0%, the 750k row held nearby at 57.8%, the 760k row
rebounded to 60.2%, and the 770k scheduled row held near that rebound at 59.8%. The 780k scheduled
row then reached 62.5%, the strongest scheduled max-damage read up to that point, before the 790k row
returned to 58.8% and the 800k row held nearby at 59.2%. The 810k scheduled row then reached 62.7%,
the 820k scheduled row reached 64.0%, the 830k row stayed in the low-60s at 60.5%, and the 840k row
matched that 64.0% mark. The 820k and 840k rows are the strongest scheduled max-damage reads in the
continuation so far.

That broader upward drift against max-damage is the leading signal for this phase. The
MIT-inspired recipe expected meaningful progress to require substantially more than 500k games; see
[`foundation_500k_results.md`](foundation_500k_results.md) for the anchor readout. The current
evidence therefore supports continuing toward the 1M readout rather than treating the recipe as
exhausted. The completed 600k, 750k, and 800k high-fidelity rows show the same rising max-damage
shape: max-damage was 1,116 / 2,000 (55.8%) at 600k, 1,193 / 2,000 (59.7%) at 750k, and
1,229 / 2,000 (61.5%) at 800k, while foul-play remained a harder downstream bar at
59 / 1,000 (5.9%), 42 / 1,000 (4.2%), and 47 / 1,000 (4.7%). The 600k high-fidelity max-damage row
was below the co-located 600k scheduled read's 59.8%, so individual scheduled rows should still be
treated as noisy. The later high-fidelity reads support the high-50s/low-60s max-damage trend, but
the higher-fidelity reads remain the cleaner trend signal.

Foul-play remains the higher-quality benchmark because it is a stronger opponent than max-damage, and
it is expected to beat PokeZero until the policy is substantially stronger. Low early foul-play scores
should be treated as a lagging high-bar readout, not as negative evidence while max-damage is holding
or climbing. The working expectation is that foul-play progress may be delayed and nonlinear: it can
stay low until the model crosses a practical competence threshold, then move more sharply. The
interpretation remains falsifiable: if larger-scale max-damage reads stall or collapse, the recipe
should be reassessed, and any isolated foul-play uptick should be corroborated by continued
max-damage strength before treating it as a durable breakthrough. The next meaningful check is
whether max-damage holds or improves as the run crosses later 10k thresholds, and whether complete
higher-fidelity 50k reads show the same trend while foul-play remains tracked for delayed movement.

## Next Updates

- Add the next scheduled 10k read once the run crosses the next threshold.
- Include the refreshed 10k trajectory plot with the next public progress report.
- Add the 650k independent high-fidelity row once all four opponents complete.
- Add the 700k independent high-fidelity row once all four opponents complete.
