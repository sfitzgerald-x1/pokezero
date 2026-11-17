"""The commit whose gpu_engine/transition.py the light component covers certify as current.

Main (#1819) certified b8092a85. Integration's reviewed Fidelity A1, A2, R and B batches changed
transition.py after it. 0a3711dc is the integration commit main merged into, and the merge left
transition.py unchanged. A reviewed transition.py change moves this constant in the same pull request.
Fidelity M1 (#1926: Hyper Beam's recharge and Trick) moved it to 3a05573e, its transition.py commit.
Fidelity M2 (Perish Song, Yawn, Wrap and the residual scheduler) moved it to 7cc1d9f6, its transition.py commit.
M2's R1 (the Wrap and Yawn residual handlers traced only at their reachable positions) moved it to 703abcb8.
Fidelity WH (White Herb, Sacred Fire's thaw and Facade) merged with R1 moved it to a19958b5, the merge's transition.py;
its review fixes (Trick's received herb, fewer traced triggers) moved it to 91346285, and the B2 Trick-credit rule
to 13f76e5e. M2's R2 (the older residual handlers traced only at their reachable positions) moved it to a8c48fa7, and its
Encore stale-move clear to 7c984fe6. Transform (the copy, its reverts, the Update cure and the choice unlock) moved it to
e4fc7e8e, its last transition.py commit (whole-leaf masked selects and one faint revert at the step's end).
The damage modifiers (Silk Scarf, Twisted Spoon, Marvel Scale and Gen 3 Plus/Minus) moved it to 61b00f72.
The combined Transform + transition batch + Tier D train moves it to 87e75985: interception reveals, damage credit and ability effects.
Train B composes the damage modifiers with tape v4 and the Transform/tier-D train at 3029a7e6.
Train B's combined Stick tree moves it to b3b3b6cd: both critical call sites retain current item/species and both planners retain the Stick cut.
"""
CURRENT_TRANSITION = 'b3b3b6cd52a0ac0194381e45f7c4e02fb99c2bb3'
