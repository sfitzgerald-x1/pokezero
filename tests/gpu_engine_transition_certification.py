"""The commit whose gpu_engine/transition.py the light component covers certify as current.

Main (#1819) certified b8092a85. Integration's reviewed Fidelity A1, A2, R and B batches changed
transition.py after it. 0a3711dc is the integration commit main merged into, and the merge left
transition.py unchanged. A reviewed transition.py change moves this constant in the same pull request.
Fidelity M1 (#1926: Hyper Beam's recharge and Trick) moved it to 3a05573e, its transition.py commit.
Fidelity M2 (Perish Song, Yawn, Wrap and the residual scheduler) moved it to 7cc1d9f6, its transition.py commit.
M2's R1 (the Wrap and Yawn residual handlers traced only at their reachable positions) moves it to 703abcb8.
"""
CURRENT_TRANSITION = '703abcb8d86e8f86588fe14b2e125c93278a0c80'
