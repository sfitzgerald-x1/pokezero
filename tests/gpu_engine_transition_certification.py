"""The commit whose gpu_engine/transition.py the light component covers certify as current.

Main (#1819) certified b8092a85. Integration's reviewed Fidelity A1, A2, R and B batches changed
transition.py after it. 0a3711dc is the integration commit main merged into, and the merge left
transition.py unchanged. A reviewed transition.py change moves this constant in the same pull request.
Fidelity M1 (#1926: Hyper Beam's recharge and Trick) moved it to 3a05573e, its transition.py commit.
Fidelity M2 (Perish Song, Yawn, Wrap and the residual scheduler) moves it to 7cc1d9f6, its transition.py commit.
"""
CURRENT_TRANSITION = '7cc1d9f68e370f6e1676f4ae4898ec453cdce923'
