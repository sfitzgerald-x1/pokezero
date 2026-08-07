| position | depth | sims | sims/s | ms/decision | decision nodes | chance nodes | leaf evals | deep-KO triggers |
|---|---|---|---|---|---|---|---|---|
| minimal_1v1 | 4 | 1024 | 3,231,964 | 0.32 | 32 | 89 | 132 | 0 |
| midgame_3v3 | 4 | 1024 | 411,986 | 2.49 | 292 | 1024 | 4147 | 18 |
| endgame_straddle | 4 | 1024 | 4,016,476 | 0.25 | 28 | 84 | 55 | 16 |

Argmax stability across seeds 0..4 (sims=1024):

| position | depth | argmax per seed | consistent | matches depth-1 |
|---|---|---|---|---|
| minimal_1v1 | 4 | ember, ember, tackle, tackle, ember | NO | no |
| midgame_3v3 | 4 | earthquake, earthquake, earthquake, earthquake, earthquake | yes | no |
| endgame_straddle | 4 | tackle, tackle, tackle, tackle, tackle | yes | no |
