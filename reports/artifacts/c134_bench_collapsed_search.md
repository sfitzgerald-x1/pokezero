| position | depth | sims | sims/s | ms/decision | decision nodes | chance nodes | leaf evals | deep-KO triggers |
|---|---|---|---|---|---|---|---|---|
| minimal_1v1 | 4 | 1024 | 3,067,373 | 0.33 | 37 | 87 | 140 | 0 |
| midgame_3v3 | 4 | 1024 | 389,888 | 2.63 | 282 | 1024 | 4189 | 17 |
| endgame_straddle | 4 | 1024 | 3,739,542 | 0.27 | 29 | 86 | 56 | 16 |

Argmax stability across seeds 0..4 (sims=1024):

| position | depth | argmax per seed | consistent | matches depth-1 |
|---|---|---|---|---|
| minimal_1v1 | 4 | ember, ember, tackle, tackle, ember | NO | no |
| midgame_3v3 | 4 | earthquake, earthquake, earthquake, earthquake, earthquake | yes | no |
| endgame_straddle | 4 | tackle, tackle, tackle, tackle, tackle | yes | no |
