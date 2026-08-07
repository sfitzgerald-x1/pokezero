| position | depth | sims | sims/s | ms/decision | decision nodes | chance nodes | leaf evals | deep-KO triggers |
|---|---|---|---|---|---|---|---|---|
| minimal_1v1 | 4 | 1024 | 17,606 | 58.16 | 362 | 708 | 3712 | 0 |
| midgame_3v3 | 4 | 1024 | 110 | 9327.16 | 665 | 1024 | 950803 | 7 |
| endgame_straddle | 4 | 1024 | 741,563 | 1.38 | 191 | 421 | 1312 | 17 |

Argmax stability across seeds 0..4 (sims=1024):

| position | depth | argmax per seed | consistent | matches depth-1 |
|---|---|---|---|---|
| minimal_1v1 | 4 | tackle, tackle, tackle, tackle, tackle | yes | no |
| midgame_3v3 | 4 | earthquake, earthquake, earthquake, earthquake, earthquake | yes | no |
| endgame_straddle | 4 | tackle, tackle, tackle, tackle, tackle | yes | no |
