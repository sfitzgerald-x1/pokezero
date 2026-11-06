# Phase D chance source binding

This bounded change starts from explicit historical baseline
`3d8114790beb49a931b809bd019ae9a8e57967ea`. It updates only two literal
bindings in `gpu_engine/phase_d_chance.py`; all consumer logic and production
files supplying that logic remain byte unchanged. Passing source fixtures is
not scientific qualification, fidelity, Stage 4, acting or latency admission.
The change remains pending the one producer review combined with the next
numerical Phase D producer.

The Phase D module was introduced in
`9a7848b4f034f6b3b0b702ee4a9e97e351eacbf7` with transition SHA256
`97086e6a630d602e5179b608a5d8d93d4d191c0b4559fb6dd546cfb0e267431e`.
Later commit `3f7a34f0149e1e66daf3d51b88d6046d8e7be550` changed the transition
without refreshing this module's bindings. That commit removed the optional
Ground/Levitate public ability-reveal notification and added optional public
HP notifications for self-heals, Rest and Substitute. Public observation
semantics changed; the chance sampler and consumer law did not.

The transition file is now bound to its existing frozen SHA256
`643de79452d7416ee52f841742f5d6a7c627a656f405e3998e67e0c530a05c5a`.
The combined 25-module domain changes from
`de58aa3ce6bc8df33be4894007f9a04bd714365289b5d5363a26700ff6223087` to
`9d2d21a53bd50ab2beb7b335841223b906328736584a130bfbecc8371d4094a2`.
Transition is the only changed module since the original domain; the other
12 direct source pins and 24 domain modules already matched.

Static comparison finds unchanged `ChanceInputs`, `sample_chance_inputs`,
`initialize`, `step`, `step_with_public` and all function signatures. Only
the `step_with_chance` body contains those public-hook edits. All 50 chance
attribute/subscript references match structurally. The same 33 coordinates
remain: two rows with bounds `(2,100,100,100,16,16,100,4,8,2,2,2)`; two each
of phazing/Static60, secondary100, Sleep Talk60 and entry ties2; and one initial
tie2. Reused draws remain joint, and saved ordinary draws and replacement-count
tie addressing retain their existing suspension semantics. No bounds, salts,
keys, predicates, mapper, constructor or tracker behavior is changed here.

The frozen source inventory finds no downstream literal hash of
`phase_d_chance.py`. Its search caller imports the module directly; its existing
source test reads `SOURCE_PINS` dynamically. Historical fidelity consumers and
their independent pins remain untouched. Existing correspondence, suspension,
cardinality and complete-mass refusals remain in force. Both bindings must be
updated together; updating just the transition pin would leave domain drift.

Focused tests read source/AST/hash fixtures without importing production code.
They assert all 13 direct hashes, the complete 25-module domain, unchanged
consumer bytes outside the two bindings, and selected frozen mapping,
constructor, tracker, search and finite-law files. No transition, sampler,
engine, model, root, game or cluster operation is executed.
