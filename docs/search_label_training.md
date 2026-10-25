# Optional search labels for PPO

Search labeling preserves the raw sampled actor. Selection uses a hash of the
game seed, learner seat and decision index, independent of the actor RNG. Only
the assigned learner seat is wrapped, including when the opponent has the same
policy spec. Its search-proposed action is discarded.

`neural_cli iterate --search-labels --objective ppo` enables collection on a
seeded 10% of learner decisions. Set `--search-policy-loss-weight 1` and
`--search-value-mix-weight 0.5` for the treatment. Both loss switches default to
zero; collection is disabled unless requested. The teacher device defaults to
the training device and can be specified with `--search-label-device`.

The teacher uses model leaves, 2,048 simulations, depth 8, batch 64, one belief
world and one world worker, both policy priors, c_puct 1.4, unset FPU reduction,
and no early stop. Exports are prepared from the current checkpoint before
collection at startup/resume and after each 25 iterations. For a fresh run this
is iterations 1, 26, 51, etc. Every attempted label retains the checkpoint,
export hashes, refresh iteration, configuration and capture build. Changed
artifacts or configuration fail before teacher construction.

Each iteration writes `search-teacher.json`. External workers can pass that
immutable manifest to `rollout_cli collect-selfplay-training-cache` with
`--search-label-teacher`. This path validates the same artifacts, build and
teacher settings and requires Showdown collection. It does not export or
refresh independently. Benchmarks and heldout collection receive no wrapper.

Real teachers require the existing scientific quarantine and reviewed capture
admission. A counted search fallback or unavailable target produces an explicit
unlabeled attempt with false masks. Capture loss remains fatal; configuration
and provenance failures are not converted into missing labels.

Accepted targets contain nine legal visit probabilities and a signed acting-seat
value. Cache writers preserve optional target arrays and sparse teacher/refusal
metadata. Legacy caches gain no files or metadata keys. Mixing old and new
caches fills only absent legacy labels with false masks; damaged declared label
caches are refused.

PPO probabilities, advantages and recorded behavior values remain unchanged.
The auxiliary policy loss is cross-entropy on labeled rows. The critic blends
the existing PPO/GAE target with the search value only where its mask is set,
including clipped value regression. Epoch summaries retain weighted search CE
and policy/value label counts. Distributed reductions include ranks with no
labels or no examples.

This implementation does not establish a scientific training gain. The teacher
screen and registered training comparison determine whether an experiment runs
and whether it succeeds.
