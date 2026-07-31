# Fan-In Filesystem Protocol

Fan-in workers sharing an iteration cache use atomic directory creation and
same-directory rename as their distributed fence. The cache mount must provide
those operations with POSIX cross-node semantics, such as an appropriately
configured NFS or CephFS volume. Object-store FUSE mounts and filesystems that
only coordinate locks within one node are unsupported.

Each fan-in worker runs a fail-closed `mkdir`/rename probe before touching a
claimed task. That probe verifies local mount capability only; it does not prove
cross-node atomicity. Before launch, the private deployment smoke must make
workers on multiple nodes contend for one guard on the shared mount.

A long publication keeps its directory-fence owner record alive with a
heartbeat; another worker may reclaim the fence only after that record expires
and its claim is no longer live. Malformed fence or staging records are
preserved for repair rather than guessed stale.

Claim filenames are generation-unique capabilities. Acknowledgement and failed
quarantine therefore operate only on the generation that acquired the task,
never on a same-worker retry.

Guard directories form an append-only generation chain. Generation zero uses
the stable guard name; every later pathname is deterministically derived from
the predecessor pathname and claim token. Contenders for the same expired
generation therefore race to publish the same initialized successor directory.
The winner's non-empty directory makes every other atomic rename collide. No
installed generation is renamed, unlinked, or reused, and the existence of a
successor immediately revokes predecessor ownership. Release writes a separate
immutable marker keyed by the never-reused generation pathname and claim token.
Workers follow the deterministic chain to find its current tail, fail closed on
malformed or unreadable generations, and reject chains beyond the traversal
bound of 4,096 generations per guard root.

Before collecting, each task ID is bound by an append-only queue-local route
record to its complete output route and fan-in root. Recovery always consults
that root; a retry that changes a basename, parent, or policy is terminal
rather than permitted to publish a second copy. Missing, malformed, or
ambiguous route provenance fails closed.
