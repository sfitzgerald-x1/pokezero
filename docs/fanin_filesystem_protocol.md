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
the stable guard name. Each generation has one never-before-used outcome
pathname, deterministically derived from the guard root, predecessor pathname,
and predecessor claim token. An expired predecessor's contenders race to
rename the same initialized successor directory onto that pathname.

For a task-publication guard, the publisher races an initialized terminal
acceptance directory onto that same outcome pathname after its target rename.
Consequently successor installation and acceptance are one atomic filesystem
choice: a successor revokes the predecessor before the predecessor can accept,
while an accepted predecessor prevents any successor. Neither side checks
ownership and later mutates a reusable pathname. No installed generation or
acceptance is renamed, unlinked, or reused. Release writes a separate immutable
marker keyed by the generation pathname and claim token.

Workers follow owner directories until they find an absent outcome or a
terminal acceptance. They also enumerate exact installed-generation names for
the root and compare them with the reachable chain. A missing middle generation
with a surviving descendant therefore fails closed instead of making the
historical middle pathname appear reusable. Temporary and release names do not
match the exact generation grammar. Malformed, ambiguous, unreadable, orphaned,
or over-4,096-generation chains are terminal.

Before target rename, each staging shard receives a publication proof binding
the task, claim generation, guard generation, worker lineage, target/version,
manifest index and prefix, record count, manifest and metadata hashes, and a
deterministic digest of all cache files. Target rename alone does not accept
training input. The immutable terminal acceptance copies that proof and is the
commit point. A target-before-acceptance crash can be resumed without
recollection by validating the target proof and racing the same acceptance
outcome. A target that lost to a successor may remain as an orphan, but strict
inventory selects only a cumulative version whose final append authenticates
that exact target and whose complete manifest prefix belongs to the accepted
lineage. Missing or malformed target/acceptance provenance fails closed.

Before collecting, each task ID is bound by an append-only queue-local route
record to its complete output route and fan-in root. Recovery always consults
that root; a retry that changes a basename, parent, or policy is terminal
rather than permitted to publish a second copy. Missing, malformed, or
ambiguous route provenance fails closed.
