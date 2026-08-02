# Fan-In Filesystem Protocol

Fan-in workers sharing an iteration cache use atomic hard-link creation,
directory creation, and same-directory rename as their distributed protocol.
The queue and every cache mount must provide those operations with POSIX
cross-node semantics, such as an appropriately configured NFS or CephFS
volume. Object-store FUSE mounts and filesystems that only coordinate locks
within one node are unsupported.

Before a pending task is renamed into `claimed/` or any route record is bound,
the worker probes that task's queue and cache locations plus the exact
`<queue>/.fanin-routes` directory that hosts route-record CAS. A newly created
empty route directory is removed after its probe, including on failure, so a
failed preflight leaves pending, claimed, and route provenance state untouched.
Two probe files race to hard-link onto one initially absent name: exactly one
link must succeed, the other must report `EEXIST`, and the visible destination
must be the winning source inode. The probe also creates a directory, renames
it within that same location, and confirms the moved directory remains visible.
Any failure is a terminal validation error before a claim or route CAS is
touched.

The probe establishes only one-node behavior. Before launch, deployment
validation must make workers on every participating node contend for one guard
and one route record on each shared mount. The protocol assumes `link(2)` has
atomic create-if-absent collision semantics; source and destination of each
link are on the same filesystem; each `mkdir` collision is atomic; and every
rename stays within one filesystem and parent directory. Readers also rely on
directory listings and name lookups becoming mutually visible across workers
without stale negative results. The local probe cannot prove any of those
cross-node visibility properties.

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
deterministic digest of all cache entries except the self-describing publication
proof. The digest uses a no-follow,
descriptor-based walk of a real (lstat-validated) root, records each directory
or regular-file type, name, and size, and records regular-file bytes. Symlinks
and unsupported entry types are rejected before traversal. Root and entry
identities are rechecked during validation, so a replacement race fails closed.
Target rename alone does not accept training input. The immutable terminal
acceptance copies that proof and is the commit point. A target-before-acceptance
crash can be resumed without recollection by validating the target proof and
racing the same acceptance outcome. A target that lost to a successor may
remain as an orphan, but strict inventory selects only a cumulative version
whose final append authenticates that exact target and whose complete manifest
prefix belongs to the accepted lineage. Missing or malformed target/acceptance
provenance fails closed.

Before collecting, fan-in requires `a_out` to be an already canonical,
absolute physical path: relative paths, `..` normalization, and symlinked
aliases are rejected before the task is claimed. The same form is required in
every fan-in manifest task and route record, and a route record's cache root
must exactly equal the output route's parent. This prevents workers started in
different working directories from binding one task ID to different cache
roots.

Each task ID is bound by an append-only queue-local route record to that
complete output route and fan-in root. Route records must be regular,
non-symlink files; readers open them with `O_NOFOLLOW` through a stable route
directory descriptor and verify file identity before and after reading. An
`EEXIST` hard-link collision is accepted only when that stable record has the
exact canonical bytes expected for the task. Recovery always consults the
bound root; a retry that changes a basename, parent, or policy is terminal
rather than permitted to publish a second copy. Missing, malformed, special,
replaced, or ambiguous route provenance fails closed.

Guard generations, their owner records, terminal acceptance records, and
release markers use the same no-follow regular-file rule. Traversal opens every
generation and outcome as a real directory, keeps its identity stable while
reading, and rejects symlink, special-file, or replacement races. A renamed
guard or outcome replaced by a symlink is therefore terminal rather than a
redirect to another generation.
