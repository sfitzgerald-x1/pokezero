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
never on a same-worker retry. Guard release likewise writes an immutable marker
keyed by the guard inode and claim generation; it never unlinks or renames the
reusable guard path.
The next acquirer performs the only shared-name handoff by atomically moving an
expired marked guard aside before publishing its initialized successor.

Before collecting, each task ID is bound by an append-only queue-local route
record to its complete output route and fan-in root. Recovery always consults
that root; a retry that changes a basename, parent, or policy is terminal
rather than permitted to publish a second copy. Missing, malformed, or
ambiguous route provenance fails closed.
