# Fan-In Filesystem Protocol

Fan-in workers sharing an iteration cache use atomic directory creation and
same-directory rename as their distributed fence. The cache mount must provide
those operations with POSIX cross-node semantics, such as an appropriately
configured NFS or CephFS volume. Object-store FUSE mounts and filesystems that
only coordinate locks within one node are unsupported.

Each fan-in worker runs a fail-closed `mkdir`/rename preflight before touching a
claimed task. A long publication keeps its directory-fence owner record alive
with a heartbeat; another worker may reclaim the fence only after that record
expires and its claim is no longer live. Malformed fence or staging records are
preserved for repair rather than guessed stale.
