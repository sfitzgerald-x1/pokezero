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
