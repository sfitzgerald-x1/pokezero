// Generic snapshots need the latest request boundary to remain paired with the
// serialized simulator state. Showdown's direct request API can transiently
// return no request objects just after a branch resolves, while the protocol
// streams have already supplied the actionable boundary. Preserve that fresh
// stream cache until the direct API has materialized a replacement.
export function snapshotBoundaryRequests(streamRequests, simulatorRequests) {
  if (
    simulatorRequests &&
    typeof simulatorRequests === "object" &&
    Object.keys(simulatorRequests).length > 0
  ) {
    return simulatorRequests;
  }
  return streamRequests && typeof streamRequests === "object" ? streamRequests : {};
}
