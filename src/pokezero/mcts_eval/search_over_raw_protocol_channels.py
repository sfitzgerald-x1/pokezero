"""Strict singles protocol-channel filtering, without source or actor admission.

These functions are the shared kernel for original-source public-history and
constructed replay-suffix readers. Choosing a channel cannot prove that a log
belongs to that seat, is complete, or has original source/transport ownership.
Callers must establish those facts separately. No parser, model, simulator,
archive, or process is started here.
"""

MAXIMUM_BYTES = 8 * 1024 * 1024
# Supported upstream splitRegex + extractChannelMessages implementation.
FILTER_SHA256 = "2ce6a9c7cf4d740c445c8462c8b48f030eee63b5398eaa943f5c7dcfa1123b0e"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def channel_zero_lines(log):
    """Strict channel-0 equivalent on a validated singles battle-log roster.

    Unlike upstream's permissive regex, malformed/truncated split triples and
    request lines refuse. Split secrets are discarded; an empty shared message
    is omitted just as upstream does. Ordinary empty lines are also omitted.
    """
    return channel_lines(log, channel=0)


def channel_lines(log, *, channel):
    """Pinned 0/1/2 filter; seat channels ONLY on constructed replay suffixes.

    Never apply original-source channel 2 to a sampled actor: that would import
    the original opponent's private history rather than its constructed own.
    """
    require(type(channel) is int and channel in {0, 1, 2}, "exact spectator or singles actor channel required")
    require(type(log) is list and all(type(line) is str and not any(c in line for c in
        ("\n", "\r", "\u2028", "\u2029", "\x00")) for line in log),
        "source log must contain exact single protocol lines")
    require(sum(len(line.encode("utf-8")) + 1 for line in log) <= MAXIMUM_BYTES,
        "source public history cap exceeded")
    require(not any(line.startswith("|request") for line in log),
        "request-bearing source logs cannot become public history")
    public, index = [], 0
    while index < len(log):
        line = log[index]
        if line.startswith("|split"):
            require(line in {"|split|p1", "|split|p2"} and index + 2 < len(log)
                and not any(part.startswith("|split") for part in log[index + 1:index + 3]),
                "malformed/truncated source split group")
            visible = log[index + 1] if line == "|split|p" + str(channel) else log[index + 2]
            if visible: public.append(visible)
            index += 3
        else:
            if line: public.append(line)
            index += 1
    return public
