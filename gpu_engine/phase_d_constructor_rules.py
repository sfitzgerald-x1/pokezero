"""Pure Phase D request-order and public-percent rules; no engine changes."""


def public_hp_bin(maximum, shown):
    """Inverse of Gen3 public ceil percent, capped at99 below full HP.

    100 is exactly full; 99 also includes the below-full ceil100 tail.
    No sampled maximum or exact opponent HP is accepted as public evidence.
    """
    if type(maximum)is not int or maximum<=0 or type(shown)is not int or not 0<=shown<=100:
        raise ValueError('literal positive maxHP and public0..100percent required')
    if shown==0:return 0,0
    if shown==100:return maximum,maximum
    lower=((shown-1)*maximum)//100+1
    upper=maximum-1 if shown==99 else(shown*maximum)//100
    if lower>upper:raise ValueError('sampled maximum has no HP in public bin')
    return lower,upper


def request_ordered_packed(packed, sampled_names, request_names):
    """Move whole literal packed entries, preserving IVs, gender and happiness."""
    entries=packed.split(']');a=tuple(sampled_names);b=tuple(request_names)
    if len(entries)!=len(a)or len(a)!=len(b)or len(a)!=6 or len(set(a))!=6 or len(set(b))!=6 or set(a)!=set(b):
        raise ValueError('same-six-unique-own-species-required-for-request-order')
    by_name=dict(zip(a,entries))
    return ']'.join(by_name[name]for name in b)


def literal_hidden_power_agrees(literals, resolved):
    """The typed own request is evidence before static-identity uniqueness."""
    return all(resolved is not None and resolved.startswith(literal)for literal in literals)
