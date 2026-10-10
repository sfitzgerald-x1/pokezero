"""Retain initialized anchor proposals independently of resampling genealogy."""
from .paper_reference import ReferenceRefusal


def retain_initial_anchor(receipt, *, ordinal, anchor, importance_weight):
    rows = receipt.setdefault("initial_anchor_proposals", [])
    if type(ordinal) is not int or ordinal != len(rows):
        raise ReferenceRefusal("initial anchor receipt ordinal drift")
    # The proposal was already copied when the owned anchor was captured.
    # Keep that immutable-by-ownership record alive; do not serialize/deepcopy
    # its complete seed bank a second time on the decision's timed path.
    rows.append(dict(initialization_ordinal=ordinal,
        materialization_seed=anchor.get("materialization_seed"),
        materialization_seed_proposal=anchor.get("materialization_seed_proposal"),
        initial_anchor_importance_weight=importance_weight))


def initial_ledger_draw_view(receipt, empirical_draw):
    """Emit the full ledger once per population, not once per forward trajectory."""
    view = dict(receipt)
    if empirical_draw > 1:
        view.pop("initial_anchor_proposals", None)
        view["initial_anchor_proposals_retained_at_empirical_draw"] = 1
    return view
