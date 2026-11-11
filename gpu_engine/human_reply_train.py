"""π_human: the champion's policy fine-tuned on human players' decisions (g3rb_views: public Gen 3 random-battle replays, each
player's own view), with a rating-band token and optional in-game history, as the search's reply prior against human opponents.

usage:
  python -m gpu_engine.human_reply_train train VIEWS OUT --checkpoint PT [--bands] [--history] [--epochs 4] [--batch 512]
         [--lr 3e-5] [--new-lr 3e-4] [--band-dropout 0.15] [--history-events 32] [--seed 0] [--device cuda]
  python -m gpu_engine.human_reply_train evaluate VIEWS OUT --checkpoint PT --model NAME=RUN_DIR [...] [--pi-fp PT]
         [--players VIEWS_INPUT.jsonl]
  python -m gpu_engine.human_reply_train foulplay FPR OUT --checkpoint PT --pi-fp PT --test-seeds JSON [--late-from N]
         --model NAME=RUN_DIR [...]
  python -m gpu_engine.human_reply_train export RUN_DIR OUT_DIR --band unknown|unrated|below_1200|1200_1499|1500_plus

Data (VIEWS: g3rb_views shards). Inputs are the builder's seat view (view b) with π_FP-red2's peer-view-v2 redaction, each row's
own field column 116 (its view of the other side's Spikes losses, already at public precision) as the credit; that zeroes the
cells the search's peer view leaves at zero (own LAST_DAMAGE_TAKEN, the other side's LAST_DAMAGE_DEALT, SELF_HAZARD_CREDIT).
Labels: a labelled row (label >= 0) is a full label. A `cant` or `fainted_before_acting` row is a partial label, 'some move':
its likelihood is the legal moves' total probability (both only happen to a chosen move). Other censored rows (no_action,
recharge, label_undecodable) carry no choice and are dropped.
Splits: test is the strict vintage (the 33-day window inside the Showdown pin; never trained on, never used for selection);
of the rest, validation is replay hash % 10 == 1 (sha256 of 'g3rb-split:<replay>'), train the rest. Both seats of a replay
stay together.

Model (HumanReply). The champion's trunk, embeddings and policy head, window 1, with two additive inputs, both zero at
initialisation so the untrained wrapper reproduces the champion exactly (checked before training):
- bands (--bands): a learned [band, token type, width] table added to every token's input embedding. Bands: unknown (0),
  unrated, below 1200, 1200-1499, 1500+ from the replay's rating; --band-dropout replaces the band by unknown on that share of
  training rows, so unknown is the marginal. Because the term depends only on the token type, one band folds exactly into the
  champion's token_type_embedding (export): the served archive is a plain champion_evaluator archive in π_FP's format.
- history (--history): the seat's own earlier decisions in the game (up to --history-events, most recent last). Each event is
  the input embeddings of the chosen action's candidate token (a learned 'some move' vector for a partial label), the field
  token and both actives (tokens 0, 1, 7), plus the turn age, kind, forced, locked, partial and switch flags; a 2-layer
  transformer with a summary slot pools them, and two zero-initialised projections add the summary to every token and to the
  action-candidate tokens. A history model needs its event inputs at serve time, so it has no champion archive.
Training: AdamW (trunk --lr, new parameters --new-lr), linear warm-up then cosine to 10%, gradient clip 1, cross-entropy of
the full labels and -log P(some move) of the partial ones; validation every --eval-every steps on the full-label validation
NLL, the best kept. Exact fp32 on CUDA (TF32 off, MHA fast path off), as fp_reply_train.

evaluate: the strict hold-out by rating band (unrated, below 1200, 1200-1499, 1500+, all): uniform over legal, the champion's
π (on the same redacted inputs, and on the unredacted view), π recalibrated on train (temperature and switch bias), π_FP
(--pi-fp), and each --model with its true band and with band unknown. Per model: top-1, NLL, ECE (15 bins), labels under
1%/5%, switch mass, the partial-label NLL, the move/switch-decision split, the NLL-by-turn curve, and replay-clustered 90%
bootstrap intervals for the paired NLL differences, including the history contrast late (turns 18+) minus early (turns 1-3).
--players (views-input JSONL) adds the player-disjoint subset: held-out decisions whose (hashed) player never appears in a
training or validation replay.
foulplay: the same models on Foul Play's own decisions (fp_reply_data on view b, peer-view-v2, the --test-seeds / --late-from
split of π_FP's run), every band, against π_FP and the champion; battle-seed-clustered intervals.
export: fold one band into a champion-format checkpoint (pi-human.pt, marked policy-only, extra.view 'b') and its
champion_evaluator archive (pi-human-weights.npz); the folded checkpoint must reproduce the wrapper to 1e-5 and the JAX archive
the folded checkpoint to 1e-4, on held-out rows.
"""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_data as data_mod
from gpu_engine import fp_reply_train as fpt

BANDS = ('unknown', 'unrated', 'below_1200', '1200_1499', '1500_plus')
VINTAGES = ('strict', 'extended', 'outside')                       # g3rb_views.VINTAGES
CENSORS = ('none', 'cant', 'fainted_before_acting', 'no_action', 'label_undecodable', 'view_error', 'recharge')   # g3rb_views.CENSORS
PARTIAL = ('cant', 'fainted_before_acting')                        # 'some move' partial labels
MOVES, ACTIONS = data_mod.MOVES, data_mod.ACTIONS
ACTION_OFFSET = 13                                                  # showdown.ACTION_CANDIDATE_TOKEN_OFFSET (v4)
EVENT_TOKENS = (0, 1, 7)                                            # field, own active, the other side's active
EVENT_SCALARS = 6
TURN_BUCKETS = fpt.TURN_BUCKETS
EARLY, LATE = (1, 3), (18, 10**9)
IDENTITY_TOLERANCE = 1e-5
FOLD_TOLERANCE = 1e-5
SCHEMA = 'v2s-human-reply.v1'


# ---- data ------------------------------------------------------------------------------------------------------------------------

def band_of(rating):
    """Band ids (BANDS) from the replay's rating: 0 or missing is unrated; never 0 (unknown), which only dropout and serving use."""
    r = np.nan_to_num(np.asarray(rating, np.float64), nan=0.)
    return np.where(r <= 0, 1, np.where(r < 1200, 2, np.where(r < 1500, 3, 4))).astype(np.int64)


def replay_split(replay):
    """'val' where sha256('g3rb-split:<replay>') % 10 == 1, else 'train' (the test split is the strict vintage, not a hash)."""
    h = np.array([int(hashlib.sha256(f'g3rb-split:{int(r)}'.encode()).hexdigest()[:8], 16) % 10 for r in np.asarray(replay)])
    return np.where(h == 1, 'val', 'train')


def partial_mask(arrays):
    censor = np.asarray(arrays['censor'])
    return np.isin(censor, [CENSORS.index(c) for c in PARTIAL]) & arrays['legal'][:, :MOVES].any(1) & (arrays['label'] < 0)


def keep_rows(arrays):
    """Rows with a full label or a 'some move' partial label (others carry no choice)."""
    return (arrays['label'] >= 0) | partial_mask(arrays)


def redact_own_credit(numeric):
    """peer-view-v2 on human rows: the credit is each row's own field column 116 (fp_reply_train.redact_peer_view)."""
    credit = numeric[:, fpt.FIELD_TOKEN, fpt.OPP_HAZARD_CREDIT].copy()
    return fpt.redact_peer_view(numeric, 'peer-view-v2', credit)


def load_views(views_dir, unredacted=False):
    """The kept rows of every shard (games in order), redacted (peer-view-v2, own credit), with band, split, partial and group
    (replay, imputation, seat); unredacted=True also keeps the builder's view as numeric_unredacted."""
    arrays = fpt.load(views_dir)
    keep = keep_rows(arrays);arrays = {k: v[keep] for k, v in arrays.items()}
    if unredacted:arrays['numeric_unredacted'] = arrays['numeric_features'].copy()
    redact_own_credit(arrays['numeric_features'])
    arrays['partial'] = partial_mask(arrays)
    arrays['band'] = band_of(arrays['rating'])
    split = replay_split(arrays['replay'])
    arrays['split'] = np.where(arrays['vintage'] == VINTAGES.index('strict'), 'test', split)
    arrays['group'] = group_ids(arrays['replay'], arrays['imputation'], arrays['seat'])
    return arrays


def group_ids(*keys):
    """Consecutive-run ids: a new group wherever any key changes from the previous row (rows of a group are contiguous)."""
    keys = [np.asarray(k) for k in keys];n = len(keys[0])
    change = np.zeros(n, bool)
    if n:change[0] = True
    for k in keys:change[1:] |= k[1:] != k[:-1]
    return np.cumsum(change) - 1


def history_index(group, events):
    """index[i, j]: the row of the seat's (events - j)-th previous decision in the same group (most recent in the last column),
    -1 where there is none. Rows of a group are contiguous and in decision order."""
    group = np.asarray(group);n = len(group)
    start = np.zeros(n, np.int64)
    if n:
        first = np.r_[True, group[1:] != group[:-1]];starts = np.flatnonzero(first)
        start = starts[np.cumsum(first) - 1]
    index = np.full((n, events), -1, np.int64);rows = np.arange(n)
    for k in range(1, events + 1):
        prev = rows - k;index[:, events - k] = np.where(prev >= start, prev, -1)
    return index


def players_of(views_input):
    """{replay index: (p1 hash, p2 hash)} from the views-input JSONL's |player| lines (hashed names; never printed)."""
    out = {}
    with open(views_input) as f:
        for line in f:
            if not line.strip():continue
            row = json.loads(line);seats = {}
            for m in re.finditer(r'^\|player\|(p[12])\|([^|\n]*)', row['log'], re.M):
                if m.group(2):seats[m.group(1)] = m.group(2)
            out[int(row['index'])] = (seats.get('p1'), seats.get('p2'))
    return out


def player_disjoint(arrays, players, test_rows):
    """Held-out rows whose deciding player appears in no training or validation replay (either seat)."""
    fit = set()
    for r in np.unique(arrays['replay'][arrays['split'] != 'test']):fit.update(p for p in players.get(int(r), (None, None)) if p)
    keep = []
    for i in test_rows:
        p = players.get(int(arrays['replay'][i]), (None, None))[int(arrays['seat'][i])]
        keep.append(p is not None and p not in fit)
    return test_rows[np.asarray(keep, bool)] if len(test_rows) else test_rows


# ---- model -------------------------------------------------------------------------------------------------------------------------

def build(torch):
    from pokezero.showdown import ACTION_CANDIDATE_TOKEN_OFFSET
    if ACTION_CANDIDATE_TOKEN_OFFSET != ACTION_OFFSET:raise SystemExit(f'action tokens start at {ACTION_CANDIDATE_TOKEN_OFFSET}, not {ACTION_OFFSET}')
    nn = torch.nn
    F = torch.nn.functional

    def embed_tokens(base, cat, num, tt):
        """The trunk's input embedding of individual tokens (no history position): embedding_bag sums the categorical ids
        (padding id 0 contributes nothing, as in the trunk's padding_idx row), plus token type and the numeric projection."""
        shape = cat.shape[:-1];vocab = base.config.categorical_vocab_size
        ids = cat.reshape(-1, cat.shape[-1]).long().clamp(0, vocab - 1)
        summed = F.embedding_bag(ids, base.category_embedding.weight, mode='sum', padding_idx=0).view(*shape, -1)
        types = base.token_type_embedding(tt.long().clamp(0, base.config.token_type_vocab_size - 1))
        return summed + types + base.numeric_projection(num.to(base.numeric_projection.weight.dtype))

    class History(nn.Module):
        def __init__(self, width, d=256, layers=2, heads=4):
            super().__init__()
            self.tokens = nn.Linear(4 * width, d);self.scalars = nn.Linear(EVENT_SCALARS, d)
            self.partial = nn.Parameter(torch.zeros(width));self.summary = nn.Parameter(torch.randn(d) * .02)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0., batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False);self.norm = nn.LayerNorm(d)
            self.to_all = nn.Linear(d, width);self.to_action = nn.Linear(d, width)
            for p in (self.to_all, self.to_action):nn.init.zeros_(p.weight);nn.init.zeros_(p.bias)

        def forward(self, tokens, scalars, valid):
            e = self.tokens(tokens.flatten(2)) + self.scalars(scalars)
            x = torch.cat([self.summary.expand(e.shape[0], 1, -1), e], 1)
            pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
            h = self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
            return self.to_all(h), self.to_action(h)

    class HumanReply(nn.Module):
        def __init__(self, base, bands=False, history=False, events=32):
            super().__init__()
            config = base.config
            if int(config.window_size) != 1 or base.encoder is None or base.temporal_gru is not None:
                raise SystemExit('HumanReply needs a window-1 trunk with an encoder and no temporal GRU (the 9375 champion)')
            self.base = base;width = config.embedding_dim;self.events = events
            self.band = nn.Parameter(torch.zeros(len(BANDS), config.token_type_vocab_size, width)) if bands else None
            self.history = History(width) if history else None

        def forward(self, cat, num, tt, att, band=None, hist=None):
            base = self.base;width = base.config.embedding_dim
            x = base._embed_expanded_inputs(categorical_ids=cat.unsqueeze(1), numeric_features=num.unsqueeze(1),
                                            token_type_ids=tt.unsqueeze(1)).squeeze(1)
            x = x + base.history_position_embedding.weight[0]
            if self.band is not None and band is not None:
                x = x + self.band[band.unsqueeze(1), tt.long().clamp(0, base.config.token_type_vocab_size - 1)]
            if self.history is not None and hist is not None:
                to_all, to_action = self.history(*hist)
                bump = torch.zeros_like(x);bump[:, ACTION_OFFSET:ACTION_OFFSET + ACTIONS] = to_action.unsqueeze(1)
                x = x + to_all.unsqueeze(1) + bump
            encoded = base.encoder(x, src_key_padding_mask=~att.bool())
            return base.policy_head(encoded[:, ACTION_OFFSET:ACTION_OFFSET + ACTIONS]).squeeze(-1)

        def new_parameters(self):
            return [p for n, p in self.named_parameters() if not n.startswith('base.')]

    return HumanReply, embed_tokens


class Store:
    """Device-resident arrays and the batch builder (inputs, labels, band and history events)."""

    def __init__(self, arrays, torch, device, events=0, numeric_key='numeric_features'):
        self.torch, self.device, self.events = torch, device, events
        t = lambda a, dtype=None: torch.as_tensor(np.ascontiguousarray(a), device=device) if dtype is None else torch.as_tensor(np.ascontiguousarray(a), device=device).to(dtype)
        self.cat = t(arrays['categorical_ids'].astype(np.int32));self.num = t(arrays[numeric_key].astype(np.float32))
        self.tt = t(arrays['token_type_ids'].astype(np.int64));self.att = t(arrays['attention_mask'].astype(bool))
        self.legal = t(arrays['legal'].astype(bool));self.label = t(arrays['label'].astype(np.int64))
        n = len(arrays['label'])
        self.partial = t(arrays.get('partial', np.zeros(n, bool)).astype(bool))
        self.band = t(arrays.get('band', np.zeros(n, np.int64)).astype(np.int64))
        if events:
            group = arrays.get('group', group_ids(arrays['ordinal']) if 'ordinal' in arrays else np.zeros(n, np.int64))
            self.hist = t(history_index(group, events));self.turn = t(np.asarray(arrays['turn'], np.int64))
            kind = np.asarray(arrays['kind'] if 'kind' in arrays else arrays['forced'], np.float32)
            self.flags = t(np.stack([kind, np.asarray(arrays['forced'], np.float32), np.asarray(arrays.get('locked', np.zeros(n)), np.float32),
                                     np.asarray(arrays.get('partial', np.zeros(n)), np.float32),
                                     (np.asarray(arrays['label']) >= MOVES).astype(np.float32)], 1))

    def inputs(self, idx, embed_tokens=None, model=None, band_override=None, dropout=0., generator=None):
        torch = self.torch
        out = dict(cat=self.cat[idx].long(), num=self.num[idx], tt=self.tt[idx], att=self.att[idx])
        band = self.band[idx] if band_override is None else torch.full_like(self.band[idx], band_override)
        if dropout:band = torch.where(torch.rand(band.shape, device=band.device, generator=generator) < dropout, torch.zeros_like(band), band)
        out['band'] = band
        if self.events and model is not None and model.history is not None:
            out['hist'] = self.history_inputs(idx, embed_tokens, model)
        return out

    def history_inputs(self, idx, embed_tokens, model):
        torch = self.torch;base = model.base
        rows = self.hist[idx];valid = rows >= 0;safe = rows.clamp(min=0)
        label = self.label[safe];full = label >= 0
        tok = torch.tensor(EVENT_TOKENS, device=self.device).expand(*rows.shape, len(EVENT_TOKENS))
        chosen = (ACTION_OFFSET + label.clamp(min=0)).unsqueeze(-1)
        tok = torch.cat([tok, chosen], -1)                                     # [B, H, 4]
        r = safe.unsqueeze(-1).expand_as(tok)
        emb = embed_tokens(base, self.cat[r, tok], self.num[r, tok], self.tt[r, tok])   # [B, H, 4, width]
        emb = torch.cat([emb[..., :3, :], torch.where(full.unsqueeze(-1), emb[..., 3, :], model.history.partial).unsqueeze(-2)], -2)
        age = ((self.turn[idx].unsqueeze(1) - self.turn[safe]).clamp(0, 50).float() / 10.).unsqueeze(-1)
        scalars = torch.cat([age, self.flags[safe]], -1) * valid.unsqueeze(-1)
        emb = emb * valid.unsqueeze(-1).unsqueeze(-1)
        return emb, scalars, valid


def set_exact(torch):
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)


def masked_logits(logits, legal):
    return logits.float().masked_fill(~legal, -1e9)


def loss_terms(torch, logits, legal, label, partial):
    """Per-row negative log-likelihood: the full label's, or -log P(some legal move) for a partial label."""
    logp = torch.log_softmax(masked_logits(logits, legal), -1)
    full = -logp.gather(1, label.clamp(min=0).unsqueeze(1)).squeeze(1)
    moves = logp[:, :MOVES].masked_fill(~legal[:, :MOVES], float('-inf'))
    some = -torch.logsumexp(moves, -1)
    return torch.where(partial, some, full)


def predict(model, store, rows, torch, embed_tokens, band=None, size=2048, base_only=False):
    """Probabilities [n, 9] on `rows` (band: None = each row's own, else a band id for every row). base_only: the champion trunk
    alone (the wrapper's own forward with no band and no history: equal to the plain checkpoint by the identity check)."""
    model.eval();out = []
    with torch.no_grad():
        for lo in range(0, len(rows), size):
            idx = torch.as_tensor(rows[lo:lo + size], device=store.device)
            if base_only:
                logits = plain_forward(model.base, store, idx)
            else:
                logits = model(**store.inputs(idx, embed_tokens, model, band_override=band))
            out.append(torch.softmax(masked_logits(logits, store.legal[idx]), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ACTIONS), np.float32)


def plain_forward(base, store, idx):
    """The champion checkpoint's own forward (window 1), for the identity check and the champion baselines."""
    output = base(categorical_ids=store.cat[idx].long().unsqueeze(1), numeric_features=store.num[idx].unsqueeze(1),
                  token_type_ids=store.tt[idx].unsqueeze(1), attention_mask=store.att[idx].unsqueeze(1),
                  history_mask=store.torch.ones((len(idx), 1), dtype=store.torch.bool, device=store.device))
    return output.policy_logits


def saved_wrapper(run_dir, torch):
    """RUN_DIR/human.pt's wrapper record {bands, history, events} (read without building the model)."""
    return torch.load(Path(run_dir) / 'human.pt', map_location='cpu', weights_only=False)['wrapper']


def store_events(specs, torch):
    """The history length every history model in `specs` was trained with (they must agree: one store feeds them all); 0 when
    none has history."""
    events = {saved_wrapper(p, torch)['events'] for p in specs.values() if saved_wrapper(p, torch)['history']}
    if len(events) > 1:raise SystemExit(f'history models with different --history-events {sorted(events)}: evaluate them separately')
    return events.pop() if events else 0


def load_human(run_dir, torch, device):
    """A trained HumanReply from RUN_DIR/human.pt (the base checkpoint's architecture, then the wrapper's whole state)."""
    from pokezero.neural_policy import load_transformer_checkpoint
    saved = torch.load(Path(run_dir) / 'human.pt', map_location='cpu', weights_only=False)
    base, _ = load_transformer_checkpoint(saved['base_checkpoint'], map_location=device)
    HumanReply, embed_tokens = build(torch)
    model = HumanReply(base, bands=saved['wrapper']['bands'], history=saved['wrapper']['history'], events=saved['wrapper']['events']).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    return model, embed_tokens, saved


# ---- metrics -----------------------------------------------------------------------------------------------------------------------

def label_nll(probs, labels, partial=None):
    """Per-row NLL: the label's, or -log P(some legal move) on partial rows (probs are already zero off the legal set)."""
    p = probs[np.arange(len(labels)), np.clip(labels, 0, None)]
    if partial is not None:p = np.where(partial, probs[:, :MOVES].sum(1), p)
    return -np.log(np.clip(p, 1e-12, 1.))


def cluster_ci(values, clusters, draws=2000, seed=0):
    """Mean and 90% bootstrap interval resampling clusters (replays, or battle seeds) with replacement."""
    values = np.asarray(values, float)
    if not len(values):return dict(mean=None, interval_90=None, rows=0, clusters=0)
    u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inv, values, len(u));counts = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed);stats = []
    for lo in range(0, draws, 250):
        idx = rng.integers(0, len(u), (min(250, draws - lo), len(u)));stats.append(sums[idx].sum(1) / counts[idx].sum(1))
    stats = np.concatenate(stats)
    return dict(mean=float(values.mean()), interval_90=[float(q) for q in np.quantile(stats, [.05, .95])], rows=int(len(values)),
                clusters=int(len(u)))


def contrast_ci(a, b, turn, clusters, early=EARLY, late=LATE, draws=2000, seed=0):
    """Late minus early of the mean paired difference d = a - b (per-row NLLs), resampling clusters: a history model's gain
    should grow with the turn."""
    d = np.asarray(a, float) - np.asarray(b, float);turn = np.asarray(turn)
    e = (turn >= early[0]) & (turn <= early[1]);l = (turn >= late[0]) & (turn <= late[1])
    if not e.any() or not l.any():return None
    u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    def sums(m):return np.bincount(inv, np.where(m, d, 0.), len(u)), np.bincount(inv, m.astype(float), len(u))
    (se, ce), (sl, cl) = sums(e), sums(l);rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (draws, len(u)))
    stats = sl[idx].sum(1) / np.maximum(cl[idx].sum(1), 1) - se[idx].sum(1) / np.maximum(ce[idx].sum(1), 1)
    return dict(mean=float(d[l].mean() - d[e].mean()), interval_90=[float(q) for q in np.quantile(stats, [.05, .95])],
                early_rows=int(e.sum()), late_rows=int(l.sum()), clusters=int(len(u)))


def model_metrics(probs, labels, legal, partial, uniform=False):
    full = ~partial
    out = data_mod.metrics(probs[full], labels[full], legal[full], uniform=uniform)
    out['partial_rows'] = int(partial.sum())
    out['partial_nll'] = float(label_nll(probs[partial], labels[partial], np.ones(int(partial.sum()), bool)).mean()) if partial.any() else None
    return out


def curve(models, labels, partial, turn, mask):
    """NLL by turn bucket (full-label rows within `mask`) for each model."""
    out = []
    for low, high in TURN_BUCKETS:
        m = mask & ~partial & (turn >= low) & (turn <= high)
        if m.any():out.append(dict(turns=f'{low}' if low == high else (f'{low}-{high}' if high < 10**9 else f'{low}+'), decisions=int(m.sum()),
                                   **{name: float(label_nll(p[m], labels[m]).mean()) for name, p in models.items()}))
    return out


def report_rows(models, arrays, rows, cluster_key, pairs, groups, contrasts=()):
    """models: {name: probs on rows}. Per group (a mask over rows): metrics per model, the turn curve, and paired NLL intervals
    (full-label rows; partial rows separately in each model's partial_nll)."""
    labels, legal, partial, turn = (np.asarray(arrays[k])[rows] for k in ('label', 'legal', 'partial', 'turn'))
    clusters = np.asarray(arrays[cluster_key])[rows]
    per = {name: label_nll(p, labels) for name, p in models.items()}
    out = {}
    for gname, mask in groups.items():
        g = dict(models={name: model_metrics(p[mask], labels[mask], legal[mask], partial[mask], uniform=name == 'uniform_legal')
                         for name, p in models.items()},
                 curve=curve(models, labels, partial, turn, mask), pairs={}, contrasts={})
        full = mask & ~partial
        for a, b in pairs:
            if a in per and b in per:g['pairs'][f'{a} - {b}'] = cluster_ci(per[a][full] - per[b][full], clusters[full])
        for a, b in contrasts:
            if a in per and b in per:g['contrasts'][f'({a} - {b}) late - early'] = contrast_ci(per[a][full], per[b][full], turn[full], clusters[full])
        out[gname] = g
    return out


def band_groups(arrays, rows):
    band = np.asarray(arrays['band'])[rows]
    groups = dict(all=np.ones(len(rows), bool))
    for b in range(1, len(BANDS)):groups[BANDS[b]] = band == b
    kind = np.asarray(arrays['kind'])[rows]
    groups['move_decision'] = kind == 0;groups['switch_decision'] = kind == 1
    return groups


# ---- train -------------------------------------------------------------------------------------------------------------------------

def train(args):
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256, load_transformer_checkpoint
    set_exact(torch);torch.manual_seed(args.seed);rng = np.random.default_rng(args.seed)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    if (out / 'human.pt').exists():raise SystemExit(f'{out} already holds a run')
    arrays = load_views(args.views)
    rows = {name: np.flatnonzero(arrays['split'] == name) for name in ('train', 'val', 'test')}
    counts = {name: dict(rows=int(len(r)), partial=int(arrays['partial'][r].sum()), replays=int(len(np.unique(arrays['replay'][r]))),
                         bands={BANDS[b]: int((arrays['band'][r] == b).sum()) for b in range(1, len(BANDS))}) for name, r in rows.items()}
    print(json.dumps(dict(event='DATA', **counts)), flush=True)
    base, _ = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    HumanReply, embed_tokens = build(torch)
    model = HumanReply(base, bands=args.bands, history=args.history, events=args.history_events).to(args.device)
    store = Store(arrays, torch, args.device, events=args.history_events if args.history else 0)
    # identity at initialisation: the wrapper (band and history paths at zero) is the champion's own forward
    probe = rows['val'][:4096]
    wrapped = predict(model, store, probe, torch, embed_tokens);plain = predict(model, store, probe, torch, embed_tokens, base_only=True)
    identity = float(np.abs(wrapped - plain).max())
    print(json.dumps(dict(event='IDENTITY', max_abs_probability_difference=identity, tolerance=IDENTITY_TOLERANCE)), flush=True)
    if not identity <= IDENTITY_TOLERANCE:raise SystemExit(f'the untrained wrapper differs from the champion by {identity}')
    new = model.new_parameters();new_ids = {id(p) for p in new}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in new_ids], lr=args.lr)]
    if new:groups.append(dict(params=new, lr=args.new_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(rows['train']) / args.batch);total = steps_per_epoch * args.epochs
    def factor(step):
        if step < args.warmup:return (step + 1) / args.warmup
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - args.warmup) / max(1, total - args.warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator(device=args.device);generator.manual_seed(args.seed)
    val_full = rows['val'][~arrays['partial'][rows['val']]]

    def validate():
        p = predict(model, store, rows['val'], torch, embed_tokens)
        v = arrays['partial'][rows['val']];l = arrays['label'][rows['val']]
        nl = label_nll(p, l, v)
        return dict(val_nll=float(nl[~v].mean()), val_partial_nll=float(nl[v].mean()) if v.any() else None)

    best = dict(step=0, **validate());best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [dict(step=0, epoch=0., **{k: best[k] for k in ('val_nll', 'val_partial_nll')})]
    print(json.dumps(dict(event='EVAL', **history[-1])), flush=True)
    step = 0;losses = []
    for epoch in range(args.epochs):
        order = rng.permutation(rows['train'])
        for lo in range(0, len(order), args.batch):
            model.train();idx = torch.as_tensor(order[lo:lo + args.batch], device=args.device)
            inputs = store.inputs(idx, embed_tokens, model, dropout=args.band_dropout if args.bands else 0., generator=generator)
            logits = model(**inputs)
            loss = loss_terms(torch, logits, store.legal[idx], store.label[idx], store.partial[idx]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step();scheduler.step()
            losses.append(float(loss));step += 1
            if step % args.eval_every == 0 or step == total:
                v = validate();history.append(dict(step=step, epoch=round(step / steps_per_epoch, 3), train_loss=float(np.mean(losses)), **v))
                losses = [];print(json.dumps(dict(event='EVAL', **history[-1])), flush=True)
                if v['val_nll'] < best['val_nll']:
                    best = dict(step=step, **v);best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    base_sha = checkpoint_file_sha256(args.checkpoint)
    marker = dict(policy_only=True, base_checkpoint_sha256=base_sha, view='b', redaction='peer-view-v2', data=str(args.views),
                  split='test: strict vintage; val: sha256(g3rb-split:<replay>) % 10 == 1; train: the rest', bands=list(BANDS),
                  band_dropout=args.band_dropout if args.bands else None, history_events=args.history_events if args.history else None,
                  best_step=best['step'], note='policy fine-tuned on human decisions; the value output is not trained')
    torch.save(dict(schema=SCHEMA, base_checkpoint=str(args.checkpoint), base_checkpoint_sha256=base_sha,
                    wrapper=dict(bands=bool(args.bands), history=bool(args.history), events=args.history_events),
                    state_dict=best_state, human_reply=marker), out / 'human.pt')
    report = dict(schema=SCHEMA, views=str(args.views), checkpoint=str(args.checkpoint), checkpoint_sha256=base_sha, args=vars(args),
                  data=counts, identity=identity, history=history, best=best, steps_per_epoch=steps_per_epoch,
                  saved=dict(path=str(out / 'human.pt'), sha256=checkpoint_file_sha256(out / 'human.pt')), seconds=round(time.time() - started, 1))
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='DONE', best=best, seconds=report['seconds'])), flush=True)
    return 0


# ---- evaluate (human hold-out) ------------------------------------------------------------------------------------------------------

def recalibration(probs, labels, legal, torch):
    params, apply = fpt.recalibrate(probs, labels, legal, torch)
    return params, apply


def parse_models(specs):
    out = {}
    for s in specs or ():
        name, _, path = s.partition('=')
        if not name or not path or not re.fullmatch(r'[a-z0-9_]+', name):raise SystemExit(f'--model NAME=RUN_DIR (lowercase name), got {s!r}')
        out[name] = path
    return out


def plain_model(path, torch, device):
    from pokezero.neural_policy import load_transformer_checkpoint
    base, _ = load_transformer_checkpoint(path, map_location=device)
    HumanReply, embed_tokens = build(torch)
    return HumanReply(base).to(device).eval(), embed_tokens


def evaluate(args):
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256
    set_exact(torch);out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    if (out / 'report-human.json').exists():raise SystemExit(f'{out} already holds an evaluation')
    arrays = load_views(args.views, unredacted=True)
    test = np.flatnonzero(arrays['split'] == 'test');train = np.flatnonzero(arrays['split'] == 'train')
    champion, embed_tokens = plain_model(args.checkpoint, torch, args.device)
    store = Store(arrays, torch, args.device, events=store_events(parse_models(args.model), torch))
    models = dict(uniform_legal=arrays['legal'][test] / arrays['legal'][test].sum(1, keepdims=True))
    models['champion'] = predict(champion, store, test, torch, embed_tokens, base_only=True)
    raw = Store({k: arrays[k][test] for k in ('categorical_ids', 'token_type_ids', 'attention_mask', 'legal', 'label')} |
                dict(numeric_features=arrays['numeric_unredacted'][test]), torch, args.device)
    models['champion_unredacted'] = predict(champion, raw, np.arange(len(test)), torch, embed_tokens, base_only=True)
    del raw;arrays.pop('numeric_unredacted')
    fit = np.random.default_rng(0).choice(train[~arrays['partial'][train]], min(args.recalibration_rows, int((~arrays['partial'][train]).sum())), replace=False)
    params, apply = recalibration(predict(champion, store, fit, torch, embed_tokens, base_only=True), arrays['label'][fit], arrays['legal'][fit], torch)
    models['champion_recalibrated'] = apply(models['champion'], arrays['legal'][test])
    if args.pi_fp:
        pi_fp, _ = plain_model(args.pi_fp, torch, args.device);models['pi_fp'] = predict(pi_fp, store, test, torch, embed_tokens, base_only=True);del pi_fp
    identities = {}
    for name, path in parse_models(args.model).items():
        model, embed_tokens, saved = load_human(path, torch, args.device);identities[name] = dict(run=str(path), sha256=checkpoint_file_sha256(Path(path) / 'human.pt'), wrapper=saved['wrapper'])
        models[name] = predict(model, store, test, torch, embed_tokens)
        if saved['wrapper']['bands']:models[f'{name}_band_unknown'] = predict(model, store, test, torch, embed_tokens, band=0)
        del model
    names = [n for n in parse_models(args.model)]
    pairs = [('champion', 'uniform_legal'), ('champion_recalibrated', 'champion')] + [(n, 'champion') for n in names] + \
            [(n, 'champion_recalibrated') for n in names] + [(n, 'pi_fp') for n in names if args.pi_fp] + \
            [(n, f'{n}_band_unknown') for n in names if f'{n}_band_unknown' in models] + \
            [(a, b) for a in names for b in names if a < b] + ([('pi_fp', 'champion')] if args.pi_fp else [])
    contrasts = [(a, b) for a in names for b in names if a != b] + [(n, 'champion') for n in names]
    np.savez(out / 'test-probs.npz', rows=test, replay=arrays['replay'][test], seat=arrays['seat'][test], turn=arrays['turn'][test],
             label=arrays['label'][test], partial=arrays['partial'][test], band=arrays['band'][test], **{f'probs_{k}': v for k, v in models.items()})
    report = dict(schema='v2s-human-reply-eval.v1', views=str(args.views), checkpoint=str(args.checkpoint),
                  checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), pi_fp=args.pi_fp,
                  pi_fp_sha256=checkpoint_file_sha256(args.pi_fp) if args.pi_fp else None, models=identities, recalibration=params,
                  recalibration_rows=int(len(fit)), held_out=dict(rows=int(len(test)), replays=int(len(np.unique(arrays['replay'][test])))),
                  test=report_rows(models, arrays, test, 'replay', pairs, band_groups(arrays, test), contrasts))
    if args.players:
        players = players_of(args.players);disjoint = player_disjoint(arrays, players, test)
        keep = np.isin(test, disjoint)
        report['player_disjoint'] = dict(rows=int(keep.sum()), replays=int(len(np.unique(arrays['replay'][disjoint]))),
            test=report_rows({k: v[keep] for k, v in models.items()}, arrays, disjoint, 'replay', pairs, band_groups(arrays, disjoint), contrasts))
    report['seconds'] = round(time.time() - started, 1)
    (out / 'report-human.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='EVALUATED', all={k: v['nll'] for k, v in report['test']['all']['models'].items()},
                          pairs={k: v for k, v in report['test']['all']['pairs'].items()})), flush=True)
    return 0


# ---- Foul Play transfer -------------------------------------------------------------------------------------------------------------

def foulplay(args):
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256
    set_exact(torch);out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    if (out / 'report-foulplay.json').exists():raise SystemExit(f'{out} already holds an evaluation')
    arrays = fpt.load(args.fpr)
    fpt.use_view(arrays, 'b');fpt.apply_redaction(arrays, 'peer-view-v2')
    test_seeds = fpt.read_test_seeds(args.test_seeds, arrays)
    split = fpt.split_rows(arrays, test_seeds, args.late_from)
    n = len(arrays['label'])
    arrays['partial'] = np.zeros(n, bool);arrays['kind'] = np.asarray(arrays['forced'], np.int64);arrays['locked'] = np.zeros(n, bool)
    arrays['group'] = group_ids(arrays['ordinal']);arrays['band'] = np.zeros(n, np.int64)
    specs = parse_models(args.model)
    store = Store(arrays, torch, args.device, events=store_events(specs, torch))
    champion, embed_tokens = plain_model(args.checkpoint, torch, args.device)
    pi_fp, _ = plain_model(args.pi_fp, torch, args.device)
    report = dict(schema='v2s-human-reply-foulplay.v1', fpr=str(args.fpr), test_seeds=args.test_seeds, late_from=args.late_from,
                  pi_fp=args.pi_fp, pi_fp_sha256=checkpoint_file_sha256(args.pi_fp), checkpoint_sha256=checkpoint_file_sha256(args.checkpoint),
                  models={}, splits={})
    for split_name in [s for s in ('test', 'late') if s in split]:
        rows = split[split_name]
        models = dict(uniform_legal=arrays['legal'][rows] / arrays['legal'][rows].sum(1, keepdims=True),
                      champion_stored_env=arrays['probs'][rows],
                      champion=predict(champion, store, rows, torch, embed_tokens, base_only=True),
                      pi_fp=predict(pi_fp, store, rows, torch, embed_tokens, base_only=True))
        for name, path in specs.items():
            model, embed_tokens_h, saved = load_human(path, torch, args.device)
            report['models'][name] = dict(run=str(path), sha256=checkpoint_file_sha256(Path(path) / 'human.pt'), wrapper=saved['wrapper'])
            if saved['wrapper']['bands']:
                for b, band in enumerate(BANDS):models[f'{name}_{band}'] = predict(model, store, rows, torch, embed_tokens_h, band=b)
            else:
                models[name] = predict(model, store, rows, torch, embed_tokens_h, band=0)
            del model
        names = [k for k in models if k not in ('uniform_legal', 'champion_stored_env', 'champion', 'pi_fp')]
        pairs = [('champion', 'uniform_legal'), ('pi_fp', 'champion')] + [(k, 'champion') for k in names] + [(k, 'pi_fp') for k in names] + \
                [(k, 'uniform_legal') for k in names]
        groups = dict(all=np.ones(len(rows), bool), move_request=~np.asarray(arrays['forced'])[rows], forced_switch=np.asarray(arrays['forced'])[rows])
        report['splits'][split_name] = dict(rows=int(len(rows)), games=int(len(np.unique(arrays['ordinal'][rows]))),
                                            seeds=int(len(np.unique(arrays['seed'][rows]))),
                                            result=report_rows(models, arrays, rows, 'seed', pairs, groups))
        np.savez(out / f'{split_name}-probs.npz', rows=rows, seed=arrays['seed'][rows], ordinal=arrays['ordinal'][rows], label=arrays['label'][rows],
                 **{f'probs_{k}': v for k, v in models.items()})
    report['seconds'] = round(time.time() - started, 1)
    (out / 'report-foulplay.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='EVALUATED', **{s: {k: v['nll'] for k, v in r['result']['all']['models'].items()} for s, r in report['splits'].items()})), flush=True)
    return 0


# ---- export ------------------------------------------------------------------------------------------------------------------------

def fold_band(state, band_table, band):
    """The champion state with band `band`'s per-token-type vector added into token_type_embedding (exact: the wrapper adds it
    to every token of that type, as the trunk adds its token-type row)."""
    out = dict(state);out['token_type_embedding.weight'] = state['token_type_embedding.weight'] + band_table[band]
    return out


def export(args):
    import torch
    from gpu_engine import champion as champion_mod
    from pokezero.neural_policy import checkpoint_file_sha256, load_transformer_checkpoint
    set_exact(torch);out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    checkpoint, archive = out / 'pi-human.pt', out / 'pi-human-weights.npz'
    if checkpoint.exists() or archive.exists():raise SystemExit(f'{out} already holds an export')
    model, embed_tokens, saved = load_human(args.run, torch, args.device)
    if saved['wrapper']['history']:raise SystemExit('a history model has no champion archive (its event inputs are not in the archive format)')
    band = BANDS.index(args.band)
    base_state = {k[len('base.'):]: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith('base.')}
    state = fold_band(base_state, model.band.detach().cpu(), band) if model.band is not None else base_state
    payload = dict(torch.load(saved['base_checkpoint'], map_location='cpu', weights_only=True))
    payload['state_dict'] = state;payload['value_calibration_transform'] = None;payload['value_calibration_source_checkpoint_sha256'] = None
    if isinstance(payload.get('model_config'), dict) and 'policy_id' in payload['model_config']:
        payload['model_config'] = dict(payload['model_config'], policy_id=f"human-reply-{payload['model_config']['policy_id']}")
    marker = dict(saved['human_reply'], band=args.band, source_run=str(args.run), source_sha256=checkpoint_file_sha256(Path(args.run) / 'human.pt'))
    payload['human_reply'] = marker
    torch.save(payload, checkpoint)
    arrays = load_views(args.views)
    test = np.flatnonzero(arrays['split'] == 'test')[:args.parity_rows]
    store = Store(arrays, torch, args.device)
    wrapped = predict(model, store, test, torch, embed_tokens, band=band)
    folded, _ = load_transformer_checkpoint(checkpoint, map_location=args.device)
    HumanReply, _ = build(torch)
    plain = predict(HumanReply(folded).to(args.device), store, test, torch, embed_tokens, base_only=True)
    fold_diff = float(np.abs(wrapped - plain).max())
    if not fold_diff <= FOLD_TOLERANCE:
        checkpoint.replace(str(checkpoint) + '.failed-fold');raise SystemExit(f'the folded checkpoint differs from the wrapper by {fold_diff}')
    sub = {k: arrays[k][test] for k in fpt.OBS_KEYS + ('legal',)}
    sub['label'] = np.clip(arrays['label'][test], 0, None)   # fp_reply_train.batch reads a label; the parity check does not use it
    extra = dict(marker, view='b', redaction='peer-view-v2', policy_only=True)
    parity = fpt.archive_parity(checkpoint, archive, sub, np.arange(len(test))[:, None], np.arange(len(test)), torch, extra)
    result = dict(schema='v2s-human-reply-export.v1', run=str(args.run), band=args.band, checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_file_sha256(checkpoint), fold_max_abs_probability_difference=fold_diff,
                  fold_tolerance=FOLD_TOLERANCE, archive=parity)
    (out / 'export.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(event='EXPORTED', **result)), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train');t.add_argument('views');t.add_argument('out');t.add_argument('--checkpoint', required=True)
    t.add_argument('--bands', action='store_true');t.add_argument('--history', action='store_true')
    t.add_argument('--history-events', type=int, default=32);t.add_argument('--epochs', type=int, default=4)
    t.add_argument('--batch', type=int, default=512);t.add_argument('--lr', type=float, default=3e-5);t.add_argument('--new-lr', type=float, default=3e-4)
    t.add_argument('--weight-decay', type=float, default=0.01);t.add_argument('--warmup', type=int, default=200)
    t.add_argument('--band-dropout', type=float, default=.15);t.add_argument('--eval-every', type=int, default=300)
    t.add_argument('--seed', type=int, default=0);t.add_argument('--device', default='cuda')
    e = sub.add_parser('evaluate');e.add_argument('views');e.add_argument('out');e.add_argument('--checkpoint', required=True)
    e.add_argument('--model', action='append', default=[]);e.add_argument('--pi-fp');e.add_argument('--players')
    e.add_argument('--recalibration-rows', type=int, default=100000);e.add_argument('--device', default='cuda')
    f = sub.add_parser('foulplay');f.add_argument('fpr');f.add_argument('out');f.add_argument('--checkpoint', required=True)
    f.add_argument('--pi-fp', required=True);f.add_argument('--test-seeds', required=True);f.add_argument('--late-from', type=int)
    f.add_argument('--model', action='append', default=[]);f.add_argument('--device', default='cuda')
    x = sub.add_parser('export');x.add_argument('run');x.add_argument('out');x.add_argument('--band', choices=BANDS, required=True)
    x.add_argument('--views', required=True);x.add_argument('--parity-rows', type=int, default=4096);x.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'train' and args.history and args.history_events < 1:parser.error('--history-events must be >= 1')
    return dict(train=train, evaluate=evaluate, foulplay=foulplay, export=export)[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
