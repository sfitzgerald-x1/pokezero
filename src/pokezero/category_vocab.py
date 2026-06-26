"""Direct string→row category vocabulary for the closed Gen 3 randbat universe.

Replaces the legacy ``stable_category_id`` 1,000,000-bucket hash: every categorical token
string the encoder emits is mapped to a small, stable embedding-row index by direct lookup in
an enumerated, sorted vocabulary. Row 0 is padding; rows ``1..len(tokens)`` are the closed
vocabulary; the final ``oov_buckets`` rows are a graceful-degradation safety net for any string
not in the closed universe (collisions only ever occur among genuinely out-of-vocabulary
strings, which the lean encoding makes vanishingly rare).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Iterable, Mapping

logger = logging.getLogger("pokezero.category_vocab")


def normalize_category_value(value: str | None) -> str:
    """Match the encoder's normalization: stripped, lowercased."""
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class CategoryVocabulary:
    """Maps normalized category strings to compact embedding rows."""

    tokens: tuple[str, ...]
    oov_buckets: int = 16
    # alias token -> base token (e.g. cosmetic-forme species collapsed onto the base species).
    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.oov_buckets < 1:
            raise ValueError("oov_buckets must be >= 1.")
        index: dict[str, int] = {}
        for row, token in enumerate(self.tokens, start=1):
            normalized = normalize_category_value(token)
            if not normalized:
                raise ValueError("category vocabulary tokens must be non-empty.")
            if normalized in index:
                raise ValueError(f"duplicate category token after normalization: {normalized!r}.")
            index[normalized] = row
        # Resolve aliases to the base token's row (base must be in the vocabulary).
        for alias, base in self.aliases.items():
            base_row = index.get(normalize_category_value(base))
            if base_row is not None:
                index[normalize_category_value(alias)] = base_row
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_oov_offset", 1 + len(self.tokens))
        # Drift signal: distinct strings that fell through to the OOV band. In the closed
        # randbat universe this stays empty; anything here means a token the encoder emits is
        # missing from the vocabulary (dex update, un-aliased forme, new token kind, or a bug).
        object.__setattr__(self, "_observed_oov", set())

    @property
    def size(self) -> int:
        """Total embedding rows: padding + vocabulary + OOV buckets."""
        return 1 + len(self.tokens) + self.oov_buckets

    @property
    def observed_oov_tokens(self) -> frozenset[str]:
        """Distinct out-of-vocabulary strings seen by :meth:`encode` (drift signal; empty is healthy)."""
        return frozenset(self._observed_oov)  # type: ignore[attr-defined]

    def encode(self, value: str | None) -> int:
        """Return the embedding row for a category string (0 = padding for empty)."""
        normalized = normalize_category_value(value)
        if not normalized:
            return 0
        row = self._index.get(normalized)  # type: ignore[attr-defined]
        if row is not None:
            return row
        # Out-of-vocabulary: deterministic bucket in the reserved safety-net block.
        digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
        oov_row = self._oov_offset + (int.from_bytes(digest, "big") % self.oov_buckets)  # type: ignore[attr-defined]
        observed = self._observed_oov  # type: ignore[attr-defined]
        if normalized not in observed:
            # Warn once per distinct token so genuine drift is visible without flooding hot paths.
            observed.add(normalized)
            logger.warning(
                "category-vocab drift: %r is outside the closed Gen 3 randbat universe; "
                "hashed to safety-net row %d. This token should be enumerated, not hashed.",
                normalized,
                oov_row,
            )
        return oov_row


def build_category_vocabulary(
    tokens: Iterable[str],
    *,
    oov_buckets: int = 16,
    aliases: Mapping[str, str] | None = None,
) -> CategoryVocabulary:
    """Build a vocabulary from category strings (deduped + sorted for a stable row order)."""
    sorted_tokens = tuple(sorted({normalize_category_value(t) for t in tokens if normalize_category_value(t)}))
    return CategoryVocabulary(tokens=sorted_tokens, oov_buckets=oov_buckets, aliases=dict(aliases or {}))
