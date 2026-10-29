"""Raw item callbacks in the complete ordered encoder fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ItemStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_belief
        from pokezero.category_vocab import CategoryVocabulary
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        old=universes['snorlax'];universes['snorlax']=replace(old,variants=tuple(
            replace(v,variant_id=v.variant_id+str(i),item=item) for v in old.variants for i,item in enumerate(('Leftovers','Choice Band'))))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        labels=set(f.vocab.tokens)|{'belief:possible_item:choiceband'};vocab=CategoryVocabulary(tuple(sorted(labels)))
        f.catalog=f.e.compile_encoder(source,f.catalog.dex,vocab,checkpoint_sha256='a'*64,feature_masks=f.masks)
        f.source,f.vocab=source,vocab
        f.public=SimpleNamespace(source=raw,prefix=('|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100'))
        cls.f,cls.j,cls.jax=f,f.j,f.jax

    def test_raw_item_and_common_hp_order_matches_source(self):
        f=self.f
        lines=[*f.public.prefix,'|-heal|p1a: Snorlax|90/100|[from] item: Leftovers',
            '|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off',
            '|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off',
            '|-item|p1a: Snorlax|Choice Band','|-item|p1a: Snorlax|Choice Band',
            '|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers']
        for end in range(2,len(lines)+1):f.compare_source(lines[:end])

    def test_actual_trick_pairing_and_two_item_count(self):
        f=self.f
        lines=[*f.public.prefix,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol',
            '|-item|p2a: Absol|Choice Band|[from] move: Trick',
            '|-item|p1a: Snorlax|Leftovers|[from] move: Trick']
        for end in range(3,len(lines)+1):f.compare_source(lines[:end])
