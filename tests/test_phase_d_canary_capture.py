"""Inert actual endpoint callsite controls; no models, games or subprocesses."""
import asyncio
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from pokezero import phase_d_canary_capture as install, phase_d_owning_capture as capture
from pokezero import b4_canary_endpoint as endpoint, b4_device_canary_runtime as runtime
from pokezero import live_protocol_actor as actor
import test_phase_d_prerequisites as fixture


class CallsiteTests(unittest.TestCase):
    def bound(self, stack, *, worker=None, spawn=None):
        stack.enter_context(patch.object(install, '_INSTALLED', None))
        stack.enter_context(patch.object(runtime.Factory, 'actor', runtime.Factory.actor))
        stack.enter_context(patch.object(endpoint, 'worker', worker or endpoint.worker))
        stack.enter_context(patch.object(endpoint, '_spawn_owned', spawn or endpoint._spawn_owned))

    def factory(self):
        class Backend(actor.JaxBackend):
            def __init__(self, contract):actor.OwnedProvider.__init__(self, contract)
            ingest=fixture.Backend.ingest
            encode=fixture.Backend.encode
        return runtime.Factory('jax-owning', actor.ActorContract.champion(),
            lambda c, **kw:Backend(c),
            lambda c, **kw:actor.RawPolicy(c,lambda obs, *, rng:actor.PolicyDecision(0,'fixture'),lambda:None))

    def test_actual_canary_factory_and_worker_scope_reach_threaded_decision(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);path=root/'request.json'
            request=dict(run_directory=str(root),cell={'ordinal':7},declaration_sha256='a'*64)
            blob=capture.encoded(request);path.write_bytes(blob);digest=hashlib.sha256(blob).hexdigest()
            seen=[];sentinel=object()
            async def original_worker(actual_path, actual_sha, **options):
                self.assertEqual((actual_path,actual_sha),(path,digest));self.assertIs(options['runtime'],sentinel)
                scope=capture._SCOPE.get();self.assertEqual(scope[0],root/'phase-d-owning-wire');seen.append('original-worker')
                a=await asyncio.to_thread(self.factory().actor,(object(),object(),object()))
                a.begin_game(battle_id='fixture',owner='p1',seed=1,stream_id=9)
                fixture.feed(a,('|turn|12',fixture.request()));snapshot=a.snapshot_boundary()
                rng=random.Random(4);before=rng.getstate()
                decision=await asyncio.to_thread(a.decide,snapshot,rng=rng)
                self.assertEqual(decision.action_index,0);self.assertEqual(a.choice(snapshot),'move 1')
                self.assertEqual(before,rng.getstate());a.close_game();return 37
            self.bound(stack,worker=original_worker)
            install.install_endpoint_capture()
            first=(runtime.Factory.actor,endpoint.worker,endpoint._spawn_owned)
            install.install_endpoint_capture();self.assertEqual(first,(runtime.Factory.actor,endpoint.worker,endpoint._spawn_owned))
            self.assertEqual(asyncio.run(endpoint.worker(path,digest,runtime=sentinel,expected_pair=(2,3))),37)
            self.assertEqual(seen,['original-worker']);self.assertIsNone(capture._SCOPE.get())
            roots=list(capture.read_capture(root/'phase-d-owning-wire/7.jsonl'))
            self.assertEqual([(r['turn'],r['decision_index']) for r in roots],[(12,0)])

    def test_original_launcher_keeps_arguments_and_ownership_except_worker_entry(self):
        with ExitStack() as stack:
            spawn=AsyncMock(return_value='original-process');self.bound(stack,spawn=spawn)
            install.install_endpoint_capture()
            kwargs=dict(stdin='same-stdin',stdout='same-stdout',stderr='same-stderr',start_new_session=True)
            args=('/original/python','-B','-m','pokezero.b4_device_canary_runtime','--cpu-pair','2','3')
            self.assertEqual(asyncio.run(endpoint._spawn_owned(*args,**kwargs)),'original-process')
            spawn.assert_awaited_once_with('/original/python','-B',str(Path(install.__file__).resolve()),'--cpu-pair','2','3',**kwargs)
            spawn.reset_mock()
            other=('/original/python','-B','-m','pokezero.unrelated')
            asyncio.run(endpoint._spawn_owned(*other,**kwargs));spawn.assert_awaited_once_with(*other,**kwargs)

    def test_actual_original_worker_claim_and_same_runtime_run_are_inside_capture(self):
        import test_b4_canary_endpoint as controls
        from pokezero import b4_fidelity_canary as c
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(c,'DECLARED_GAMES',4));stack.enter_context(patch.object(c,'PANEL_SEEDS',2))
            root=(Path(folder)/'run').resolve();recipe,checkpoint=controls.ConcurrentEndpointTests().prepare(root,pairs=((0,1),))
            cell=endpoint.cells(recipe)[1]  # actual CANARY checkpoint, no fake control file
            request=dict(schema='pokezero.b4.canary.fp-request.v1',run_directory=str(root),declaration_sha256=c.file_sha(root/'declaration.json'),
                cell=cell,checkpoint=checkpoint,checkpoint_sha256=c.file_sha(checkpoint),cpu_pair=[0,1])
            endpoint._write(root/'fp-requests/1.json',request);digest=c.sha(request)
            endpoint._write(root/'fp-starts/1.json',endpoint._start(request,digest))
            endpoint._write(root/'fp-snapshot.json',dict(declaration_sha256=request['declaration_sha256'],
                CANARY=dict(checkpoint=checkpoint,sha256=request['checkpoint_sha256'])))
            calls=[]
            class Runtime:
                async def run(inner, config, *, expected_checkpoint_sha256):
                    self.assertTrue((root/'fp-claims/1.json').is_file())
                    self.assertEqual(capture._SCOPE.get()[1],cell);calls.append(config)
                    value=controls.literal(request,recipe)
                    return NS(config=config,checkpoint_sha256=expected_checkpoint_sha256,
                        foulplay_random_seed_schedule=(config.foulplay_random_seed,),
                        games=(NS(to_dict=lambda:value['source_game']),),to_dict=lambda:value['benchmark'])
            stack.enter_context(patch.object(endpoint,'pin_owned_actor'))
            self.bound(stack);install.install_endpoint_capture()
            self.assertEqual(asyncio.run(endpoint.worker(root/'fp-requests/1.json',digest,runtime=Runtime(),expected_pair=(0,1))),0)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0].search_time_ms,1000)
            self.assertEqual(calls[0].seed_start,recipe.fp_configs[0]['seed_start'])
            value=endpoint._read_result(root/'fp-results/1.json');self.assertEqual(value['cell'],cell);self.assertEqual(value['score'],1.)
            self.assertIsNone(capture._SCOPE.get())

    def test_private_capture_source_is_digest_bound_and_forwarded_to_child(self):
        import sys
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);source=root/'phase_d_owning_capture.py'
            source.write_bytes(Path(capture.__file__).read_bytes());digest=hashlib.sha256(source.read_bytes()).hexdigest()
            spawn=AsyncMock(return_value='original-process');self.bound(stack,spawn=spawn)
            stack.enter_context(patch.dict(sys.modules));sys.modules.pop('pokezero.phase_d_owning_capture',None)
            install.install_endpoint_capture(capture_source=source,expected_capture_sha256=digest)
            loaded=sys.modules['pokezero.phase_d_owning_capture']
            self.assertEqual(Path(loaded.__file__),source.resolve())
            asyncio.run(endpoint._spawn_owned('/python','-B','-m','pokezero.b4_device_canary_runtime','--cpu-pair','0','1'))
            spawn.assert_awaited_once_with('/python','-B',str(Path(install.__file__).resolve()),
                '--capture-source',str(source.resolve()),'--expected-capture-sha256',digest,'--cpu-pair','0','1')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'wrong.py';source.write_bytes(b'raise AssertionError("must not import")')
            with self.assertRaisesRegex(ValueError,'reviewed private'):install._capture(source,'a'*64)

    def test_request_hash_refuses_before_original_worker_and_no_capture_directory(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path=Path(folder)/'request.json';path.write_bytes(b'{}')
            worker=AsyncMock();self.bound(stack,worker=worker);install.install_endpoint_capture()
            with self.assertRaisesRegex(ValueError,'digest'):asyncio.run(endpoint.worker(path,'a'*64))
            worker.assert_not_awaited();self.assertFalse((Path(folder)/'phase-d-owning-wire').exists())


if __name__=='__main__':unittest.main()
