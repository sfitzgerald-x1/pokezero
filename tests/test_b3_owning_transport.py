"""Owning-only transport hardening; literal fake clients, no games/models."""
import asyncio
from dataclasses import replace
from enum import IntEnum
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace as NS

from pokezero import foulplay_bridge as b
import test_live_protocol_bridge as fixtures


class OwningTransportTests(unittest.TestCase):
    def config(self,**updates):
        return b.ControlledFoulPlayConfig(checkpoint=Path('/authored-champion'),showdown_root=Path('/authored-source'),
            policy_mode='raw',device='cpu',belief_set_source=True,allow_search_fallback=False,
            opponent_crash_retries=0,encoder_kind='jax-owning',seed_start=7,games=1,**updates)

    def test_owning_child_connect_kwargs_seed_and_argv_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'websockets.py').write_text('def connect(*args,**kwargs):return dict(args=args,kwargs=kwargs)\n')
            (root/'run.py').write_text('import json,random,sys,websockets\nprint(json.dumps(dict(transport=websockets.connect("ws://literal"),draw=random.random(),argv=sys.argv)))\n')
            config=self.config(foulplay_root=root,foulplay_python=Path(sys.executable),foulplay_random_seed=456)
            command=b._foulplay_command(config,'ws://literal',run_count=1)
            result=subprocess.run(command,cwd=root,env=b._foulplay_env(config),capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            payload=json.loads(result.stdout)
            self.assertEqual(payload['transport'],dict(args=['ws://literal'],kwargs=dict(ping_interval=None)))
            self.assertEqual(payload['draw'],random.Random(456).random())
            self.assertEqual(payload['argv'],[str(root/'run.py'),*command[4:]])
            self.assertEqual(command[command.index('--search-time-ms')+1],'1000')
            self.assertEqual(command[command.index('--search-parallelism')+1],'1')
            self.assertEqual(command[command.index('--search-threads')+1],'1')
            self.assertEqual(config.opponent_crash_retries,0)
            self.assertIn('ping_interval=None',b._foulplay_command(replace(config,encoder_kind='python-owning'),'ws://literal')[2])
            pinned=b._foulplay_command(replace(config,foulplay_cpu_affinity=7),'ws://literal')
            self.assertTrue(pinned[2].endswith(b._foulplay_command(config,'ws://literal')[2]))
            self.assertIn('os.sched_setaffinity(0, {7})',pinned[2])

    def test_legacy_wrapper_and_explicit_deadlines_unchanged(self):
        legacy=b.ControlledFoulPlayConfig(checkpoint=Path('/literal'),showdown_root=Path('/literal-source'))
        expected=("import os, random, runpy, sys; "
            "random.seed(int(os.environ['POKEZERO_FOULPLAY_RANDOM_SEED'])); "
            "script = sys.argv[1]; sys.argv = sys.argv[1:]; "
            "runpy.run_path(script, run_name='__main__')")
        self.assertEqual(b._foulplay_command(legacy,'ws://literal')[2],expected)
        config=self.config(owning_decision_timeout_seconds=60,owning_terminal_timeout_seconds=5)
        self.assertEqual((b._FoulPlayWebsocketServer._CHALLENGE_TIMEOUT_SECONDS,
            b._FoulPlayWebsocketServer._CHOICE_TIMEOUT_SECONDS,
            config.owning_decision_timeout_seconds,config.owning_terminal_timeout_seconds),(30.,120.,60,5))

    def test_disconnect_snapshot_is_bounded_and_not_a_reconnect(self):
        class Code(IntEnum):ABNORMAL=1006
        class Socket:
            close_code=Code.ABNORMAL;close_reason='r'*1000
            async def send(self,value):pass
            def __aiter__(self):return self
            async def __anext__(self):raise OSError('unretained private payload')
        async def run():
            server=b._FoulPlayWebsocketServer(username='literal',host='127.0.0.1')
            await server._handle_connection(Socket())
            self.assertIsNone(server.websocket)
            self.assertEqual(server.last_disconnect,dict(error_type='OSError',close_code=1006,close_reason='r'*256))
            with self.assertRaises(b.FoulPlayProtocolError) as caught:await server.send_room_lines('literal',['|turn|1'])
            self.assertEqual(caught.exception.transport_diagnostics,dict(last_disconnect=server.last_disconnect))
            self.assertNotIn('unretained',json.dumps(caught.exception.transport_diagnostics))
        asyncio.run(run())

    def test_old_handler_cannot_clear_successor_or_its_evidence(self):
        async def run():
            server=b._FoulPlayWebsocketServer(username='literal',host='127.0.0.1');successor=object()
            class Old:
                async def send(self,value):pass
                def __aiter__(self):return self
                async def __anext__(self):
                    server.websocket=successor;server.last_disconnect={'successor':'retained'}
                    raise StopAsyncIteration
            await server._handle_connection(Old())
            self.assertIs(server.websocket,successor);self.assertEqual(server.last_disconnect,{'successor':'retained'})
        asyncio.run(run())

    def test_failed_owning_receipt_preserves_transport_and_bounded_child_logs(self):
        fixture=fixtures.OwningBridgeTests('test_missing_mismatched_factory_or_backend_cannot_fall_back')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=fixture.config()
        server=b._FoulPlayWebsocketServer(username='literal',host='127.0.0.1')
        server.last_disconnect=dict(error_type='ConnectionClosedError',close_code=1011,close_reason='keepalive ping timeout')
        error=server._disconnected_error();logs=b._ProcessLogBuffer(stdout=['s'*1000]*20,stderr=['e'*1000]*20)
        with patch.object(b,'_run_single_game_body',new_callable=AsyncMock,side_effect=error):
            with self.assertRaises(b.FoulPlayProtocolError) as caught:
                asyncio.run(b._run_single_game(config=config,seed=7,
                    owning_actor_factory=fixture.factory(),actor_runtime=(None,None,None),
                    foulplay_process=NS(returncode=1),foulplay_logs=logs))
        receipt=caught.exception.owning_encoder_receipt;transport=receipt['transport_diagnostics']
        self.assertEqual(transport['last_disconnect']['close_code'],1011)
        self.assertEqual(transport['foulplay_returncode'],1)
        self.assertEqual(transport['stdout_tail'],['s'*256]*8)
        self.assertEqual(transport['stderr_tail'],['e'*256]*8)
        self.assertEqual(receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(receipt['choice_acknowledgements']['submissions'],0)
        self.assertFalse(receipt['B3_qualified'])


if __name__=='__main__':unittest.main()
