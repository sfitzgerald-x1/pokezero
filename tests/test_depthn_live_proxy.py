import asyncio
import json
import socket
import struct
import unittest

from scripts import depthn_live_proxy as P


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0));return s.getsockname()[1]


async def framed_backend(name, delay, served):
    """A backend speaking the search-server framing: read one length-prefixed request, answer {backend, echo} after `delay`."""
    async def handle(reader, writer):
        try:
            n = struct.unpack('!I', await reader.readexactly(4))[0];body = json.loads(await reader.readexactly(n))
            served.append(name);await asyncio.sleep(delay)
            out = json.dumps(dict(backend=name, echo=body)).encode();writer.write(struct.pack('!I', len(out)) + out);await writer.drain()
        except asyncio.IncompleteReadError:pass
        finally:writer.close()
    return await asyncio.start_server(handle, '127.0.0.1', 0)


async def request(port, body):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    raw = json.dumps(body).encode();writer.write(struct.pack('!I', len(raw)) + raw);await writer.drain()
    try:
        n = struct.unpack('!I', await reader.readexactly(4))[0];return json.loads(await reader.readexactly(n))
    except asyncio.IncompleteReadError:return None
    finally:writer.close()


class ProxyTests(unittest.TestCase):
    def run_proxy(self, scenario, backends_spec, **kw):
        async def main():
            served, servers, backends = [], [], []
            for name, delay in backends_spec:
                if delay is None:backends.append(('127.0.0.1', free_port()));continue   # nothing listens there
                s = await framed_backend(name, delay, served);servers.append(s);backends.append(('127.0.0.1', s.sockets[0].getsockname()[1]))
            box, lines = {}, []
            task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lines.append, report_seconds=0,
                started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1], pool=pool), **kw))
            for _ in range(200):
                if 'port' in box or task.done():break
                await asyncio.sleep(.01)
            if task.done():task.result()
            try:return await scenario(box['port'], box['pool'], served, lines)
            finally:
                task.cancel()
                for s in servers:s.close()
        return asyncio.run(main())

    def test_forwards_byte_for_byte_and_reports_ready(self):
        async def scenario(port, pool, served, lines):
            out = await request(port, dict(request_id='r1', lines=['|request|{}']));return out, lines
        out, lines = self.run_proxy(scenario, [('a', 0.)])
        self.assertEqual(out, dict(backend='a', echo=dict(request_id='r1', lines=['|request|{}'])))
        self.assertEqual(json.loads(lines[0])['event'], 'READY')

    def test_concurrent_requests_spread_to_the_least_busy(self):
        async def scenario(port, pool, served, lines):
            outs = await asyncio.gather(*(request(port, dict(i=i)) for i in range(3)))
            for _ in range(100):   # the proxy sees each client's close just after the response: in-flight counts drain to 0
                if not any(pool.active.values()):break
                await asyncio.sleep(.01)
            return sorted(o['backend'] for o in outs), pool.snapshot()
        names, snap = self.run_proxy(scenario, [('a', .3), ('b', .3), ('c', .3)])
        self.assertEqual(names, ['a', 'b', 'c'])   # three in flight at once: one per backend
        self.assertEqual([b['active'] for b in snap['backends']], [0, 0, 0])
        self.assertEqual(sorted(b['served'] for b in snap['backends']), [1, 1, 1])

    def test_sequential_requests_rotate(self):
        async def scenario(port, pool, served, lines):
            return [(await request(port, dict(i=i)))['backend'] for i in range(4)]
        self.assertEqual(sorted(set(self.run_proxy(scenario, [('a', 0.), ('b', 0.)]))), ['a', 'b'])

    def test_a_refusing_backend_is_skipped_and_counted(self):
        async def scenario(port, pool, served, lines):
            outs = [await request(port, dict(i=i)) for i in range(3)];return [o['backend'] for o in outs], pool.snapshot()
        names, snap = self.run_proxy(scenario, [('a', 0.), ('dead', None)])
        self.assertEqual(names, ['a', 'a', 'a']);self.assertGreaterEqual(snap['backends'][1]['refused'], 1)

    def test_no_backend_closes_the_client_connection(self):
        async def scenario(port, pool, served, lines):
            return await request(port, dict(i=0)), pool.unserved
        out, unserved = self.run_proxy(scenario, [('dead1', None), ('dead2', None)])
        self.assertIsNone(out);self.assertEqual(unserved, 1)

    def test_wait_ready_refuses_when_a_backend_never_listens(self):
        async def main():
            dead = ('127.0.0.1', free_port())
            with self.assertRaises(SystemExit):await P.run(('127.0.0.1', 0), [dead], wait_ready_seconds=.6, report_seconds=0, log=lambda line: None)
        asyncio.run(main())

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):P.main(['--listen', 'nohost', '--backend', '127.0.0.1:1'])
        with self.assertRaises(ValueError):P.Pool([('h', 1), ('h', 1)])


if __name__ == '__main__':
    unittest.main()
