"""Synthetic saved-trace subprocess for NEW recorder tests, never a game."""
import json
import os
from pathlib import Path
import sys

root=Path(sys.argv[1]);directory=Path(sys.argv[2]);mode=sys.argv[3];fd=int(sys.argv[4])
manifest=json.loads((root/'manifest.json').read_text())
audits=(root/'audit.jsonl').read_bytes().splitlines(keepends=True)
events=(root/'events.jsonl').read_bytes().splitlines(keepends=True)
stdout=(root/'stdout.jsonl').read_bytes().splitlines(keepends=True)
previous_stdout=0
(directory/'witness-source-registry.json').write_bytes(
    b'{"registry":{"synthetic":true},"sourceRegistryHash":"'+manifest['source_registry_hash'].encode()+b'"}\n'
    if mode!='bad-registry' else b'{"wrong":"observer"}\n')
os.write(sys.stderr.fileno(),b'original unparsed stderr \xff\n')

def append(name,data):
    with (directory/name).open('ab') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())

for index,received in enumerate(sys.stdin.buffer):
    command=manifest['commands'][index]
    assert received==(root/command['stdin']['path']).read_bytes(),'unexpected stdin bytes'
    assert (directory/'witness-control.json').read_bytes()==(root/command['native_control']['path']).read_bytes()
    assert (directory/'witness-transport-control.json').read_bytes()==(root/command['transport_control']['path']).read_bytes()
    if mode=='no-completion':continue
    native_wire=(root/command['native_receipt']['path']).read_bytes();native=json.loads(native_wire)
    transport_wire=(root/command['transport_receipt']['path']).read_bytes();transport=json.loads(transport_wire)
    append('witness-audit.jsonl',b''.join(audits[native['firstSequence']:native['lastSequence']+1]))
    append('witness-transport-events.jsonl',b''.join(events[slice(*transport['event_range'])]))
    append('witness-receipts.jsonl',native_wire);append('witness-transport-receipts.jsonl',transport_wire)
    # Exercise the completion/actual-stdout capture race without any timer as
    # drain evidence. The parent must await the actual saved stdout frontier.
    os.write(fd,transport_wire if mode!='bad-notification' else transport_wire.replace(b'"hash":"',b'"hash":"0',1))
    if mode=='bad-stdout':os.write(sys.stdout.fileno(),b'{"observer":"extra"}\n')
    os.write(sys.stdout.fileno(),b''.join(stdout[previous_stdout:transport['stdout_frontier']]))
    previous_stdout=transport['stdout_frontier']
    if json.loads(received)['type']=='close':
        if mode=='late-tail':os.write(sys.stdout.fileno(),b'late unaccounted stdout\n')
        if mode=='nonzero':sys.exit(7)
        break
