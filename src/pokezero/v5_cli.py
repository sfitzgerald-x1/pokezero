"""Explicit train-only v5 entrypoint; legacy CLI/loaders remain unchanged."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

from .v5_model import V5ModelConfig
from .v5_loss import V5LossConfig
from . import v5_training as training


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train',help='Train explicit v5 caches; no rollout or iteration command.')
    train.add_argument('--enable-v5',action='store_true')
    origin=train.add_mutually_exclusive_group(required=True)
    origin.add_argument('--config',type=Path,help='Fresh scratch JSON model/loss/training metadata.')
    origin.add_argument('--resume',type=Path,help='Explicit same-v5 checkpoint and original cache cursor.')
    train.add_argument('--cache',type=Path,action='append',required=True)
    train.add_argument('--out',type=Path,required=True)
    train.add_argument('--summary',type=Path)
    train.add_argument('--max-updates',type=int)
    args=parser.parse_args(argv)
    if args.enable_v5 is not True:raise ValueError('explicit v5 trainer enablement required')
    if args.out.exists():raise FileExistsError(args.out)
    if args.summary is not None and args.summary.exists():raise FileExistsError(args.summary)
    if args.resume is not None:
        runtime=training.load_checkpoint(args.resume,enable_v5=True)
    else:
        config=json.loads(args.config.read_text())
        model=V5ModelConfig.from_metadata(config['model'])
        loss=V5LossConfig(**{field.name:config['loss'][field.name] for field in fields(V5LossConfig)})
        if training._sha(loss.metadata())!=training._sha(config['loss']):
            raise ValueError('explicit v5 loss metadata required')
        runtime=training.create(model,loss,training.TrainConfig(**config['training']),enable_v5=True)
    report=training.train_caches(runtime,args.cache,max_updates=args.max_updates)
    report['checkpoint']=training.save_checkpoint(args.out,runtime)
    if args.summary is not None:
        with args.summary.open('x') as stream:json.dump(report,stream,sort_keys=True,allow_nan=False)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
