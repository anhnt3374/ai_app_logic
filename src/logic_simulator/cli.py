import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from .config import load_config,SimulatorConfig,sanitize


def main():
    parser=argparse.ArgumentParser(description='MP4 face recognition logic simulator')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['run','validate','doctor','provision']:
        sub=commands.add_parser(name)
        sub.add_argument('--config',type=Path,default=Path('configs/host-cpu.yaml'))
        if name=='run':
            sub.add_argument('--provider',choices=['mock','real','replay'])
            sub.add_argument('--input')
            sub.add_argument('--output-root')
            sub.add_argument('--replay-from')
            sub.add_argument('--replay-policy',choices=['strict','fill_missing'])
            sub.add_argument('--logic-snapshot')
        if name in ['doctor','provision']:
            sub.add_argument('--target',default='host_cpu')
    fetch=commands.add_parser('fetch-models')
    fetch.add_argument('--registry',default='configs/models.yaml')
    demo=commands.add_parser('demo')
    demo.add_argument('--output',default='data/input/demo.mp4')
    demo.add_argument('--image')
    demo.add_argument('--frames',type=int,default=90)
    enrollment=commands.add_parser('enroll')
    enrollment.add_argument('--images',required=True)
    enrollment.add_argument('--output',required=True)
    enrollment.add_argument('--registry',default='configs/models.yaml')
    enrollment.add_argument('--face-model',default='yunet')
    enrollment.add_argument('--embedding-model',default='sface')
    compare=commands.add_parser('compare')
    compare.add_argument('left',type=Path)
    compare.add_argument('right',type=Path)
    commands.add_parser('worker')
    args=parser.parse_args()
    try:
        if args.command=='worker':
            from .inference.worker import main as worker
            worker()
        elif args.command=='demo':
            from .assets import make_demo
            if args.frames<=0:
                raise ValueError('frames must be positive')
            print(make_demo(args.output,args.image,args.frames))
        elif args.command=='fetch-models':
            from .assets import fetch_models
            fetch_models(args.registry)
        elif args.command=='enroll':
            from .assets import enroll
            gallery=enroll(args.images,args.output,args.registry,args.face_model,args.embedding_model)
            print(json.dumps({'people':len(gallery['people']),'output':args.output}))
        elif args.command=='compare':
            a=[json.loads(v) for v in (args.left/'events.jsonl').read_text().splitlines()]
            b=[json.loads(v) for v in (args.right/'events.jsonl').read_text().splitlines()]
            print(json.dumps({'events_equal':a==b,'left_events':len(a),'right_events':len(b)}))
            return 0 if a==b else 1
        else:
            config=load_config(args.config)
            if args.command=='validate':
                print(json.dumps(sanitize(config.model_dump(mode='json',exclude_none=True)),indent=2))
            elif args.command=='provision':
                from .assets import provision_models
                print(json.dumps(provision_models(config.targets[args.target],config.models['registry']),indent=2))
            elif args.command=='doctor':
                from .inference.transport import WorkerClient
                if args.target not in config.targets:
                    raise ValueError('Unknown target: '+args.target)
                target=config.targets[args.target]
                client=WorkerClient(target,config.execution.worker_timeout_s)
                try:
                    result=client.call({'request_id':uuid4().hex,'command':'probe','target':target.model_dump(exclude={'connection'})})
                    print(json.dumps({'status':'runtime_available','model_inference':'not_yet_verified','backend':result['backend']},indent=2))
                finally:
                    client.close()
            else:
                if args.provider:
                    for stage in config.stages.values():
                        stage.provider=args.provider
                if args.input:
                    config.input.path=args.input
                if args.output_root:
                    config.output.root=args.output_root
                if args.replay_from:
                    config.replay.source=args.replay_from
                    for stage in config.stages.values():
                        stage.provider='replay'
                if args.replay_policy:
                    config.replay.policy=args.replay_policy
                config=SimulatorConfig.model_validate(config.model_dump(mode='json',exclude_none=True))
                from .engine import run
                result=run(config,args.logic_snapshot)
                print(json.dumps({'status':result.status,'run':str(result.path)},ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print('Interrupted',file=sys.stderr)
        return 130
    except Exception as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
