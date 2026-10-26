import hashlib
import json
import shutil
import platform
from importlib.metadata import distributions
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import yaml
from .config import sanitize


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class RunContext:
    path: Path
    status: str = 'running'


def create_run(config,logic_snapshot=None):
    stamp=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%dT%H%M%S%z')
    path=Path(config.output.root)/config.logic_id/f'{stamp}-{uuid4().hex[:12]}'
    path.mkdir(parents=True,exist_ok=False)
    (path/'evidence').mkdir()
    data=config.model_dump(mode='json',exclude_none=True)
    (path/'config.resolved.yaml').write_text(yaml.safe_dump(sanitize(data),sort_keys=False,allow_unicode=True))
    source=Path(logic_snapshot) if logic_snapshot else Path(config.logic_root)/config.logic_id
    if source.is_dir():
        shutil.copytree(source,path/'logic',ignore=shutil.ignore_patterns('__pycache__'))
    package=Path(__file__).parent
    shutil.copytree(package,path/'source'/'logic_simulator',ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob('*.py'))}
    requirements=Path('requirements.txt')
    if requirements.is_file():
        shutil.copy2(requirements,path/'requirements.txt')
    dependencies={dist.metadata['Name']:dist.version for dist in distributions() if dist.metadata['Name']}
    (path/'requirements.lock.txt').write_text(''.join(f'{name}=={version}\n' for name,version in sorted(dependencies.items())))
    run=RunContext(path)
    finish_run(run,'running',{'logic_id':config.logic_id,'config_hash':digest(sanitize(data)),
                            'source_hash':digest(hashes),'source_files':hashes,
                            'python':platform.python_version(),'platform':platform.platform(),
                            'dependencies':dependencies})
    return run


def finish_run(run, status, summary):
    run.status=status
    manifest_path=run.path/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(sanitize(summary))
    manifest['status']=status
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    (run.path/'README.md').write_text(
        '# Simulator run\n\n'
        f'Logic: `{manifest.get("logic_id", "unknown")}`\n\nStatus: **{status}**\n\n'
        'Person/face: periodic detection (default 18 FPS). Embedding: logic selects '
        'one eligible face per turn by round-robin (see resolved business policy).\n\n'
        '## Reproduce\n\nFrom project root (replace RUN with this run directory):\n\n'
        '```sh\nPYTHONPATH=RUN/source python -m logic_simulator run '
        '--config RUN/config.resolved.yaml --logic-snapshot RUN/logic\n```\n\n'
        'Keep original input/models/gallery mounted at resolved paths. Install the '
        'saved requirements.txt to reproduce the Python dependencies.\n\n'
        '## Artifacts and results\n\n'
        'Config: `config.resolved.yaml`; model/input/logic fingerprints below. '
        'Events: `events.jsonl`; selection reasons: `trace.jsonl`; inference: '
        '`inference.jsonl`; video: `output.mp4` if enabled; metrics: `metrics.json`.\n\n'
        '```json\n'+json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+'\n```\n')


def append_json(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')
