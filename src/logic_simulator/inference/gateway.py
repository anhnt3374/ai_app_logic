import hashlib
from pathlib import Path
from uuid import uuid4
import yaml

from .cache import Cache
from .mock import MockProvider
from .transport import WorkerClient
from .adapters import ADAPTER_VERSION
from ..config import resolve_environment
from ..run_store import digest,file_hash


class InferenceGateway:
    def __init__(self,config,run_path,video_hash):
        self.config=config
        self.path=run_path
        self.video_hash=video_hash
        self.registry=yaml.safe_load(Path(config.models['registry']).read_text())
        scenario_path=config.mock.get('scenario')
        self.scenario=yaml.safe_load(Path(scenario_path).read_text()) if scenario_path else {'people':[]}
        self.mock=MockProvider(self.scenario,config.execution.seed)
        self.output=Cache(run_path/'inference.jsonl')
        self.output.path.touch()
        source=Path(config.replay.source) if config.replay.source else None
        if source and source.is_dir():
            source=source/'inference.jsonl'
        if source and not source.is_file():
            raise ValueError('Replay source does not exist')
        self.source=Cache(source) if source else None
        self.workers={}
        self.counts={'mock':0,'real':0,'replay':0}
        self.backends={}

    def infer(self,stage,image,frame,scheduled=None,context=None):
        stage_config=self.config.stages[stage]
        provider=stage_config.provider
        manifest=self.registry[stage_config.model]
        context=context or {}
        target=self.config.targets[stage_config.target]
        kinds=sorted(self.source.kinds.get(stage,{'real'})) if provider=='replay' and self.source else [provider]
        def signature_for(kind):
            source_fingerprint=digest([self.scenario,self.config.execution.seed,'mock-v1']) if kind=='mock' else 'real'
            return {'schema_version':1,'video':self.video_hash,'frame_id':frame.frame_id,
                'source_pts':str(frame.original_pts),'stage':stage,'model':manifest,
                'adapter_version':ADAPTER_VERSION,'pixels':hashlib.sha256(image.tobytes()).hexdigest(),
                'shape':image.shape,'frame_shape':frame.pixels.shape,'coordinate_system':'input_pixels',
                'context':context,'source_fingerprint':source_fingerprint,
                'runtime':[target.runtime,target.device,target.provider,target.runtime_options] if kind=='real' else 'mock'}
        kind=kinds[0]
        signature=signature_for(kind)
        key=digest(signature)
        request={'request_id':uuid4().hex,'stage':stage,'frame_id':frame.frame_id,
            'original_ms':float(frame.original_pts*1000),'dispatched_at':frame.ms,
            'scheduled_at':float(scheduled*1000) if scheduled is not None else None,
            'context':context}
        result=None
        if provider=='replay':
            if self.source:
                for kind in kinds:
                    signature=signature_for(kind)
                    key=digest(signature)
                    result=self.source.lookup(key,request['request_id'])
                    if result is not None:
                        break
            if result is None and self.config.replay.policy=='strict':
                raise RuntimeError(f'Replay miss: stage={stage} frame={frame.frame_id} key={key}; capture required frames/crops or enable fill_missing')
            if result is None:
                provider='real'
                # A miss is a newly executed real observation, not synthetic replay.
                signature=signature_for('real')
                key=digest(signature)
                kind='real'
        if result is None:
            if provider=='mock':
                payload=self.mock.infer(request,image)
                backend={'device':'synthetic','runtime':'mock'}
            else:
                if stage_config.target not in self.workers:
                    self.workers[stage_config.target]=WorkerClient(target,self.config.execution.worker_timeout_s)
                worker=self.workers[stage_config.target]
                wire=dict(request)
                actual=dict(manifest)
                actual['path']=resolve_environment(actual['path'])
                if file_hash(actual['path'])!=actual['sha256']:
                    raise ValueError('Model checksum mismatch: '+stage_config.model)
                if target.location=='board':
                    actual['path']='/models/'+Path(actual['path']).name
                wire.update(manifest=actual,target=target.model_dump(exclude={'connection'}))
                reply=worker.call(wire,image)
                payload,backend=reply['payload'],reply['backend']
                self.backends[stage_config.target]=backend
            result={'request_id':request['request_id'],'observation_id':key,'payload':payload,
                'provenance':{'provider':provider,'source_kind':kind,'model_id':stage_config.model,
                    'model_hash':manifest['sha256'],'backend':backend},'schema_version':1}
        result['timing']={k:request[k] for k in ['frame_id','original_ms','dispatched_at','scheduled_at']}
        self.counts[result['provenance']['provider']]+=1
        self.output.store(key,result,stage,signature)
        return result

    def close(self):
        for worker in self.workers.values():
            worker.close()
