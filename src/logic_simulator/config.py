import os
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class InputConfig(StrictModel):
    path: str = 'data/input/demo.mp4'
    start_ms: int = Field(0, ge=0)
    end_ms: int | None = Field(None, gt=0)
    business_start_at: datetime = datetime.fromisoformat('2026-10-25T08:00:00+07:00')


class OutputConfig(StrictModel):
    root: str = 'runs'
    video: bool = True
    codec: Literal['libx264'] = 'libx264'
    audio: Literal['drop'] = 'drop'


class StageConfig(StrictModel):
    provider: Literal['mock', 'real', 'replay'] = 'mock'
    model: str = ''
    target: str = 'host_cpu'
    enabled: bool = True
    trigger: Literal['periodic', 'logic'] = 'periodic'
    fps: float | str | None = None
    interval_ms: float | None = None
    offset_ms: float | None = None
    dependency_policy: Literal['same_frame', 'skip_missing'] = 'same_frame'

    def period(self):
        return Fraction(1) / Fraction(str(self.fps)) if self.fps is not None else Fraction(str(self.interval_ms)) / 1000


class TargetConfig(StrictModel):
    location: Literal['host', 'board'] = 'host'
    device: Literal['cpu', 'gpu', 'npu'] = 'cpu'
    runtime: Literal['opencv', 'onnxruntime', 'openvino'] = 'opencv'
    provider: str | None = None
    allow_fallback: bool = False
    connection: dict = Field(default_factory=dict)
    worker_image: str = 'camera-logic-simulator:cpu'
    model_root: str = '/models'
    runtime_options: dict = Field(default_factory=dict)


class EmbeddingConfig(StrictModel):
    strategy: Literal['round_robin'] = 'round_robin'
    max_faces_per_turn: int = Field(1, gt=0)
    stop_when_confirmed: bool = True


class BusinessConfig(StrictModel):
    gallery: str | None = None
    embedding_scheduler: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    similarity_threshold: float = Field(0.45, ge=-1, le=1)
    ambiguity_margin: float = Field(0.05, ge=0)
    confirm_observations: int = Field(3, gt=0)
    confirm_window_ms: int = Field(2000, gt=0)
    track_ttl_ms: int = Field(1000, gt=0)
    track_iou_threshold: float = Field(0.3, ge=0, le=1)
    blacklist_cooldown_ms: int = Field(30000, ge=0)
    min_face_size: int = Field(20, gt=0)
    min_sharpness: float = Field(0, ge=0)


class ReplayConfig(StrictModel):
    policy: Literal['strict', 'fill_missing'] = 'strict'
    source: str | None = None


class ExecutionConfig(StrictModel):
    clock: Literal['video_pts'] = 'video_pts'
    latency: Literal['ideal'] = 'ideal'
    seed: int = 42
    worker_timeout_s: float = Field(120, gt=0)


class SimulatorConfig(StrictModel):
    schema_version: Literal[1] = 1
    logic_id: str = 'face_recognition-2026-10-25-213000'
    logic_root: str = 'logic_versions'
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    targets: dict[str, TargetConfig] = Field(default_factory=dict)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    models: dict = Field(default_factory=lambda: {'registry':'configs/models.yaml'})
    mock: dict = Field(default_factory=lambda: {'scenario':'configs/mock-scenario.yaml'})

    @model_validator(mode='after')
    def schedules(self):
        if not re.fullmatch(r'[a-zA-Z0-9_]+-\d{4}-\d{2}-\d{2}-\d{6}', self.logic_id):
            raise ValueError('logic_id must be taskname-YYYY-MM-DD-HHmmss')
        if self.input.business_start_at.tzinfo is None:
            raise ValueError('business_start_at requires timezone')
        if self.input.end_ms is not None and self.input.end_ms <= self.input.start_ms:
            raise ValueError('end_ms must exceed start_ms')
        if set(self.stages) - {'person','face','embedding'}:
            raise ValueError('unknown stage')
        for role, model in [('person','yolox'),('face','yunet'),('embedding','sface')]:
            stage = self.stages.setdefault(role, StageConfig())
            if not stage.model:
                stage.model = model
            if role == 'embedding':
                if {'fps','interval_ms','offset_ms'} & stage.model_fields_set:
                    raise ValueError('embedding is logic-triggered, no fps/interval/offset')
                stage.trigger = 'logic'
            else:
                if stage.trigger != 'periodic':
                    raise ValueError('detection requires periodic trigger')
                if stage.fps is not None and stage.interval_ms is not None:
                    raise ValueError('choose fps OR interval_ms')
                if stage.fps is None and stage.interval_ms is None:
                    stage.fps = 18.0
                if stage.fps is not None and Fraction(str(stage.fps)) <= 0:
                    raise ValueError('fps must be positive')
                if stage.period() <= 0:
                    raise ValueError('schedule must be positive')
                if stage.offset_ms is not None and stage.offset_ms < 0:
                    raise ValueError('offset must be nonnegative')
        self.targets.setdefault('host_cpu', TargetConfig())
        for stage in self.stages.values():
            if stage.target not in self.targets:
                raise ValueError(f'Unknown target {stage.target}')
        person, face = self.stages['person'], self.stages['face']
        if face.enabled and face.dependency_policy == 'same_frame':
            if not person.enabled or (face.period()/person.period()).denominator != 1:
                raise ValueError('same_frame: face schedule must be subset of person schedule')
            delta = Fraction(str(face.offset_ms or 0))-Fraction(str(person.offset_ms or 0))
            if delta < 0 or (delta/1000/person.period()).denominator != 1:
                raise ValueError('same_frame: offsets must align')
        return self


def load_config(path: Path) -> SimulatorConfig:
    return SimulatorConfig.model_validate(yaml.safe_load(path.read_text()) or {})


def resolve_environment(value):
    if isinstance(value, dict):
        return {k:resolve_environment(v) for k,v in value.items()}
    if isinstance(value, list):
        return [resolve_environment(v) for v in value]
    if isinstance(value, str):
        def resolve(match):
            key=match.group(1)
            if key not in os.environ:
                raise ValueError(f'Missing environment variable: {key}')
            return os.environ[key]
        return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}', resolve, value)
    return value


def sanitize(value):
    if isinstance(value, dict):
        return {k:('[REDACTED]' if any(word in k.lower() for word in ['password','token','secret']) else sanitize(v)) for k,v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value
