"""Opt-in real-model acceptance test: SIM_REAL_CPU=1 pytest ... ."""
import json
import os
import av
import pytest
from logic_simulator.assets import make_demo
from logic_simulator.config import load_config
from logic_simulator.engine import run


@pytest.mark.skipif(os.environ.get('SIM_REAL_CPU')!='1',reason='set SIM_REAL_CPU=1 with downloaded models and enrolled demo gallery')
def test_real_person_face_embedding_then_exact_replay(tmp_path):
    video=make_demo(tmp_path/'real.mp4','data/input/astronaut.png',frames=9)
    config=load_config(__import__('pathlib').Path('configs/real-cpu.yaml'))
    config.input.path=str(video)
    config.output.root=str(tmp_path/'runs')
    actual=run(config)
    metrics=json.loads((actual.path/'metrics.json').read_text())
    assert metrics['providers']['real']>0
    assert metrics['providers']['mock']==0
    assert metrics['backends']['host_cpu']['device']=='cpu'
    records=[json.loads(row) for row in (actual.path/'inference.jsonl').read_text().splitlines()]
    assert {r['stage'] for r in records}=={'person','face','embedding'}
    assert len([r for r in records if r['stage']=='embedding'])==3
    events=(actual.path/'events.jsonl').read_text()
    assert 'astronaut_sample' in events
    assert 'attendance' in events
    with av.open(str(actual.path/'output.mp4')) as container:
        assert len(list(container.decode(video=0)))==9
    for stage in config.stages.values():
        stage.provider='replay'
    config.replay.source=str(actual.path/'inference.jsonl')
    replay=run(config)
    assert (replay.path/'events.jsonl').read_text()==events


@pytest.mark.skipif(os.environ.get('SIM_REAL_CPU')!='1',reason='real fill_missing requires model artifacts')
def test_mixed_mock_hit_real_miss_cache_replays_both(tmp_path):
    from pathlib import Path
    from logic_simulator.config import SimulatorConfig
    from logic_simulator.inference.gateway import InferenceGateway
    from logic_simulator.video import read_video
    from logic_simulator.run_store import file_hash
    video=make_demo(tmp_path/'mixed.mp4',frames=3)
    frames=list(read_video(video))
    config=SimulatorConfig()
    directories=[tmp_path/f'cache-{n}' for n in range(3)]
    for directory in directories:
        directory.mkdir()
    original=InferenceGateway(config,directories[0],file_hash(video))
    first=original.infer('person',frames[0].pixels,frames[0])
    original.close()
    config.stages['person'].provider='replay'
    config.replay.source=str(directories[0]/'inference.jsonl')
    config.replay.policy='fill_missing'
    filled=InferenceGateway(config,directories[1],file_hash(video))
    filled.infer('person',frames[0].pixels,frames[0])
    second=filled.infer('person',frames[1].pixels,frames[1])
    filled.close()
    assert second['provenance']['source_kind']=='real'
    config.replay.source=str(directories[1]/'inference.jsonl')
    config.replay.policy='strict'
    replay=InferenceGateway(config,directories[2],file_hash(video))
    try:
        assert replay.infer('person',frames[0].pixels,frames[0])['payload']==first['payload']
        assert replay.infer('person',frames[1].pixels,frames[1])['payload']==second['payload']
    finally:
        replay.close()


@pytest.mark.skipif(os.environ.get('SIM_ORT_CPU')!='1',reason='set SIM_ORT_CPU=1 with optional onnxruntime installed')
def test_onnxruntime_cpu_full_pipeline(tmp_path):
    from pathlib import Path
    config=load_config(Path('configs/real-cpu.yaml'))
    config.input.path=str(make_demo(tmp_path/'ort.mp4','data/input/astronaut.png',frames=9))
    config.output.root=str(tmp_path/'runs')
    config.targets['host_cpu'].runtime='onnxruntime'
    actual=run(config)
    metrics=json.loads((actual.path/'metrics.json').read_text())
    assert metrics['backends']['host_cpu']['provider']=='CPUExecutionProvider'
    assert 'astronaut_sample' in (actual.path/'events.jsonl').read_text()


@pytest.mark.skipif(os.environ.get('SIM_OPENVINO_CPU')!='1',reason='set SIM_OPENVINO_CPU=1 with optional OpenVINO installed')
def test_openvino_cpu_full_pipeline(tmp_path):
    from pathlib import Path
    config=load_config(Path('configs/real-cpu.yaml'))
    config.input.path=str(make_demo(tmp_path/'ov.mp4','data/input/astronaut.png',frames=9))
    config.output.root=str(tmp_path/'runs')
    config.targets['host_cpu'].runtime='openvino'
    actual=run(config)
    metrics=json.loads((actual.path/'metrics.json').read_text())
    assert metrics['backends']['host_cpu']['provider']=='CPU'
    assert 'astronaut_sample' in (actual.path/'events.jsonl').read_text()


@pytest.mark.skipif(os.environ.get('SIM_GPU')!='1',reason='set SIM_GPU=1 in CUDA-enabled image/device environment')
def test_cuda_gpu_full_pipeline(tmp_path):
    from pathlib import Path
    config=load_config(Path('configs/real-gpu.yaml'))
    config.input.path=str(make_demo(tmp_path/'gpu.mp4','data/input/astronaut.png',frames=9))
    config.output.root=str(tmp_path/'runs')
    actual=run(config)
    metrics=json.loads((actual.path/'metrics.json').read_text())
    assert metrics['backends']['host_gpu']['provider']=='CUDAExecutionProvider'
    assert metrics['backends']['host_gpu']['device']=='gpu'
    assert 'astronaut_sample' in (actual.path/'events.jsonl').read_text()

