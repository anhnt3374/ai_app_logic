import json
import av
import numpy as np
from logic_simulator.config import SimulatorConfig
from logic_simulator.engine import run


def make_video(path, count=30):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=30)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = 'yuv420p'
        for i in range(count):
            image = np.zeros((240,320,3), dtype=np.uint8)
            image[:, :, 1] = i * 3
            image[30:150, 20+i:100+i] = (50,180,230)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='bgr24')):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_mock_video_and_replay(tmp_path):
    video = tmp_path / 'input.mp4'
    make_video(video)
    config = SimulatorConfig(input={'path':str(video)}, output={'root':str(tmp_path/'runs')})
    first = run(config)
    assert first.status == 'completed'
    events = rows(first.path / 'events.jsonl')
    face_record=next(r for r in rows(first.path/'inference.jsonl') if r['stage']=='face')
    assert 'parent_box' in face_record['request']['context']
    assert face_record['request']['frame_shape']==[240,320,3]
    assert {e['type'] for e in events} == {'attendance', 'blacklist'}
    assert len([e for e in events if e['type']=='attendance']) == 2
    trace = rows(first.path / 'trace.jsonl')
    turns = [e for e in trace if e['type']=='embedding_selection']
    assert [e['selected'] for e in turns[:4]] == [[1],[2],[1],[2]]
    assert all(len(e['selected']) <= 1 for e in turns)
    with av.open(str(first.path/'output.mp4')) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 30
    assert not np.array_equal(frames[0].to_ndarray(),frames[-1].to_ndarray())
    assert all(a.pts*a.time_base < b.pts*b.time_base for a,b in zip(frames,frames[1:]))
    replay = config.model_copy(deep=True)
    for stage in replay.stages.values():
        stage.provider = 'replay'
    replay.replay.source = str(first.path/'inference.jsonl')
    second = run(replay)
    assert rows(second.path/'events.jsonl') == events
    assert first.path != second.path


def test_replay_missing_marks_failed(tmp_path):
    video=tmp_path/'input.mp4'
    make_video(video, 5)
    source=tmp_path/'empty.jsonl'
    source.write_text('')
    config=SimulatorConfig(input={'path':str(video)}, output={'root':str(tmp_path/'runs')},
        stages={'person':{'provider':'replay'}}, replay={'source':str(source)})
    import pytest
    with pytest.raises(Exception, match='Replay miss'):
        run(config)
    manifests=list((tmp_path/'runs').glob('*/*/manifest.json'))
    assert len(manifests)==1
    assert json.loads(manifests[0].read_text())['status']=='failed'
