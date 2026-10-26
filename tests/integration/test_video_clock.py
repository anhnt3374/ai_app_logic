import json
import time
from fractions import Fraction
import av
import numpy as np
from logic_simulator.config import SimulatorConfig
from logic_simulator.engine import run
from logic_simulator.video import read_video
from logic_simulator.inference.gateway import InferenceGateway


def test_vfr_source_and_segment_preserve_original_business_time(tmp_path):
    source=tmp_path/'vfr.mp4'
    with av.open(str(source),'w') as out:
        stream=out.add_stream('libx264',rate=30)
        stream.width,stream.height,stream.pix_fmt=320,240,'yuv420p'
        stream.time_base=Fraction(1,1000)
        stream.codec_context.time_base=Fraction(1,1000)
        for timestamp in [0,40,210,700]:
            frame=av.VideoFrame.from_ndarray(np.full((240,320,3),timestamp%255,np.uint8),format='bgr24')
            frame.pts=timestamp
            frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    frames=list(read_video(source,start_ms=200))
    assert frames[0].original_pts==Fraction(21,100)
    assert frames[0].time==0
    config=SimulatorConfig(input={'path':str(source),'start_ms':200,'business_start_at':'2026-10-25T23:59:59.900+07:00'},
        output={'root':str(tmp_path/'runs')},business={'confirm_observations':1})
    result=run(config)
    events=[json.loads(row) for row in (result.path/'events.jsonl').read_text().splitlines()]
    assert events[0]['business_time'].startswith('2026-10-26T00:00:00.110')
    with av.open(str(result.path/'output.mp4')) as container:
        output=list(container.decode(video=0))
    assert [f.pts*f.time_base for f in output]==[Fraction(0),Fraction(49,100)]


def test_worker_delay_does_not_change_business_events(tmp_path,monkeypatch):
    from logic_simulator.assets import make_demo
    source=make_demo(tmp_path/'input.mp4',frames=12)
    config=SimulatorConfig(input={'path':str(source)},output={'root':str(tmp_path/'runs'),'video':False})
    first=run(config)
    original=InferenceGateway.infer
    def delayed(*args,**kwargs):
        time.sleep(.002)
        return original(*args,**kwargs)
    monkeypatch.setattr(InferenceGateway,'infer',delayed)
    second=run(config)
    assert (first.path/'events.jsonl').read_text()==(second.path/'events.jsonl').read_text()
