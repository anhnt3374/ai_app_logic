import json
import pytest
from logic_simulator.assets import make_demo
from logic_simulator.config import SimulatorConfig
from logic_simulator.engine import run
from logic_simulator.inference.gateway import InferenceGateway


def test_cleanup_failure_cannot_report_completed(tmp_path,monkeypatch):
    source=make_demo(tmp_path/'input.mp4',frames=3)
    config=SimulatorConfig(input={'path':str(source)},output={'root':str(tmp_path/'runs'),'video':False})
    def broken_close(self):
        raise RuntimeError('worker cleanup failed')
    monkeypatch.setattr(InferenceGateway,'close',broken_close)
    with pytest.raises(RuntimeError,match='cleanup'):
        run(config)
    manifest=next((tmp_path/'runs').glob('*/*/manifest.json'))
    assert json.loads(manifest.read_text())['status']=='failed'


def test_requested_plugin_snapshot_is_saved_in_new_run(tmp_path):
    source=make_demo(tmp_path/'input.mp4',frames=3)
    snapshot=tmp_path/'old-logic'
    snapshot.mkdir()
    code='from logic_simulator.face_pipeline import FacePipeline\nMARKER="archived-plugin"\ndef create(c,g,p): return FacePipeline(c,g,p)\n'
    (snapshot/'logic.py').write_text(code)
    (snapshot/'README.md').write_text('Archived test plugin')
    config=SimulatorConfig(input={'path':str(source)},output={'root':str(tmp_path/'runs'),'video':False})
    result=run(config,logic_snapshot=str(snapshot))
    assert (result.path/'logic/logic.py').read_text()==code
