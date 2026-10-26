import json
from logic_simulator.config import SimulatorConfig
from logic_simulator.run_store import create_run, finish_run


def test_unique_run_and_sanitized_snapshot(tmp_path):
    c = SimulatorConfig(output={'root': str(tmp_path)}, targets={
        'private': {'location': 'board', 'connection': {'password': 'secret-value'}}})
    a, b = create_run(c), create_run(c)
    assert a.path != b.path
    finish_run(a, 'failed', {'error': 'worker unavailable'})
    assert json.loads((a.path / 'manifest.json').read_text())['status'] == 'failed'
    for file in ['README.md', 'config.resolved.yaml', 'manifest.json']:
        assert 'secret-value' not in (a.path / file).read_text()


def test_run_snapshots_executable_framework_and_dependencies(tmp_path):
    run=create_run(SimulatorConfig(output={'root':str(tmp_path)}))
    assert (run.path/'source/logic_simulator/face_pipeline.py').is_file()
    assert (run.path/'requirements.txt').is_file()
    assert 'source_hash' in json.loads((run.path/'manifest.json').read_text())
