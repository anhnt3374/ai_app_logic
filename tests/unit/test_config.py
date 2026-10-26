import pytest
from logic_simulator.config import SimulatorConfig, load_config


def test_detection_defaults_and_embedding_has_no_clock():
    c = SimulatorConfig()
    assert c.stages['person'].fps == c.stages['face'].fps == 18
    assert c.stages['embedding'].trigger == 'logic'
    assert c.business.embedding_scheduler.max_faces_per_turn == 1


@pytest.mark.parametrize('stage,changes', [
    ('person', {'fps': 0}), ('face', {'fps': -2}),
    ('face', {'fps': 18, 'interval_ms': 100}),
    ('embedding', {'fps': 2}), ('embedding', {'interval_ms': 10}),
    ('embedding', {'offset_ms': 0}),
])
def test_invalid_schedule(stage, changes):
    with pytest.raises(ValueError):
        SimulatorConfig(stages={stage: changes})


def test_incompatible_same_frame_schedule():
    with pytest.raises(ValueError, match='same_frame'):
        SimulatorConfig(stages={'person': {'fps': 5}, 'face': {'fps': 18}})


def test_unused_board_environment_is_not_resolved(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text('targets:\n  unused:\n    location: board\n    connection:\n      host: ${MISSING_SIM_VAR}\n')
    assert load_config(p).stages['person'].target == 'host_cpu'


def test_zero_face_budget_rejected():
    with pytest.raises(ValueError):
        SimulatorConfig(business={'embedding_scheduler': {'max_faces_per_turn': 0}})
