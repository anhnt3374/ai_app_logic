from logic_simulator.recognition import EmbeddingPolicy, RecognitionState


def test_round_robin_and_disappearing_faces_do_not_block():
    p = EmbeddingPolicy(1)
    assert [p.select([1,2,3])[0] for _ in range(4)] == [1,2,3,1]
    assert p.select([3]) == [3]
    assert p.select([]) == []


def test_new_tracks_do_not_starve_existing_tracks():
    p = EmbeddingPolicy(1)
    assert p.select([1,2]) == [1]
    assert p.select([1,2,3]) == [2]
    assert p.select([1,2,3,4]) == [1]


def test_identity_confirmation_requires_distinct_observations():
    state = RecognitionState(3, 2000)
    assert not state.observe(1, 'alice', 0, 'a')
    assert not state.observe(1, 'alice', 20, 'a')
    assert not state.observe(1, 'alice', 100, 'b')
    assert state.observe(1, 'alice', 200, 'c')
    assert state.confirmed[1] == 'alice'


def test_expired_or_changed_candidate_resets():
    state = RecognitionState(3, 2000)
    state.observe(1, 'alice', 0, 'a')
    state.observe(1, 'alice', 100, 'b')
    assert not state.observe(1, 'alice', 3000, 'c')
    assert not state.observe(1, 'bob', 3010, 'd')
    assert not state.observe(1, None, 3020, 'e')
    assert not state.observe(1, 'bob', 3030, 'f')
