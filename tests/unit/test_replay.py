import pytest
from logic_simulator.inference.cache import Cache


def test_replay_rebinds_request_id_and_preserves_observation(tmp_path):
    source=tmp_path/'cache.jsonl'
    cache=Cache(source)
    cache.store('key', {'request_id':'old', 'observation_id':'stable', 'payload':{'embedding':[1,0]}, 'provenance':{'provider':'mock'}})
    loaded=Cache(source)
    result=loaded.lookup('key', 'new')
    assert result['request_id']=='new'
    assert result['observation_id']=='stable'
    assert result['provenance']['original_request_id']=='old'
    assert loaded.lookup('different','new') is None


def test_invalid_cache_records_are_not_silently_ignored(tmp_path):
    path=tmp_path/'cache.jsonl'
    path.write_text('{broken\n')
    with pytest.raises(ValueError):
        Cache(path)


def test_cache_retains_mixed_source_kinds_per_stage(tmp_path):
    path=tmp_path/'cache.jsonl'
    cache=Cache(path)
    for kind in ['mock','real']:
        cache.store(kind,{'request_id':kind,'observation_id':kind,'payload':{},
                    'provenance':{'provider':kind,'source_kind':kind}},stage='person')
    assert Cache(path).kinds['person']=={'mock','real'}
