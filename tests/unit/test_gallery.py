import json
from logic_simulator.gallery import Gallery
import pytest


def test_multiple_templates_for_one_person_do_not_create_false_ambiguity(tmp_path):
    path=tmp_path/'gallery.json'
    path.write_text(json.dumps({'embedding_space':'s','people':[
        {'id':'a','embeddings':[[1,0],[.99,.01]]}, {'id':'b','embeddings':[[0,1]]}]}))
    gallery=Gallery(path)
    assert gallery.match([1,0],'s',.45,.05)[0]=='a'
    assert gallery.match([1,1],'s',.45,.05)[0] is None
    with pytest.raises(ValueError,match='embedding-space'):
        gallery.match([1,0],'different',.45,.05)


def test_bad_embeddings_are_rejected():
    with pytest.raises(ValueError,match='Invalid embedding'):
        Gallery().match([0,0,0],'mock-v1',.45,.05)
