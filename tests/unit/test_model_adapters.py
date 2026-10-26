import numpy as np
from logic_simulator.inference.adapters import align_face, decode_yolox


def test_yolox_maps_letterboxed_person_to_source():
    outputs=np.zeros((1,8400,85),dtype=np.float32)
    # anchor (0,0), stride 8 -> center (40,48), dimensions (32,40)
    outputs[0,0,:4]=[5,6,np.log(4),np.log(5)]
    outputs[0,0,4]=1
    outputs[0,0,5]=.9
    result=decode_yolox(outputs,scale=.5,width=1280,height=720,threshold=.5)
    assert len(result)==1
    assert np.allclose(result[0]['bbox'],[48,56,112,136],atol=.001)


def test_alignment_uses_entire_observation_landmarks():
    image=np.zeros((112,112,3),dtype=np.uint8)
    image[50:60,50:60]=255
    template=[[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]]
    aligned=align_face(image,template)
    assert aligned.shape==(112,112,3)
    assert np.abs(aligned.astype(float)-image).max()<=1
