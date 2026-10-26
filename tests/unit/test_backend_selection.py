import pytest
from logic_simulator.config import TargetConfig
from logic_simulator.inference.backends import select_backend


def test_unavailable_accelerator_requires_explicit_fallback():
    with pytest.raises(RuntimeError,match='unavailable'):
        select_backend(TargetConfig(device='npu',runtime='opencv'))
    report=select_backend(TargetConfig(device='npu',runtime='opencv',allow_fallback=True))
    assert report['device']=='cpu'
    assert report['fallback'] is True


def test_cpu_provider_must_not_be_reported_as_npu():
    with pytest.raises(RuntimeError,match='provider'):
        select_backend(TargetConfig(device='npu',runtime='onnxruntime',provider='CPUExecutionProvider'))


def test_opencv_cuda_cannot_silently_violate_strict_policy(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2.cuda,'getCudaEnabledDeviceCount',lambda:1)
    monkeypatch.setattr(cv2,'getBuildInformation',lambda:'CUDA: YES')
    monkeypatch.setattr(cv2.dnn,'getAvailableTargets',lambda backend:[cv2.dnn.DNN_TARGET_CUDA])
    with pytest.raises(RuntimeError,match='CPU-layer'):
        select_backend(TargetConfig(device='gpu'))
    report=select_backend(TargetConfig(device='gpu',allow_fallback=True))
    assert report['cpu_layers_possible'] is True
    assert report['fallback'] is True


def test_openvino_requires_requested_device(monkeypatch):
    import sys
    import types
    fake=types.SimpleNamespace(Core=lambda:types.SimpleNamespace(available_devices=['CPU']))
    monkeypatch.setitem(sys.modules,'openvino',fake)
    with pytest.raises(RuntimeError,match='unavailable'):
        select_backend(TargetConfig(device='npu',runtime='openvino'))
    assert select_backend(TargetConfig(device='cpu',runtime='openvino'))['provider']=='CPU'
