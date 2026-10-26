from ..config import TargetConfig


def select_backend(target: TargetConfig):
    import cv2
    selected=None
    if target.runtime=='opencv':
        if target.device=='cpu':
            selected={'runtime':'opencv','device':'cpu','backend':cv2.dnn.DNN_BACKEND_OPENCV,'target':cv2.dnn.DNN_TARGET_CPU}
        elif target.device=='gpu' and cv2.cuda.getCudaEnabledDeviceCount()>0 and 'CUDA' in cv2.getBuildInformation():
            available=cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
            if cv2.dnn.DNN_TARGET_CUDA in available:
                if not target.allow_fallback:
                    raise RuntimeError('OpenCV CUDA cannot enforce no CPU-layer fallback; use ONNX Runtime strict or allow_fallback')
                selected={'runtime':'opencv','device':'gpu','backend':cv2.dnn.DNN_BACKEND_CUDA,
                          'target':cv2.dnn.DNN_TARGET_CUDA,'cpu_layers_possible':True,'fallback':True}
    elif target.runtime=='openvino':
        try:
            import openvino as ov
            provider={'cpu':'CPU','gpu':'GPU','npu':'NPU'}[target.device]
            core=ov.Core()
            if provider in core.available_devices:
                selected={'runtime':'openvino','device':target.device,'provider':provider,
                          'options':target.runtime_options}
        except ImportError:
            pass
    else:
        compatible={'cpu':{'CPUExecutionProvider'},
                    'gpu':{'CUDAExecutionProvider','TensorrtExecutionProvider','ROCMExecutionProvider','MIGraphXExecutionProvider'},
                    'npu':{'QNNExecutionProvider','RKNPUExecutionProvider','VitisAIExecutionProvider'}}
        if target.provider and target.provider not in compatible[target.device]:
            raise RuntimeError('Execution provider does not match requested device')
        try:
            import onnxruntime as ort
            provider=target.provider or {'cpu':'CPUExecutionProvider','gpu':'CUDAExecutionProvider','npu':'QNNExecutionProvider'}[target.device]
            if provider in ort.get_available_providers():
                selected={'runtime':'onnxruntime','device':target.device,'provider':provider,'options':target.runtime_options}
        except ImportError:
            pass
    if selected:
        return {**selected,'fallback':selected.get('fallback',False)}
    if target.allow_fallback:
        return {**select_backend(TargetConfig()),'fallback':True,'requested_device':target.device}
    raise RuntimeError(f'{target.runtime}/{target.device} unavailable; install compatible runtime/device or explicitly allow_fallback')
