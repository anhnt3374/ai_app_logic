# Third-party model and sample sources

OpenCV Zoo revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.

| Asset | Source | License |
|---|---|---|
| YOLOX ONNX, person class | [OpenCV Zoo YOLOX](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox) | [Apache 2.0](licenses/yolox-Apache-2.0.txt) |
| YuNet ONNX | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) | [MIT](licenses/yunet-MIT.txt) |
| SFace ONNX | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface) | [Apache 2.0](licenses/sface-Apache-2.0.txt) |

YOLOX preprocessing/decoding follows the upstream Zoo demo and adapter. SFace
five-point similarity alignment and raw RGB input follow the [OpenCV 4.11
implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/objdetect/src/face_recognize.cpp).
Implementation here adapts these methods to normalized simulator observations.
OpenCV is distributed under Apache 2.0; retained model license texts are in licenses/.

Local acceptance sample `data/input/astronaut.png` was obtained from
[scikit-image v0.19.3](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png).
The scikit-image astronaut sample is a NASA public-domain image. Gallery label
`astronaut_sample` is a local test ID; the simulator does not infer a person's
real-world name. The sample PNG was re-encoded to remove an invalid embedded ICC
profile. Models and demo data are local ignored artifacts, not committed code.
