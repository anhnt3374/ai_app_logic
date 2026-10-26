# Camera AI Logic Simulator — Python / Docker

Simulator MP4 cho **face recognition, điểm danh và blacklist**. Person/face detection
mặc định **18 FPS**. Embedding **không có FPS**: mỗi lượt face detection, logic
chọn tối đa **1 mặt hợp lệ**, luân phiên giữa các track chưa xác nhận.

Thời gian nghiệp vụ chạy theo PTS video. Máy xử lý chậm chỉ khiến chạy simulator
lâu hơn; không làm rớt frame, thay timestamp sự kiện hoặc giảm FPS giả lập.
Đây là công cụ nghiên cứu flow, không phải benchmark tốc độ phần cứng.

## Chạy thử bằng Docker

Chạy từ thư mục project:

```sh
mkdir -p data models runs
docker compose build
docker compose run --rm simulator demo
docker compose run --rm simulator validate --config configs/host-cpu.yaml
docker compose run --rm simulator run --config configs/host-cpu.yaml
```

Nếu máy thiếu buildx: `DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.cpu -t camera-logic-simulator:cpu .`.
Compose mặc định chạy user 1000:1000. Nếu user host khác, đặt LOCAL_UID/LOCAL_GID
cho đúng để ghi được data/models/runs. Config và logic được mount read-only;
data/models cần ghi cho demo/fetch/enroll. Run processing chỉ đọc input và model.

Mock mặc định có hai người: Alice được điểm danh; Bob vừa điểm danh vừa cảnh báo
blacklist. Mock xuất bbox/landmarks/embedding; sự kiện vẫn đi qua logic thật.
Không cần tải model để chạy mock hoặc replay mock.

## Model thật

Baseline: **YOLOX person (COCO class 0) → YuNet → SFace**. Artifact pin revision và
SHA-256 trong [model registry](configs/models.yaml); license trong [THIRD_PARTY.md](THIRD_PARTY.md).

```sh
docker compose run --rm simulator fetch-models
```

Chuẩn bị enrollment:

```text
data/enrollment/
  person_a/
    face1.jpg
    face2.jpg
    person.json
  person_b/
    face.jpg
```

Ảnh enrollment phải có đúng một mặt. `person.json` tùy chọn:
`{"name":"Tên hiển thị","blacklist":true}`. ID là tên thư mục, không phải tên
model tự suy đoán. Gallery lưu embedding đã chuẩn hóa và fingerprint model.

```sh
docker compose run --rm simulator enroll --images data/enrollment --output data/gallery/real.json
docker compose run --rm simulator run --config configs/real-cpu.yaml --input data/input/your-video.mp4
```

Config real-cpu mặc định đọc `data/input/real-demo.mp4`. Có thể tạo clip từ một ảnh
để smoke-test: `docker compose run --rm simulator demo --image data/input/photo.png --output data/input/real-demo.mp4 --frames 30`.
Demo ảnh chỉ chứng minh pipeline hoạt động; dùng video thực tế để nghiên cứu tracking.

## Chỉnh model, lịch detection và logic

```yaml
stages:
  person: {provider: real, model: yolox, target: host_cpu, fps: 18}
  face: {provider: real, model: yunet, target: host_cpu, fps: 18}
  embedding: {provider: real, model: sface, target: host_cpu, trigger: logic}
business:
  embedding_scheduler:
    strategy: round_robin
    max_faces_per_turn: 1
    stop_when_confirmed: true
```

Model registry chứa weights/adapter/preprocessing/options; policy embedding thuộc
business logic. Không đặt fps/interval/offset cho embedding. Có thể đổi số mặt
mỗi lượt trong policy khi nghiên cứu; mặc định đã chốt là 1. Mặt không đạt quality
gate không chiếm lượt; không có mặt thì không gọi embedding. Runtime batch size
không quyết định số mặt mà logic chọn.

Detection nhận `fps` hoặc `interval_ms`, thêm `offset_ms` tùy chọn. Face dùng
person observation trên cùng frame, nên lịch face phải là tập con lịch person;
chọn `dependency_policy: skip_missing` nếu muốn khảo sát lịch lệch. Nguồn ít FPS
hơn lịch sẽ gộp slot, không nhân bản frame. Start/end dùng millisecond của video gốc.

Đổi weights cùng kiến trúc: thêm manifest rồi sửa `model`. Kiến trúc mới: thêm
adapter chuẩn hóa bbox/landmarks/embedding ở `inference/adapters.py`. Đổi embedding
model phải enroll lại; gallery sai embedding space bị từ chối.

Đổi thuật toán: tạo thư mục `logic_versions/taskname-YYYY-MM-DD-HHmmss` có
`logic.py`, `default.yaml`, README. Plugin xuất `create(config,gateway,run_path)`;
object có `tick(frame)`, `render(frame)`, `finalize()`. Sửa `logic_id` trong config.
Tham khảo [README logic baseline](logic_versions/face_recognition-2026-10-25-213000/README.md).
Chỉ đổi config thì không cần nhân bản code logic.

## Replay và so sánh

```sh
docker compose run --rm simulator run --config configs/host-cpu.yaml --replay-from runs/face_recognition-2026-10-25-213000/RUN_ID
docker compose run --rm simulator compare runs/face_recognition-2026-10-25-213000/RUN_A runs/face_recognition-2026-10-25-213000/RUN_B
```

Dùng config real-cpu khi replay bản ghi real, giữ cùng video/model/scenario. Replay
chạy lại logic và policy chọn mặt, không đọc thẳng events cũ. Key gồm hash video,
frame/PTS, pixels/crop, model/adapter/runtime và mock scenario/seed khi dùng mock.
Ngưỡng nghiệp vụ không nằm trong key. Cache hit giữ observation ID và gắn request
ID mới, không biến một observation cũ thành nhiều bằng chứng xác nhận.

Mặc định strict: thiếu frame/crop chính xác thì lỗi `Replay miss`. Đổi detection
FPS, số mặt, tracker hoặc crop có thể cần inference mới. `--replay-policy fill_missing`
gọi real cho request thiếu và ghi cache mới, không sửa cache nguồn; chỉ dùng khi
model/gallery/provider của các stage tương thích. Cache thưa không tạo thêm dữ liệu.

## Output mỗi run

`runs/<taskname-YYYY-MM-DD-HHmmss>/<timestamp+timezone-unique-id>/`:

- `README.md`, `config.resolved.yaml`, `manifest.json`: trạng thái, config và fingerprints.
- `logic/`, `source/`, `requirements.txt`: snapshot code và dependencies để truy lại logic.
- `output.mp4`: H.264, giữ PTS tương đối và frame count; bỏ audio ở bản đầu.
- `events.jsonl`, `evidence/`: điểm danh/blacklist và frame bằng chứng.
- `inference.jsonl`: cache kết quả chuẩn hóa, provider/model/backend thực dùng.
- `trace.jsonl`: lượt detection, chọn/hoãn mặt, quality, candidate và confirmed.
- `metrics.json`: số frame/lượt gọi, slot gộp, nguồn mock/real/replay, backend.

Mỗi run có ID riêng, không ghi đè. Run lỗi có trạng thái failed; Ctrl+C là interrupted.
Snapshot loại secrets. Ngày nghiệp vụ = `business_start_at` (có timezone, ứng với
đầu video gốc) + PTS gốc; không dùng đồng hồ máy để xác nhận/cooldown.

## CPU / GPU / NPU trên host và board

Core chọn target riêng cho từng stage. Inference thật chạy ở worker process riêng;
model load một lần. Local dùng stdin/stdout framed JSON + PNG; board dùng cùng
protocol qua SSH đến worker Docker, không cần chia sẻ filesystem hay RTSP.

```sh
docker compose run --rm simulator doctor --config configs/real-cpu.yaml --target host_cpu
```

CPU OpenCV và ONNX Runtime/OpenVINO CPU đã kiểm chứng. GPU OpenCV cần bản OpenCV build CUDA, hoặc
dùng ONNX Runtime với CUDAExecutionProvider. NPU dùng ONNX Runtime provider đúng
thiết bị (ví dụ QNNExecutionProvider) và artifact/quantization/runtime SDK phù hợp.
Image CPU không cài driver hoặc provider tăng tốc. Với runtime image riêng, dùng
config target `runtime: onnxruntime`, `device: gpu|npu`, `provider` và
`runtime_options`; phải cấp device/driver cho container. Backend không có sẽ lỗi;
`allow_fallback: true` mới cho phép CPU fallback và ghi rõ CPU thực dùng. ORT tắt
CPU graph fallback cho target tăng tốc; model không tương thích phải lỗi.
Doctor chỉ xác minh provider có thể khởi tạo, không chứng minh model đã chạy NPU.
OpenCV CUDA có thể tự chạy layer không hỗ trợ trên CPU: chế độ strict sẽ từ chối
backend này. Khi cho phép fallback, provenance ghi rõ khả năng chạy mixed GPU/CPU.
Đường ONNX Runtime CPU cũng có test tùy chọn (`SIM_ORT_CPU=1`, cài onnxruntime==1.23.2).

### GPU trên máy WSL hiện tại

Đã chạy thật YOLOX/YuNet/SFace trên NVIDIA RTX PRO 500 Blackwell trong Docker,
CUDA 13.2, ONNX Runtime GPU 1.30.0, `allow_fallback: false`. Override này dùng
driver/CUDA đã có trên host, không cài hoặc thay driver host.

```sh
# Wheel được cache ở project này. Chuẩn bị lại khi dùng máy/project mới:
python3 -m pip download --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.11 --implementation cp --abi cp311 --dest .wheelhouse/gpu onnxruntime-gpu==1.30.0
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.gpu -t camera-logic-simulator:gpu .
docker compose -f compose.yaml -f compose.gpu-wsl.yaml run --rm simulator run --config configs/real-gpu.yaml
```

`compose.gpu-wsl.yaml` map /dev/dxg, WSL driver store, CUDA và cuDNN của máy hiện
tại. Máy khác phải chỉnh mount/version phù hợp, hoặc dùng NVIDIA Container Toolkit
và CUDA runtime image tương ứng. [Bảng tương thích ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Intel NPU dùng config `configs/real-npu.yaml`, runtime OpenVINO 2026.0.0, **không
CPU fallback**. Trên WSL hiện tại không có /dev/accel và OpenVINO trong Docker báo
NPU unavailable; chưa thể chạy inference NPU ở môi trường này. Có config không
đồng nghĩa NPU đã được nghiệm thu. Dùng doctor để kiểm tra lại khi OS expose NPU.

### Board/dev kit — để giai đoạn sau

Theo phạm vi cập nhật, không cần triển khai hoặc thử board trong đợt này. Helper
SSH/provision đã có dưới dạng thử nghiệm, chưa nghiệm thu; hướng dẫn dưới đây chỉ
để tham khảo khi mở lại phạm vi remote.

Board: copy [board.example.yaml](configs/board.example.yaml), điền endpoint của
session/LAN, port và username qua biến môi trường. Password nhập ở prompt với
`docker compose run --rm -it ...`; hoặc file secret mount và `password_file`.
Không đặt password trên command line hoặc commit YAML chứa password.

1. Chuẩn bị Docker image đúng kiến trúc trên board (build native từ source hoặc
   load image đã build), tên trùng `worker_image` trong config.
2. Với QDC: kích hoạt session, mở Connect SSH tunnel và xác minh mapping hiện tại.
   Nếu Connect chỉ bind host 127.0.0.1, host.docker.internal không truy cập được
   listener đó. Trên Linux dùng thêm `-f compose.qdc.yaml` với
   `BOARD_SSH_HOST=127.0.0.1` để container dùng host network; ví dụ
   `docker compose -f compose.yaml -f compose.qdc.yaml run --rm -it simulator doctor --config configs/board.yaml --target board`.
   Nếu listener bind địa chỉ host-gateway đã được cho phép thì có thể dùng
   host.docker.internal. Windows/macOS cần route Connect tương ứng có thể truy cập
   từ Docker; phải xác minh route trước khi chạy. Board LAN dùng IP trực tiếp.
3. Chạy doctor rồi provision model (SFTP + SHA-256), sau đó run:

```sh
docker compose run --rm -it simulator doctor --config configs/board.yaml --target board
docker compose run --rm -it simulator provision --config configs/board.yaml --target board
docker compose run --rm -it simulator run --config configs/board.yaml
```

SSH dùng known_hosts mặc định; có thể bật accept_new_host_key cho session QDC có
host key tạm thời. NPU board cần custom image có SDK/provider và config
`connection.devices` để map device; NVIDIA cần thêm `connection.gpus: all`.
Không có session/thiết bị thì chưa thể nghiệm thu remote hoặc tăng tốc.

## Kiểm thử

```sh
docker compose run --rm --entrypoint python simulator -m pytest -q
docker compose run --rm -e SIM_REAL_CPU=1 --entrypoint python simulator -m pytest -q
# Toàn bộ backend host đã kiểm chứng, bao gồm GPU:
docker compose -f compose.yaml -f compose.gpu-wsl.yaml run --rm -e SIM_REAL_CPU=1 -e SIM_ORT_CPU=1 -e SIM_OPENVINO_CPU=1 -e SIM_GPU=1 --entrypoint python simulator -m pytest -q
```

Real acceptance test cần model đã tải, `data/input/astronaut.png` và gallery demo
`astronaut_sample` đã enroll; thiếu thì test real báo lỗi, không giả thành pass.
Suite mặc định ghi skip rõ cho real test chưa được bật. Test gồm 18 FPS/PTS phân
số, VFR/đoạn qua nửa đêm, round-robin, TTL, xác nhận/dedup, gallery, replay, lỗi run,
worker protocol và backend fallback. Bộ demo không thay thế kiểm thử đông người.

Local dev (Python 3.10+; container dùng 3.11): tạo venv, cài requirements.txt và
`pip install -e .`; dùng `logic-simulator` tương tự entrypoint Docker.
