# AI Camera Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai trước face recognition bằng Python/Docker, person/face detection mặc định 18 FPS, embedding lên lịch theo số mặt trong logic; có mock/real/replay và hồ sơ từng run.

**Architecture:** Video PTS điều khiển scheduler, logic plugin yêu cầu inference qua gateway; worker process độc lập thực thi adapter. Kết quả chuẩn hóa phục vụ nghiệp vụ, cache và output.

**Tech Stack:** Python 3.11, PyAV/FFmpeg, NumPy/OpenCV, Pydantic/YAML, pytest, Docker Compose; runtime GPU/NPU theo thiết bị được xác minh.

**Spec:** [Thiết kế](../specs/2026-10-25-ai-camera-simulator-design.md)

**Status:** Người dùng đã yêu cầu triển khai. Baseline face recognition đã hiện thực;
phạm vi mới chỉ host CPU/GPU/NPU, không làm/thử board đợt này. Tiến độ và bằng
chứng thực tế nằm ở `docs/implementation-progress.md`; checklist dưới đây lưu ý định thiết kế.

## Global Constraints

- Python 3.11; ứng dụng chạy trong Docker; MP4 input/output, thời gian dựa trên PTS.
- FPS là lịch bơm frame; wall-clock inference/network không thay đổi video clock.
- `logic_id` dạng `taskname-YYYY-MM-DD-HHmmss`; README cho từng logic và từng run.
- Provider mock/real/replay và model chọn theo stage. Person/face mặc định 18 FPS,
  đổi được qua YAML; embedding không có fps/interval/offset. Policy chọn số mặt
  nằm trong logic nghiệp vụ, không thuộc model config. Chỉ triển khai face recognition.
- Host/board và CPU/GPU/NPU kiểm tra compatibility; `allow_fallback: false` mặc định.
- Không lưu password vào repository, snapshot config, README, log hoặc command line.
- Không thêm RTSP vào MVP; không kết luận GPU/NPU hoạt động khi chỉ chạy mock/CPU.

## Review Focus

- VFR, FPS phân số, slot trùng và đoạn video: task 2 kiểm tra PTS và không lặp frame.
- Crop/identity đổi khi replay: task 5 kiểm tra cache-miss và embedding-space mismatch.
- Nhiều người, xác nhận lặp, người vừa blacklist vừa đã điểm danh: task 3 kiểm tra dedup.
- Crash giữa run và secrets: task 1/4/7 kiểm tra trạng thái failed và redaction.
- Provider unavailable hoặc âm thầm chạy CPU: task 7 kiểm tra fail/fallback/provenance.

## File structure dự kiến

```text
pyproject.toml / dependency lock / README.md / .gitignore
docker/{Dockerfile.cpu,Dockerfile.gpu,Dockerfile.board}
compose.yaml
configs/{models.yaml,host-cpu.yaml,board.example.yaml}
src/logic_simulator/
  cli.py                         # validate, run, enroll, doctor, compare
  config.py / contracts.py       # schema và các protocol
  run_store.py                   # run ID, snapshots, README, trạng thái
  video.py / scheduler.py        # PTS và lịch detection; không lên lịch FPS embedding
  engine.py / plugins.py         # vòng chạy và load logic plugin
  inference/{gateway,mock,replay,cache,protocol,worker,transport}.py
  inference/adapters/{base,mediapipe_person,yunet,sface,quality}.py
  inference/backends/{opencv,onnxruntime,qnn}.py
  tracking.py / gallery.py / events.py / render.py
logic_versions/face_recognition-2026-10-25-213000/
  README.md / logic.py / default.yaml
tests/{unit,integration,fixtures}/
```

Các adapter/backend chưa kiểm chứng không đăng ký như implementation khả dụng.
Không tạo file stub để giả vờ đã hỗ trợ hết backend. Fixtures video được sinh trong
test; video/gallery dùng nghiên cứu thực mount từ ngoài repository.

### Task 1: Config, contracts và hồ sơ run

**Files:** `config.py`, `contracts.py`, `run_store.py`, `cli.py`, `pyproject.toml`,
`tests/unit/test_config.py`, `tests/unit/test_run_store.py`.

**Interfaces:** `load_config(path: Path) -> SimulatorConfig`;
`create_run(config: SimulatorConfig) -> RunContext`;
`finish_run(run: RunContext, status: str, summary: dict) -> None`.
`FramePacket` chứa frame ID, original/source PTS, relative time, pixels;
`InferenceRequest`/`InferenceResult` tuân theo spec; `Event` chứa video time/evidence.
`StageSpec` chứa role, dependency, provider mapping và trigger: periodic cho detection,
logic cho embedding. Embedding scheduler là policy của plugin, không phải StageSpec FPS.

- [ ] Viết test: person/face mặc định 18 FPS; detection chỉ một fps/interval;
      embedding từ chối fps/interval/offset, nhận trigger logic; giá trị âm/0 và
      max_faces_per_turn <= 0 bị từ chối; board target chưa
      dùng không yêu cầu biến môi trường; hai run cùng giây có thư mục khác nhau;
      snapshot/README không có nội dung password; invalid config không tạo output.
- [ ] Chạy `pytest tests/unit/test_config.py tests/unit/test_run_store.py -q` và xác nhận fail vì chưa có implementation.
- [ ] Implement validation, schema version, run lifecycle, logic/config/source hash,
      README generator và CLI `validate`; tách secret resolution khỏi snapshot.
- [ ] Chạy lại lệnh trên, yêu cầu tất cả pass. Commit task nếu Git writable/khả dụng.

### Task 2: PTS scheduler và core engine

**Files:** `video.py`, `scheduler.py`, `engine.py`, `plugins.py`,
`tests/unit/test_scheduler.py`, `tests/unit/test_engine.py`.

**Consumes:** `FramePacket`, `StageSpec`, `SimulatorConfig`, `RunContext` từ task 1.
**Produces:** `read_video(path: Path, start_ms: int, end_ms: int | None) -> Iterator[FramePacket]`;
`Scheduler.due(frame: FramePacket) -> list[ScheduledStage]`;
`run(config: SimulatorConfig) -> RunSummary`;
`LogicPlugin.build_stages(config) -> list[StageSpec]`, `on_frame(context)`,
`on_result(result, context)`, `on_tick(video_time)`, `finalize()`.

- [ ] Viết test: nguồn 30 FPS/person và face mặc định 18 FPS gọi frame
      `[0,2,4,5,7,9,10,12,14,15,17,19,20,22,24,25,27,29]` trong 1 giây;
      VFR PTS `[0, 40, 210, 700]` ms với interval 200 ms gọi mỗi frame đến hạn
      tối đa một lần, gộp slot ở 700 ms; FPS phân số không trôi; cắt đoạn giữ PTS
      gốc và ngày nghiệp vụ qua nửa đêm; thêm delay worker không đổi timestamp;
      timer vẫn chạy giữa các slot.
- [ ] Chạy `pytest tests/unit/test_scheduler.py tests/unit/test_engine.py -q`, xác nhận fail.
- [ ] Implement scheduler phân số, topological stage order, giao diện plugin,
      skipped_dependency, trace và timeout infrastructure tách khỏi clock logic;
      chỉ scheduler periodic phát lượt detection, plugin phát request embedding.
- [ ] Chạy lại tests, yêu cầu pass; commit task nếu khả dụng.

### Task 3: Mock, face recognition và lịch embedding theo số mặt

**Files:** `inference/{gateway,mock}.py`, `tracking.py`, `events.py`,
`logic_versions/face_recognition-2026-10-25-213000/{logic.py,default.yaml,README.md}`,
`tests/unit/test_attendance.py`, `tests/unit/test_stage_dependencies.py`.

**Consumes:** Plugin/engine và request/result từ task 1–2.
**Produces:** `InferenceGateway.infer(request: InferenceRequest) -> InferenceResult`;
`Tracker.update(detections, video_time) -> list[Track]`;
`MockProvider.infer(request)`; face recognition plugin đúng giao diện task 2;
`EmbeddingPolicy.select(faces: list[FaceObservation], state: RecognitionState) -> list[FaceObservation]`
thuộc logic plugin, chọn tối đa N mặt mỗi lượt detection và cập nhật round-robin.
Mock scenario xuất detection/landmark/embedding observations, không xuất event cuối.

- [ ] Viết test: 2/3 observation chưa xác nhận, 3/3 trong 2000 ms thì xác nhận;
      duplicate observation không tăng vote; cosine dưới 0.45 hoặc margin dưới
      0.05 là unknown/ambiguous; nhiều track cùng person chỉ điểm danh một lần/ngày;
      blacklist vẫn cảnh báo độc lập; cooldown 30000 ms; track hết TTL không giữ
      identity cho người mới; không ghép crop/landmarks khác observation.
      Policy đã chốt N=1: 0 mặt -> 0 call, 1 mặt hợp lệ -> tối đa 1 call mỗi lượt,
      3 track hợp lệ -> lượt A,B,C,A với crop mới nhất; N=2 -> tối đa 2 mặt/lượt;
      crop chất lượng thấp/track mất không chặn track khác; confirmed ra khỏi queue;
      cùng observation không được tính phiếu mới. Không có timer embedding; batch
      size adapter không thay số mặt logic chọn. Kiểm tra đông mặt khiến thiếu
      bằng chứng trong cửa sổ 2 giây phải có lý do trong trace, không tự xác nhận.
- [ ] Chạy `pytest tests/unit/test_attendance.py tests/unit/test_stage_dependencies.py -q`, xác nhận fail.
- [ ] Implement mock deterministic, IoU tracker, state machine, gallery giả lập và
      evidence links và bounded per-track queue. README ghi flow, lịch person/face
      18 FPS, policy N mặt/round-robin, thresholds nghiên cứu, timeline và cách đổi logic.
      Policy mặc định đã chốt với người dùng: 1 mặt mỗi lượt, round-robin.
- [ ] Chạy lại tests, yêu cầu pass; commit task nếu khả dụng.

### Task 4: MP4 output và Docker MVP

**Files:** `render.py`, `docker/Dockerfile.cpu`, `compose.yaml`, `README.md`,
`tests/integration/test_mock_video.py`, `tests/integration/test_run_failure.py`.

**Consumes:** `run(config)`, `Event`, `RunContext` và mock logic từ task 1–3.
**Produces:** `render_frame(frame: FramePacket, state: dict) -> ndarray`;
MP4 H.264 với PTS tương đối theo source, không audio ở MVP; JSONL events/trace,
README/metrics và trạng thái run hoàn chỉnh.

- [ ] Viết integration test sinh MP4 có nội dung chuyển động và mock scenario;
      assert decoded output frame count bằng input đoạn chọn, PTS tăng đúng,
      first/last frame khác nhau và events đúng timestamp mong đợi; lỗi worker
      đánh dấu failed, README không ghi completed; run lần hai không ghi đè.
- [ ] Chạy `pytest tests/integration/test_mock_video.py tests/integration/test_run_failure.py -q`, xác nhận fail.
- [ ] Implement renderer, resource cleanup, output finalization và Compose CPU;
      mount input/model read-only và run output writable; README có lệnh Docker.
- [ ] Build `docker compose build`, chạy `docker compose run --rm simulator validate --config /config/host-cpu.yaml`
      với profile mock hợp lệ, rồi `docker compose run --rm simulator run --config /config/host-cpu.yaml`.
      Decode MP4 output bằng FFmpeg; chạy lại integration tests, yêu cầu pass.
      Commit task nếu khả dụng.

### Task 5: Cache/replay và thử nghiệm thay config

**Files:** `inference/{cache,replay}.py`, `cli.py`,
`tests/unit/test_replay.py`, `tests/integration/test_replay_equivalence.py`.

**Consumes:** Gateway và normalized contracts từ task 1–3.
**Produces:** `Cache.lookup(request: InferenceRequest) -> InferenceResult | None`;
`Cache.store(request, result) -> None`; `ReplayProvider.infer(request)`;
`compare_runs(left: Path, right: Path) -> Comparison` so events/trace chuẩn hóa.

- [ ] Viết test: replay cùng requests cho cùng business events; đổi threshold
      business không invalidate inference; đổi crop/model/preprocess/video hash
      gây miss; tăng FPS và giảm FPS lệch grid đều có thể miss; strict báo rõ,
      fill_missing chỉ gọi request thiếu và không sửa cache gốc; gallery khác
      embedding-space bị từ chối; replay frame cũ không tính vote mới; đổi mock
      scenario/seed/generator gây miss; cache hit rebind request ID hiện tại và
      giữ source observation ID/original request ID, không tái dùng dispatch time cũ;
      đổi N/thứ tự chọn mặt chạy lại policy, chỉ inference request thiếu mới miss,
      không lấy quyết định hàng đợi cũ thay logic hiện tại.
- [ ] Chạy `pytest tests/unit/test_replay.py tests/integration/test_replay_equivalence.py -q`, xác nhận fail.
- [ ] Implement key/index, JSONL schema, source provenance, strict/fill_missing,
      immutable source cache và comparison; README mô tả giới hạn replay khi đổi lịch.
- [ ] Chạy lại tests, yêu cầu pass; commit task nếu khả dụng.

### Task 6: Worker process, model thật CPU và enrollment

**Files:** `inference/{protocol,worker,transport}.py`, `inference/adapters/*.py`,
`inference/backends/opencv.py`, `gallery.py`, `configs/models.yaml`,
`tests/unit/test_model_adapters.py`, `tests/integration/test_real_cpu.py`.

**Consumes:** Gateway/request/result/cache từ task 1–5.
**Produces:** `ModelAdapter.load(manifest: ModelManifest, runtime: RuntimeConfig) -> None`;
`ModelAdapter.infer(images: list[ndarray]) -> list[ModelOutput]`;
`WorkerClient.infer(request: InferenceRequest) -> InferenceResult`;
`enroll(images: Path, model_id: str, target: str, output: Path) -> GalleryManifest`.
Framing: length-prefixed JSON metadata + PNG payload, response correlated theo request ID;
protocol stdout dành riêng dữ liệu, log stderr. Local dùng Unix socket, model load một lần.

- [ ] Xác minh và pin URL/revision/SHA-256/license artifact, phiên bản OpenCV/Python;
      ghi model manifests và download command có kiểm checksum. Đánh giá baseline
      person detector với clip nhiều người; nếu không phù hợp, đề xuất adapter thay thế.
- [ ] Viết test: crop/letterbox map bbox về frame gốc, landmarks alignment, output
      embedding chuẩn hóa đúng dimension, quality tách detector score, model chỉ
      load một lần, request ID sai bị từ chối, worker chậm không đổi video clock.
- [ ] Chạy `pytest tests/unit/test_model_adapters.py -q`, xác nhận fail; implement
      adapters CPU và worker; chạy lại yêu cầu pass.
- [ ] Thêm bộ ảnh/video enrollment có quyền sử dụng, chạy `pytest tests/integration/test_real_cpu.py -q`
      trong Docker với model mounts. Yêu cầu có person/face/embedding observations
      thật, gallery fingerprint đúng, output MP4 decode được. Không yêu cầu số bbox
      giống tuyệt đối giữa runtime; không biến test thiếu model thành pass giả.
- [ ] Chạy real -> replay trên cùng requests; events phải tương đương; cập nhật
      README logic với artifact/model versions và giới hạn; commit nếu khả dụng.

### Task 7: Backend CPU/GPU/NPU trên máy hiện tại

**Files:** `inference/backends.py`, `inference/adapters.py`, `docker/Dockerfile.gpu`,
`compose.gpu-wsl.yaml`, `configs/{real-gpu,real-npu}.yaml`,
`tests/unit/test_backend_selection.py`, `tests/integration/test_real_cpu.py`.

**Consumes:** Worker protocol và adapters từ task 6.
**Produces:** CPU OpenCV/ORT/OpenVINO, GPU CUDA EP và Intel NPU OpenVINO có preflight
rõ ràng; backend chưa được OS expose phải lỗi khi không cho phép fallback.

- [x] Kiểm kê host: Intel Core Ultra 7 255H, NVIDIA RTX PRO 500 Blackwell, WSL;
      CUDA 13.2/cuDNN 9 có sẵn. Kiểm tra ngoài sandbox để phân biệt device bị sandbox che.
- [x] Pin ONNX Runtime GPU 1.30.0 cho CUDA 13; image CPU Python 3.11 làm base.
      Mount WSL driver store, /dev/dxg, CUDA và cuDNN read-only trong override host.
- [x] Chạy YOLOX/YuNet/SFace CUDA thật với CPU graph fallback bị tắt; xác nhận
      person/face/embedding observations, attendance, MP4 và replay.
- [x] Kiểm thử OpenVINO CPU cùng interface; kiểm tra Intel NPU từ Docker OpenVINO.
      Hiện NPU unavailable và /dev/accel không có; không báo đã chạy NPU.
- [x] Kiểm thử strict/fallback/provider provenance; OpenCV CUDA không được giả
      là thuần GPU khi thư viện có thể fallback từng layer sang CPU.
- [ ] Chạy NPU thật khi môi trường Linux/Docker được OS expose NPU. Đây là giới
      hạn môi trường hiện tại, không chặn bàn giao CPU/GPU.

Board/dev kit không thuộc đợt này theo chỉ dẫn mới nhất. Helper remote đã viết
trước đó giữ trạng thái thử nghiệm, không kiểm thử hoặc báo hoàn tất remote.

## Handoff và nghiệm thu

Thứ tự đề xuất: 1–4 tạo MVP mock nhìn thấy được; 5–6 cho nghiên cứu bằng model thật
và replay; 7 kiểm chứng CPU/GPU/NPU host theo khả năng truy cập. Khói/lửa không thuộc đợt
triển khai này. Không ước lượng ngày công trước khi xác minh
phần cứng/runtime và dữ liệu video/gallery.

Mỗi milestone lưu run mẫu với README và lệnh tái hiện; tests đã pass chỉ chứng minh
phạm vi được chạy. Báo cáo cuối liệt kê rõ backend đã verified và backend chưa có
thiết bị. Người dùng đã yêu cầu triển khai; xem README và nhật ký để biết trạng thái thực tế.
