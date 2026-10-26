# Face recognition — 2026-10-25 21:30:00 (UTC+07)

Baseline nghiên cứu: MP4 -> person -> tracking -> face -> quality -> chọn mặt ->
embedding -> gallery matching -> xác nhận -> điểm danh/blacklist -> MP4 + events.

## Lịch và trạng thái

- Person và face mặc định 18 FPS, cùng offset, theo PTS video. Thời gian model
  xử lý không làm thay đổi clock. Video ít FPS hơn thì không tạo frame giả.
- Mỗi lượt face detection, chọn **tối đa 1 mặt hợp lệ** theo round-robin giữa
  các track chưa confirmed. A/B/C -> A,B,C,A...; không có mặt thì không gọi model.
- Crop mới thay crop cũ theo track. Mặt vắng ở lượt hiện tại không được chọn và
  không cản các track khác. Track mới được thêm cuối queue; hết TTL thì xóa.
- Quality gate: kích thước tối thiểu 20 pixel, độ nét Laplacian >= ngưỡng config.
  Nếu một person crop có nhiều mặt thì baseline chọn mặt lớn nhất.
- Matching cosine >= 0.45, chênh lệch top-1/top-2 >= 0.05. Đây là ngưỡng
  nghiên cứu, cần hiệu chỉnh. Unknown/ambiguous không được điểm danh.
- Cần 3 observation khác nhau trong 2000 ms; đổi identity/unknown reset candidate.
  Sau confirmed ngừng embedding cho track đó. Ví dụ A/B ở các lượt 0..5:
  A,B,A,B,A-confirmed,B-confirmed. Không dùng embedding FPS.
- Track TTL 1000 ms. Track lost không truyền danh tính sang track mới.
- Điểm danh một lần/person/ngày nghiệp vụ trong run; blacklist độc lập, cooldown
  30000 ms. Ngày nghiệp vụ tính từ business_start_at + PTS gốc của video.
- IoU tracker có thể đổi ID khi người che khuất/cắt nhau. Không có liveness,
  chống giả mạo hay cam kết độ chính xác nhận diện sản phẩm.

## Config và thay logic

Model/provider/target ở `stages`. Person/face có `fps` hoặc `interval_ms`.
Embedding chỉ có `trigger: logic`. `business.embedding_scheduler` là policy
nghiệp vụ; model registry chỉ mô tả artifact và adapter. Model khác kiến trúc
cần adapter mới; embedding model khác cần enroll lại gallery.

`create(config, gateway, run_path)` trả plugin có `tick(frame)`, `render(frame)` và
`finalize()`. Baseline dùng `FacePipeline`; copy thư mục thành
`taskname-YYYY-MM-DD-HHmmss` và thay plugin để nghiên cứu flow khác. README mới
phải mô tả thay đổi. Chỉ đổi tham số thì dùng config mới, không cần copy logic.

Mock xuất observations theo scenario, không tạo thẳng event. Replay chạy lại
toàn bộ logic chọn mặt; thiếu frame/crop đúng key thì lỗi strict hoặc chạy bổ
sung real khi bật fill_missing. Real models: YOLOX (COCO person), YuNet, SFace.
Đây là phiên bản đầu; không có phiên bản trước để so sánh.

## Chạy Docker

Từ project root:

```sh
docker compose run --rm simulator demo
docker compose run --rm simulator run --config configs/host-cpu.yaml
docker compose run --rm simulator fetch-models
docker compose run --rm simulator enroll --images data/enrollment --output data/gallery/real.json
docker compose run --rm simulator run --config configs/real-cpu.yaml
```

Enrollment: mỗi thư mục con là person ID, chứa ảnh mặt. Dùng `person.json` với
`{"name":"Tên hiển thị","blacklist":true}` để đánh dấu blacklist. Thư mục run có
README/config/source snapshot/MP4/events/cache/trace/metrics/evidence riêng.
