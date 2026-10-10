# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyen Tuan Thanh
- MSSV: 2A202602640
- Email:
- Link repo (fork): https://github.com/Chika1357/K4-L2L3-DAY23-NguyenTuanThanh-2A202602640-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit cuối cùng trên `main`, hash nộp kèm trên LMS

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3437 m², 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668 m², 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: xem đoạn dưới.

**Khác biệt hai mode.** Hai mode có cùng `matches` = 502, `ghost_track_frames` = 0,
`missed_gt_frames` = 239 và `mean_confirmed_tracks` = 2.5226. Đối chiếu từng dòng
`grade_run.log` thì `confirmed`, `matches`, `ghosts`, `misses` của `lidar` và `fused`
trùng nhau ở cả 199 frame. Vì vậy hai RMSE được tính trên **cùng số lượng 502 cặp
ghép track–GT**, trong cùng segment, khoảng frame và seed. Log không lưu ID của
từng cặp ghép, nên chưa thể khẳng định danh tính các cặp hoàn toàn giống nhau.
Camera chỉ tinh chỉnh trạng thái EKF; code quy định score, khởi tạo và xoá track
đều do lidar quyết định. Số lượng confirmed track, ghost và miss trong lần chạy
này không thay đổi giữa hai mode.

- `sum_sq_err` giảm từ 11.3437 xuống 9.2668 m² (−18.3%), RMSE giảm từ 0.1503 xuống
  0.1359 m (`rmse_fused − rmse_lidar` = −0.0145 m, đạt điều kiện ≤ 0.05 m).
- Mức cải thiện không đều: trong 195 frame có confirmed track (frame 4–198), fused có
  `sum_sq_err` nhỏ hơn ở 137 frame và lớn hơn ở 58 frame. Ví dụ frame 4 (ngay sau khi
  track vừa được xác nhận) fused 0.0320 so với lidar 0.0113; frame 150 fused 0.0327
  so với lidar 0.0761.
- `ghost_track_frames` = 0 nên `precision_track` = 502 / (502 + 0) = 1.0: RMSE thấp
  không phải do đổi lấy track ma.
- `missed_gt_frames` = 239 chủ yếu do detector: 222 nhãn detector không phát hiện
  (`detection.fn`). Phần còn lại 239 − 222 = 17 = `det_tp` − `matches` = 519 − 502 là
  xe đã được phát hiện nhưng track chưa confirmed, trong đó 8 lượt nằm ở frame 0–3
  (2 xe × 4 frame, lúc `confirmed` = 0). `coverage` = 502 / 519 = 0.967.

**Giới hạn.** Đo camera là tâm hộp 2D ground-truth của FRONT cộng nhiễu Gauss
(`run_lab.py` dùng `rng.normal(0, 0.5, 2)` pixel, trong khi EKF giả định
`sigma_cam` = 5 pixel), không phải đầu ra của camera detector. Mức giảm RMSE ở trên
cho thấy update camera cải thiện sai số tổng hợp trong lần chạy này; nó không
chứng minh chất lượng của một hệ perception camera thật.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   Lidar đo `z = (x, y, z)` là tâm hộp 3D, đơn vị mét. Hàm đo tuyến tính nên `H` là
   ma trận 3×6 cố định (phần xoay `veh_to_sens`, ba cột vận tốc bằng 0) và
   `R = diag(0.1²)` m². Camera đo `z = (u, v)` là pixel. Hàm đo là phép chiếu pinhole
   phi tuyến `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`
   (`camera_fusion.py`, `camera_measurement_prediction`), nên `H` là Jacobian 2×6 tính
   lại tại trạng thái dự báo mỗi lần update, và `R = diag(5²)` pixel². Camera không đo
   trực tiếp độ sâu: một đo camera chỉ ràng buộc hướng nhìn tới xe, còn khoảng cách
   dọc trục nhìn vẫn do lidar và mô hình chuyển động quyết định. Trong `ekf_update`
   công thức giống nhau cho hai cảm biến; khác nhau nằm ở `z`, `h(x)`, `H`, `R` và số
   chiều của `S` (3×3 so với 2×2).

2. Vì sao cần gating Mahalanobis trước khi gán?

   Gán greedy luôn chọn cặp có chi phí nhỏ nhất còn lại; nếu không có cổng, một
   detection sai hoặc của xe khác vẫn bị gán vào track gần nhất, làm lệch trạng thái,
   đồng thời detection đó không sinh track mới và track không bị tính miss. Khoảng
   cách Mahalanobis `d² = γᵀ S⁻¹ γ` với `S = H P Hᵀ + R` chuẩn hoá sai lệch theo độ bất
   định: track mới (P lớn) chấp nhận đo ở xa hơn, track đã hội tụ chỉ nhận đo gần, và
   mét với pixel trở nên so sánh được. Ngưỡng là `chi2.ppf(0.995, dim_meas)`: 12.84
   cho lidar (3 chiều) và 10.60 cho camera (2 chiều) (`association.py`, `chi2_gate`).
   Cặp vượt ngưỡng giữ chi phí `inf` trong `association_cost_matrix`. Cặp ngoài FOV
   bị loại trước đó, không gọi `get_H`/`get_hx`, vì điểm sau camera không chiếu được.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Track-then-fuse. Chỉ có một danh sách track; trong vòng lặp frame của
   `platform/fusion_lab/scripts/run_lab.py` mỗi track được `KF.predict` một lần, sau đó
   `associate_and_update(..., lidar_sensor)` rồi mới tới
   `associate_and_update(..., camera_sensor)`, không predict lại ở giữa. Trên
   `grade_run.log`: ở cả 199 frame, `det_tp/det_fp/det_fn`, `confirmed`, `matches`,
   `ghosts`, `misses` của `mode: lidar` và `mode: fused` trùng nhau; chỉ `sum_sq_err`
   khác, ở 195 frame bắt đầu từ frame 4 (frame đầu tiên có confirmed track). Ví dụ
   frame 150: cả hai mode `matches` = 3, `sum_sq_err` = 0.0761 (lidar) và 0.0327
   (fused). Nếu là fuse-then-track, dữ liệu hai cảm biến được gộp trước khi vào
   tracker. Kiến trúc được xác định từ thứ tự xử lý trong code; các số đếm giống
   nhau trong log là bằng chứng hỗ trợ, không đủ để tự kết luận kiến trúc hay
   danh tính của từng track và cặp ghép GT.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Bài không làm thí nghiệm lệch calibration (không làm bonus), nên phần này là suy
   luận từ mô hình. Khi extrinsic/intrinsic sai, `h(x)` chiếu sai chỗ nên innovation
   camera `γ = z − h(x)` không còn dao động quanh 0 mà có độ lệch hệ thống: cùng dấu,
   lặp lại qua các frame và trên nhiều track. Lệch góc xoay cho độ lệch pixel gần như
   không đổi theo khoảng cách; lệch tịnh tiến cho độ lệch lớn dần khi xe ở gần. `d²`
   của camera tăng vượt mức kỳ vọng (trung bình bằng 2 với 2 bậc tự do). Lệch nhỏ thì
   cặp vẫn qua cổng 10.60, mỗi lần update camera kéo trạng thái lệch sang một bên rồi
   update lidar frame sau kéo lại, nên innovation lidar cũng lệch và RMSE fused lớn
   hơn RMSE lidar. Lệch lớn thì `d²` vượt cổng, camera không còn được gán, kết quả
   fused quay về gần bằng lidar-only.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng không còn `meas.sensor` nào để suy ra lượt này thuộc cảm biến
   nào, nhưng `manage_tracks` vẫn phải chạy và hành xử khác nhau: lượt lidar rỗng phải
   tính miss cho các track trong FOV và xoá track hết điểm; lượt camera rỗng không
   được làm gì (`manager.py`: `if sensor.name != "lidar": return`). Vì vậy
   `associate_and_update` trong `association.py` luôn gọi
   `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` ở cuối, kể cả
   khi bỏ qua bước gán. Lidar quyết định tồn tại vì nó cho vị trí 3D đầy đủ để khởi
   tạo trạng thái (`init_track_state_from_meas`), phủ toàn bộ vùng BEV và đến từ
   detector thật. Camera chỉ đo 2D nên không khởi tạo được độ sâu, chỉ phủ FOV phía
   trước, và trong lab là đo mô phỏng từ nhãn. Nếu camera cũng cộng/trừ score thì mỗi
   frame bị đếm hai lần (cửa sổ `window` = 6 được định nghĩa theo frame lidar) và
   track ngoài FOV camera bị xử lý khác track trong FOV. Bằng chứng trên log: số
   `confirmed` hai mode trùng nhau ở mọi frame.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   - Xác nhận: track mới có `score` = 1/6, trạng thái `initialized`. Mỗi hit lidar
     cộng 1/6 (tối đa 1) và chuyển sang `tentative`; khi `score > 0.8` thì thành
     `confirmed`. Mức 5/6 là mức đầu tiên vượt 0.8 nên cần 5 frame lidar liên tiếp có
     detection (frame khởi tạo cộng 4 hit). Trên log: frame 0–3 `confirmed` = 0, frame 4 `confirmed` = 2.
   - Giữ confirmed sau miss: miss trong FOV lidar chỉ trừ 1/6, `update_track_score`
     không hạ trạng thái. Từ `score` = 1.0, hai miss liên tiếp cho 5/6 rồi 4/6, vẫn
     ≥ 0.6 nên track còn; miss thứ ba cho 0.5 < 0.6 thì bị xoá.
   - Xoá (`should_delete_track`, các điều kiện là OR, chỉ xét ở lượt lidar):
     `P[0,0]` hoặc `P[1,1]` > `max_P` = 9 m² bất kể score; track confirmed có
     `score < 0.6`; track chưa confirmed có `score <= 0` (ví dụ track mới 1/6 bị miss
     ngay frame sau).

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude Opus); Codex
- Dùng cho phần nào (hàm, câu hỏi, debug): Tôi suy nghĩ về bài toán, xác định yêu cầu và trao đổi với Claude Code để tham khảo ý tưởng, cách thực hiện. Tôi nhờ Claude Code viết code theo yêu cầu của mình cho các hàm Part E–H trong `kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`. Claude Code cũng hỗ trợ cài môi trường, chạy `fusion-run-lab` và soạn bản nháp báo cáo, gồm phần giải thích kết quả và 6 câu trả lời, dựa trên `metrics.json` và `grade_run.log` của lần chạy trong repo. Codex rà soát bài theo rubric, chạy lại test và kiểm tra tính nhất quán artifacts; chỉnh lập luận về số lượng cặp ghép track–GT, giới hạn của kết quả camera và bằng chứng về kiến trúc fusion; cập nhật khai báo AI theo thông tin tôi cung cấp; hỗ trợ commit và push bài.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` báo 128 passed, không còn `failed`/`xfailed`; chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198; đối chiếu từng số trong báo cáo với `metrics.json` và tính lại các số theo frame từ `grade_run.log`; chạy `python tools/check_submission.py`.

  Khi rà soát bằng Codex trên Windows, chạy `.venv\Scripts\python.exe -X utf8 -m pytest student/tests -q` đạt 128 passed và `.venv\Scripts\python.exe -X utf8 tools/check_submission.py` báo sẵn sàng nộp ở bản đã commit. Codex kiểm tra lại artifacts hiện có, không chạy lại toàn segment Waymo.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa) — không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
