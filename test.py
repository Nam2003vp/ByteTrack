import numpy as np
import json
from yolox.tracker.byte_tracker import BYTETracker  # Sửa lỗi import

# Khởi tạo ByteTrack
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)

# Giả lập dữ liệu đầu vào: Danh sách bounding boxes [x, y, w, h, score, class]
dets = np.array([
    [100, 200, 50, 50, 0.9, 1],
    [300, 400, 60, 60, 0.8, 2],
    [500, 600, 70, 70, 0.85, 1]
], dtype=np.float32)

# Chạy tracking
tracked_objects = tracker.update(dets)

# Xử lý kết quả
tracked_results = []
for track in tracked_objects:
    x, y, w, h, track_id, cls = track[:6]
    tracked_results.append({
        "id": int(track_id),
        "box": [float(x), float(y), float(w), float(h)],
        "class": int(cls)
    })

# In kết quả
print(json.dumps(tracked_results, indent=4))

# Lưu ra file
with open("tracked_output.json", "w") as f:
    json.dump(tracked_results, f, indent=4)

print("✅ Kết quả tracking đã được lưu vào 'tracked_output.json'")
