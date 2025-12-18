import cv2
import os
import sys

# =========================
# Đọc file annotation
# =========================
def read_annotation(txt_path):
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Bỏ qua dòng rỗng hoặc comment
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Dòng hợp lệ phải có ít nhất 5 cột
            if len(parts) < 5:
                continue

            # Kiểm tra start/end có phải số không
            if not parts[1].isdigit() or not parts[2].isdigit():
                continue

            start_ms = int(parts[1])
            end_ms   = int(parts[2])
            label    = parts[4]

            segments.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "label": label
            })

    return segments


# =========================
# Cắt video theo từng từ
# =========================
def cut_words(video_name):
    video_path = os.path.join("DATA", "Raw data", video_name)
    txt_path   = os.path.join(
        "DATA",
        "Gold data",
        os.path.splitext(video_name)[0] + ".txt"
    )

    output_root = os.path.join("DATA", "Segmented")

    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    if not os.path.exists(txt_path):
        print(f"❌ Không tìm thấy annotation: {txt_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không mở được video")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segments = read_annotation(txt_path)
    os.makedirs(output_root, exist_ok=True)

    word_count = {}

    for seg in segments:
        label = seg["label"]
        start_frame = int(seg["start_ms"] / 1000 * fps)
        end_frame   = int(seg["end_ms"]   / 1000 * fps)

        # Folder theo từng nhãn
        word_dir = os.path.join(output_root, label)
        os.makedirs(word_dir, exist_ok=True)

        word_count[label] = word_count.get(label, 0) + 1
        out_name = f"{label}_{word_count[label]:03d}.mp4"
        out_path = os.path.join(word_dir, out_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"✅ Lưu: {out_path}")

    cap.release()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Cách dùng:")
        print("   python cut_words.py <ten_video.mp4>")
        sys.exit(1)

    video_name = sys.argv[1]
    cut_words(video_name)