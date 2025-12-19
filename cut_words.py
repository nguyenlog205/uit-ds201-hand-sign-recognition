import cv2
import os
import sys
import io

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
        print(f"[ERROR] Khong tim thay video: {video_path}")
        return False

    if not os.path.exists(txt_path):
        print(f"[ERROR] Khong tim thay annotation: {txt_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Khong mo duoc video: {video_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segments = read_annotation(txt_path)
    if not segments:
        print(f"[WARNING] Khong co segment nao trong file: {txt_path}")
        cap.release()
        return False

    os.makedirs(output_root, exist_ok=True)

    word_count = {}
    saved_count = 0

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
        saved_count += 1
        print(f"  -> Luu: {out_path}")

    cap.release()
    print(f"[OK] Da luu {saved_count} segment tu video: {video_name}")
    return True


# =========================
# Xử lý tất cả các file
# =========================
def process_all():
    gold_data_dir = os.path.join("DATA", "Gold data")
    raw_data_dir = os.path.join("DATA", "Raw data")
    
    if not os.path.exists(gold_data_dir):
        print(f"[ERROR] Khong tim thay thu muc: {gold_data_dir}")
        return
    
    if not os.path.exists(raw_data_dir):
        print(f"[ERROR] Khong tim thay thu muc: {raw_data_dir}")
        return
    
    # Lấy tất cả file .txt trong Gold data
    txt_files = [f for f in os.listdir(gold_data_dir) if f.endswith(".txt")]
    
    if not txt_files:
        print("[ERROR] Khong tim thay file .txt nao trong Gold data")
        return
    
    print(f"[INFO] Tim thay {len(txt_files)} file annotation")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for txt_file in txt_files:
        # Tìm file video tương ứng (bỏ .txt, thêm .mp4)
        video_name = os.path.splitext(txt_file)[0] + ".mp4"
        video_path = os.path.join(raw_data_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"[SKIP] Bo qua: Khong tim thay video tuong ung: {video_name}")
            fail_count += 1
            continue
        
        print(f"\n[DANG XU LY] {video_name}")
        print("-" * 60)
        
        try:
            if cut_words(video_name):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"[ERROR] Loi khi xu ly {video_name}: {str(e)}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"[TONG KET]")
    print(f"   Thanh cong: {success_count}")
    print(f"   That bai: {fail_count}")
    print(f"   Tong so: {len(txt_files)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Xử lý một video cụ thể
        video_name = sys.argv[1]
        cut_words(video_name)
    else:
        # Xử lý tất cả các video
        process_all()