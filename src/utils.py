import cv2
import mediapipe as mp
import numpy as np
import os

# KHỞI TẠO MODEL (chỉ một lần)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Cấu hình vẫn giữ nguyên để tìm tối đa 2 tay
hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)


def hand_extraction(
    image_path: str,
    is_extract_arm: bool = False,
    save_path: str | None = None,
    is_return_picture: bool = False
):
    """
    Trích xuất tọa độ, phân biệt tay TRÁI/PHẢI và lưu chung vào MỘT file duy nhất.
    Đảm bảo tất cả bàn tay được phát hiện đều được vẽ lên ảnh nếu `is_return_picture` là True.

    Args:
        image_path (str): Đường dẫn đến file ảnh đầu vào.
        is_extract_arm (bool): (Vẫn giữ lại) Nếu True, khung chữ nhật vẽ ra sẽ bao gồm cả cánh tay.
        save_path (str | None): Đường dẫn để lưu MỘT file .npy duy nhất chứa dictionary dữ liệu.
        is_return_picture (bool): Nếu True, trả về cả dữ liệu tọa độ và ảnh đã vẽ.

    Returns:
        - dict[str, np.ndarray|None]: Nếu `is_return_picture` là False.
                                      VD: {'left': array, 'right': array}.
        - tuple[dict, np.ndarray]: Nếu `is_return_picture` là True.
                                   Trả về (dictionary tọa độ, ảnh đã vẽ).
    """
    # 1. Đọc và xử lý ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ '{image_path}'")
        return ({'left': None, 'right': None}, None) if is_return_picture else {'left': None, 'right': None}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)
    
    # Khởi tạo ảnh để vẽ nếu cần, đảm bảo đây là bản sao mới nhất
    # và sẽ được cập nhật trong vòng lặp
    annotated_image = image.copy() if is_return_picture else None 
    hand_data = {'left': None, 'right': None}
    
    # 2. LẶP QUA TỪNG BÀN TAY ĐỂ THU THẬP DỮ LIỆU
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx]
            hand_label = handedness.classification[0].label.lower()

            landmark_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
            )
            
            hand_data[hand_label] = landmark_coords
            
            # --- ĐIỂM SỬA CHỮA CHÍNH: Đảm bảo vẽ lên ảnh HIỆN TẠI ---
            if is_return_picture and annotated_image is not None:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
    else:
        print(f"Cảnh báo: Không tìm thấy bàn tay trong ảnh '{os.path.basename(image_path)}'")
        # Nếu không tìm thấy tay nào, và is_return_picture là True, vẫn trả về ảnh gốc
        return (hand_data, image) if is_return_picture else hand_data

    # 3. LƯU DỮ LIỆU SAU KHI ĐÃ THU THẬP ĐỦ
    if save_path:
        try:
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            np.save(save_path, hand_data)
            print(f"✅ Đã lưu dữ liệu 2 tay vào chung file: {save_path}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu: {e}")

    # 4. Trả về kết quả
    if is_return_picture:
        return hand_data, annotated_image
    else:
        return hand_data

def example(
    image_path: str,
    is_extract_arm: bool,
    save_path: str
):
    """Hàm ví dụ để gọi và kiểm tra chức năng hand_extraction."""
    print(f"Đang xử lý ảnh: {image_path}")
    coordinates = hand_extraction(
        image_path=image_path,
        is_extract_arm=is_extract_arm,
        save_path=save_path
    )
    
    if coordinates is not None:
        print(f"Trích xuất thành công. Shape của dữ liệu: {coordinates.shape}")
        print("-" * 30)
    else:
        print("Trích xuất thất bại.")
        print("-" * 30)