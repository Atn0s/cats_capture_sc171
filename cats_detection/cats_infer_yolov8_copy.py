# infer_yolov8.py

# 导入必要的库
import os # [!!] 导入os库用于处理文件路径
from api_infer import *
import numpy as np
import cv2
from utils import detect_postprocess_yolov8, draw_detect_res, preprocess_img
import time
import threading

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

dlc_path = "/home/fibo/Cats_detection_project/cats_detection/cats_detection.dlc"
# [!!] 修改：使用摄像头而不是静态图片
video_device = "/dev/video2"

# 结果保存路径
save_path_full_image = "/home/fibo/Cats_detection_project/cats_detection/result.jpg"

# [!!] 新增：裁剪图片的保存目录
CROP_SAVE_DIR = "/home/fibo/Cats_detection_project/cats_detection/cropped_objects"

# 模型相关的关键参数
MODEL_INPUT_NAME = "images"
MODEL_OUTPUT_NAME = "output0"
MODEL_INPUT_SHAPE = (640, 640)
NUM_CLASSES = 1
CLASS_NAMES = ['cat']

# 后处理参数
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# [!!] 新增：实时处理相关参数
DISPLAY_WINDOW = os.environ.get('DISPLAY') is not None  # 自动检测并决定是否显示窗口
SAVE_CROPPED_OBJECTS = True  # 是否保存裁剪的物体
FPS_LIMIT = 15  # 帧率限制


# ==============================================================================
# --- 实时摄像头处理函数 ---
# ==============================================================================
def process_frame(frame, frame_counter):
    """
    处理单帧图像，进行目标检测和结果绘制
    """
    original_shape = frame.shape
    
    # 预处理图像
    preprocessed_img = preprocess_img(frame, target_shape=MODEL_INPUT_SHAPE, div_num=255, means=None, stds=None)
    
    # 模型推理
    input_feed = {MODEL_INPUT_NAME: preprocessed_img}
    output_names = []
    
    start_time = time.time()
    outputs = snpe_ort.Execute(output_names, input_feed)
    end_time = time.time()
    
    # 后处理
    if MODEL_OUTPUT_NAME not in outputs:
        print(f"Error: Expected output tensor '{MODEL_OUTPUT_NAME}' not found in model outputs.")
        return frame, []
    
    output_data = np.array(outputs[MODEL_OUTPUT_NAME])
    
    channels = 4 + NUM_CLASSES
    num_predictions = 8400
    try:
        reshaped_output = output_data.reshape(1, channels, num_predictions)
    except ValueError:
        print(f"Error: Cannot reshape array of size {output_data.size} into shape (1, {channels}, {num_predictions}).")
        return frame, []
    
    final_detections = detect_postprocess_yolov8(
        reshaped_output,
        original_shape,
        (MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 3),
        num_classes=NUM_CLASSES,
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD
    )
    
    # 绘制检测结果
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_frame = draw_detect_res(frame_rgb, final_detections, class_names=CLASS_NAMES)
    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
    
    # 在图像上显示FPS和检测信息
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(result_frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 统计检测到的猫数量
    total_cats = sum(len(detections) for detections in final_detections)
    cv2.putText(result_frame_bgr, f"Cats detected: {total_cats}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存裁剪的物体（如果启用）
    if SAVE_CROPPED_OBJECTS and total_cats > 0:
        # 使用线程异步保存，避免阻塞主循环
        save_thread = threading.Thread(target=save_cropped_objects, args=(frame.copy(), final_detections, frame_counter))
        save_thread.start()
    
    return result_frame_bgr, final_detections

def save_cropped_objects(frame, detections, frame_counter):
    """
    保存检测到的目标物体的裁剪图像
    """
    os.makedirs(CROP_SAVE_DIR, exist_ok=True)
    
    crop_counter = 0
    for class_id, class_detections in enumerate(detections):
        if len(class_detections) == 0:
            continue
        
        class_name = CLASS_NAMES[class_id]
        
        for box in class_detections:
            x, y, w, h = [int(t) for t in box[:4]]
            
            # 确保裁剪坐标不会超出图像边界
            crop_x1 = max(0, x)
            crop_y1 = max(0, y)
            crop_x2 = min(frame.shape[1], x + w)
            crop_y2 = min(frame.shape[0], y + h)
            
            # 执行裁剪
            cropped_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                # 使用帧号和计数器构建唯一文件名
                crop_filename = f"frame_{frame_counter:06d}_{class_name}_{crop_counter}.jpg"
                crop_filepath = os.path.join(CROP_SAVE_DIR, crop_filename)
                
                cv2.imwrite(crop_filepath, cropped_image)
                crop_counter += 1

# 全局SNPE上下文变量
snpe_ort = None

# ==============================================================================
# --- 主程序：实时摄像头处理 ---
# ==============================================================================
def main():
    """
    主函数，用于初始化和运行实时摄像头检测循环
    """
    global snpe_ort
    
    # 初始化SNPE
    print("Initializing SNPE context...")
    try:
        snpe_ort = SnpeContext(dlc_path, [], Runtime.GPU, PerfProfile.HIGH_PERFORMANCE, LogLevel.INFO)
        assert snpe_ort.Initialize() == 0
        print("SNPE context initialized successfully.")
    except Exception as e:
        print(f"Error initializing SNPE: {e}")
        return

    # 打开摄像头
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        print(f"Error: Could not open video device {video_device}")
        return

    # 设置帧率
    if FPS_LIMIT > 0:
        cap.set(cv2.CAP_PROP_FPS, FPS_LIMIT)

    frame_counter = 0
    
    print("Starting real-time cat detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. End of stream?")
            break

        # 处理单帧图像
        processed_frame, detections = process_frame(frame, frame_counter)
        
        # 在终端打印检测到的猫的位置
        if any(len(d) > 0 for d in detections):
            print(f"--- Frame {frame_counter}: Cats Detected ---")
            for class_id, class_detections in enumerate(detections):
                if len(class_detections) == 0:
                    continue
                
                class_name = CLASS_NAMES[class_id]
                for i, box in enumerate(class_detections):
                    x, y, w, h, score = box
                    print(f"  - Cat #{i+1}: BBox(x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}) Score={score:.2f}")

        # 如果启用了显示窗口，则显示结果
        if DISPLAY_WINDOW:
            cv2.imshow("YOLOv8 Real-time Cat Detection", processed_frame)

        # 检查退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_counter += 1

    # 循环结束后释放资源
    cap.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()
    print("Video stream stopped and resources released.")

if __name__ == "__main__":
    main()

