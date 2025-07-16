# infer_yolov8_realtime.py

# 导入必要的库
import os
import time
import datetime
import numpy as np
import cv2
from api_infer import *
from utils import detect_postprocess_yolov8, draw_detect_res, preprocess_img

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# --- 模型与SNPE配置 ---
dlc_path = "/home/fibo/Cats_detection_project/cats_detection/cats_detection.dlc"
MODEL_INPUT_NAME = "images"
MODEL_OUTPUT_NAME = "output0"
MODEL_INPUT_SHAPE = (640, 640)
NUM_CLASSES = 1
CLASS_NAMES = ['cat']

# --- 后处理参数 ---
CONF_THRESHOLD = 0.6
IOU_THRESHOLD = 0.45

# --- 新增：实时捕捉与自动拍摄配置 ---
# USB摄像头ID，通常为 0
CAMERA_ID = 2 
# 结果保存目录
SAVE_DIR = "/home/fibo/Cats_detection_project/cats_detection/captures"
# 持续捕获到目标的帧数阈值（避免误识别触发拍摄）
SUSTAINED_DETECTION_FRAMES_THRESHOLD = 20 
# 完成拍摄后的冷却时间（秒）
CAPTURE_COOLDOWN_SECONDS = 80
# 视频拍摄时长（秒）
VIDEO_DURATION_SECONDS = 20


# ==============================================================================
# --- 程序初始化 ---
# ==============================================================================

# 确保保存目录存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created save directory: {SAVE_DIR}")

# --- SNPE 初始化 ---
print("Initializing SNPE context...")
try:
    snpe_ort = SnpeContext(dlc_path, [], Runtime.GPU, PerfProfile.BALANCED, LogLevel.INFO)
    assert snpe_ort.Initialize() == 0
    print("SNPE context initialized successfully.")
except Exception as e:
    print(f"Error initializing SNPE: {e}")
    exit()

# --- 摄像头初始化 ---
print(f"Opening USB camera with ID: {CAMERA_ID}...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Error: Could not open camera with ID {CAMERA_ID}.")
    exit()

# 获取摄像头的分辨率和FPS用于视频保存
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 # 如果获取不到FPS，则默认为30
print(f"Camera opened successfully. Resolution: {frame_width}x{frame_height}, FPS: {fps}")


# ==============================================================================
# --- 主循环与状态管理 ---
# ==============================================================================

# 状态变量
detection_counter = 0           # 连续检测到目标的帧数计数器
last_capture_time = 0           # 上次拍摄（照片或视频）完成的时间戳
is_recording = False            # 是否正在录像的标志
video_writer = None             # VideoWriter对象
video_start_time = 0            # 当前视频开始录制的时间戳

print("\nStarting real-time detection loop... Press 'q' to quit.")

try:
    while True:
        # 1. 捕获摄像头画面
        ret, frame_bgr = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera. Retrying...")
            continue
        
        original_shape = frame_bgr.shape

        # 2. 图像预处理
        preprocessed_img = preprocess_img(frame_bgr, target_shape=MODEL_INPUT_SHAPE, div_num=255, means=None, stds=None)

        # 3. 模型推理
        input_feed = {MODEL_INPUT_NAME: preprocessed_img}
        outputs = snpe_ort.Execute([], input_feed)

        # 4. 输出塑形和后处理
        output_data = np.array(outputs[MODEL_OUTPUT_NAME])
        reshaped_output = output_data.reshape(1, 4 + NUM_CLASSES, 8400)
        final_detections = detect_postprocess_yolov8(
            reshaped_output,
            original_shape,
            (MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 3),
            num_classes=NUM_CLASSES,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD
        )

        # 5. 核心逻辑：根据检测结果更新状态并执行动作
        is_target_detected = len(final_detections[0]) > 0

        if is_target_detected:
            detection_counter += 1
        else:
            detection_counter = 0 # 如果当前帧没有目标，重置计数器

        # 检查是否满足拍摄条件
        is_cooldown_over = (time.time() - last_capture_time) > CAPTURE_COOLDOWN_SECONDS
        is_sustained_detection = detection_counter >= SUSTAINED_DETECTION_FRAMES_THRESHOLD

        if is_sustained_detection and is_cooldown_over and not is_recording:
            # --- 触发拍照和录像 ---
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # (1) 拍摄图片
            photo_path = os.path.join(SAVE_DIR, f"photo_{timestamp_str}.jpg")
            cv2.imwrite(photo_path, frame_bgr)
            print(f"✅ Sustained target detected! Photo saved to: {photo_path}")

            # (2) 开始10秒视频拍摄
            video_path = os.path.join(SAVE_DIR, f"video_{timestamp_str}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID' for .avi
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            is_recording = True
            video_start_time = time.time()
            print(f"🎥 Starting {VIDEO_DURATION_SECONDS}-second video recording to: {video_path}")

            # 更新冷却计时器
            last_capture_time = time.time() 
            # 重置检测计数器，防止在本次录像期间再次触发
            detection_counter = 0


        # 6. 处理正在录制中的视频
        if is_recording:
            # 将当前帧（无论是否带框）写入视频文件
            video_writer.write(frame_bgr) 
            
            # 检查录像是否达到指定时长
            if time.time() - video_start_time >= VIDEO_DURATION_SECONDS:
                video_writer.release()
                video_writer = None
                is_recording = False
                print(f"🎬 Video recording finished. Entering {CAPTURE_COOLDOWN_SECONDS}-second cooldown.")
                # 注意：last_capture_time在开始录制时已更新，此处无需再更新

        # 7. 结果可视化
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result_image = draw_detect_res(frame_rgb, final_detections, class_names=CLASS_NAMES)
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        # 在画面上显示状态信息
        status_text = ""
        if is_recording:
            status_text = f"RECORDING... {int(time.time() - video_start_time)}s"
        elif not is_cooldown_over:
            remaining_cooldown = int(CAPTURE_COOLDOWN_SECONDS - (time.time() - last_capture_time))
            status_text = f"COOLDOWN: {remaining_cooldown}s left"
        else:
             status_text = f"Detecting... ({detection_counter}/{SUSTAINED_DETECTION_FRAMES_THRESHOLD})"
        
        cv2.putText(result_image_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time YOLOv8 Detection', result_image_bgr)

        # 检测退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 8. 释放资源
    print("\nCleaning up and shutting down...")
    if is_recording and video_writer is not None:
        video_writer.release() # 确保程序退出时，正在录制的视频能被保存
        print("Saved pending video before exit.")
    cap.release()
    cv2.destroyAllWindows()
    # snpe_ort.Terminate() # 如果api_infer库有提供终止方法，请调用
    print("Cleanup complete. Exiting.")