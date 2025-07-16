# infer_yolov8.py

# 导入必要的库
import os # [!!] 导入os库用于处理文件路径
from api_infer import *
import numpy as np
import cv2
from utils import detect_postprocess_yolov8, draw_detect_res, preprocess_img
import time

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

dlc_path = "/home/fibo/Cats_detection_project/cats_detection/cats_detection.dlc"
img_path = "/home/fibo/Cats_detection_project/cats_detection/002.jpg"

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


# ==============================================================================
# --- SNPE 初始化 ---
# ==============================================================================
print("Initializing SNPE context...")
try:
    snpe_ort = SnpeContext(dlc_path, [], Runtime.GPU, PerfProfile.BALANCED, LogLevel.INFO)
    assert snpe_ort.Initialize() == 0
    print("SNPE context initialized successfully.")
except Exception as e:
    print(f"Error initializing SNPE: {e}")
    exit()

# ==============================================================================
# --- 图像预处理 ---
# ==============================================================================
print(f"Reading and preprocessing image: {img_path}")
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    print(f"Error: Could not read image at {img_path}")
    exit()

original_shape = img_bgr.shape
preprocessed_img = preprocess_img(img_bgr, target_shape=MODEL_INPUT_SHAPE, div_num=255, means=None, stds=None)
print(f"Image preprocessed. Shape: {preprocessed_img.shape}")


# ==============================================================================
# --- 模型推理 ---
# ==============================================================================
print("Executing inference...")
input_feed = {MODEL_INPUT_NAME: preprocessed_img}
output_names = []

start_time = time.time()
outputs = snpe_ort.Execute(output_names, input_feed)
end_time = time.time()
print(f"Inference execution finished in {end_time - start_time:.4f} seconds.")


# ==============================================================================
# --- 输出塑形和后处理 ---
# ==============================================================================
print("Performing post-processing...")
if MODEL_OUTPUT_NAME not in outputs:
    print(f"Error: Expected output tensor '{MODEL_OUTPUT_NAME}' not found in model outputs.")
    print("Available output tensors:", list(outputs.keys()))
    exit()

output_data = np.array(outputs[MODEL_OUTPUT_NAME])
print(f"Model raw output shape: {output_data.shape}")

channels = 4 + NUM_CLASSES
num_predictions = 8400
try:
    reshaped_output = output_data.reshape(1, channels, num_predictions)
    print(f"Reshaped output to: {reshaped_output.shape}")
except ValueError:
    print(f"Error: Cannot reshape array of size {output_data.size} into shape (1, {channels}, {num_predictions}).")
    exit()

final_detections = detect_postprocess_yolov8(
    reshaped_output,
    original_shape,
    (MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 3),
    num_classes=NUM_CLASSES,
    conf_thres=CONF_THRESHOLD,
    iou_thres=IOU_THRESHOLD
)
print("Post-processing complete.")
print(final_detections[0])


# ==============================================================================
# --- 结果可视化与保存 (全景图) ---
# ==============================================================================
print("Drawing detection results on the original image...")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
result_image = draw_detect_res(img_rgb, final_detections, class_names=CLASS_NAMES)
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

try:
    cv2.imwrite(save_path_full_image, result_image_bgr)
    print(f"Result image saved successfully to: {save_path_full_image}")
except Exception as e:
    print(f"Error saving image: {e}")


# ==============================================================================
# --- [!!] 新增：裁剪并保存每个检测到的目标 ---
# ==============================================================================
print(f"Cropping and saving detected objects to '{CROP_SAVE_DIR}'...")

# 确保裁剪保存目录存在，如果不存在则创建
os.makedirs(CROP_SAVE_DIR, exist_ok=True)

# 初始化一个计数器，用于为每个裁剪的物体生成唯一的文件名
crop_counter = 0

# 遍历每个类别的检测结果
for class_id, class_detections in enumerate(final_detections):
    if len(class_detections) == 0:
        continue

    class_name = CLASS_NAMES[class_id]

    # 遍历该类别下的每一个检测框
    for box in class_detections:
        # box的格式是 [x, y, w, h, score]
        x, y, w, h = [int(t) for t in box[:4]]
        
        # [!!] 关键步骤：从原始BGR图像中裁剪
        # 我们使用原始图像 img_bgr 进行裁剪，以获得最高质量的图像
        # 注意：要确保裁剪坐标不会超出图像边界
        crop_x1 = max(0, x)
        crop_y1 = max(0, y)
        crop_x2 = min(img_bgr.shape[1], x + w)
        crop_y2 = min(img_bgr.shape[0], y + h)

        # 执行裁剪
        cropped_image = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 如果裁剪出的图像有效（宽度和高度都大于0），则保存
        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            # 构建保存路径和文件名
            # 文件名示例: cropped_cat_0.jpg, cropped_cat_1.jpg, ...
            crop_filename = f"cropped_{class_name}_{crop_counter}.jpg"
            crop_filepath = os.path.join(CROP_SAVE_DIR, crop_filename)
            
            # 保存裁剪的图片
            cv2.imwrite(crop_filepath, cropped_image)
            
            # 计数器加一
            crop_counter += 1

print(f"Successfully saved {crop_counter} cropped objects.")


# ==============================================================================
# --- 可选：弹窗显示 ---
# ==============================================================================
# try:
#     cv2.imshow("YOLOv8 Detection Result", result_image_bgr)
#     print("Displaying result. Press any key to exit.")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# except cv2.error as e:
#     print(f"Could not display the image (likely running in a headless environment): {e}")
