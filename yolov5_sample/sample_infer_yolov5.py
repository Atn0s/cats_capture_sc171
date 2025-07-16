# 导入必要的库
from api_infer import *  # 导入SNPE推理相关库
import numpy as np 
import cv2 
from utils import detect_postprocess, draw_detect_res, preprocess_img  # 导入自定义工具函数
from PIL import Image 

#配置区 
dlc_path = "/home/fibo/Cats_detection_project/yolov5_sample/yolov5.dlc"  # 模型文件路径（.dlc格式）
img_path = "/home/fibo/Cats_detection_project/yolov5_sample/bus.jpg"      # 测试图片输入路径
save_path = "/home/fibo/Cats_detection_project/yolov5_sample/result_bus.jpg"  # 结果保存路径

#SNPE初始化部分
snpe_ort = SnpeContext(dlc_path, [], Runtime.GPU, PerfProfile.BALANCED, LogLevel.INFO)
assert snpe_ort.Initialize() == 0 

#  图像预处理部分 
pic = cv2.imread(img_path)
pic = cv2.resize(pic, (640, 640))
# 调用预处理函数（参数说明）：
# target_shape: 目标尺寸640x640
# div_num=255: 像素值归一化到0-1范围
pic = preprocess_img(pic, target_shape=(640, 640), div_num=255, means=None, stds=None)

#  模型推理部分
input_feed = {"serving_default_input_1:0": pic}
# 定义输出节点列表（空列表表示获取所有输出）
output_names = []
# 执行推理操作（返回输出字典）
outputs = snpe_ort.Execute(output_names, input_feed)

#  后处理部分 
date = outputs['StatefulPartitionedCall:0']
date = np.array(date)
# 打印实际输出 shape
print("模型输出 shape:", date.shape)
print("模型输出 size:", date.size)
# 重塑输出形状为(1, 25200, 6)：
# 1: batch_size
# 25200: 预测框数量（YOLO的3个尺度输出总和）
# 6: 每个预测框的参数（x_center, y_center, width, height, obj_conf, class_conf）
pred = date.reshape(1, 25200, 6)

#  结果可视化部分 
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 执行后处理（参数说明）：
# pred: 模型原始输出
# [1080,810,3]: 原始图像形状（height, width, channels）
# [640,640,3]: 模型输入图像形状
# conf_thres=0.5: 置信度阈值
# iou_thres=0.45: NMS的IOU阈值
pred = detect_postprocess(pred, [1080, 810, 3], [640, 640, 3], conf_thres=0.5, iou_thres=0.45)
# 在图像上绘制检测结果
res_img = draw_detect_res(img, pred)

#  结果保存与展示 
# 转换为PIL格式进行保存
res_img = Image.fromarray(res_img)
res_img.save(save_path)         
# 读取保存结果并弹窗显示
# blended2 = Image.open(save_path)
# blended2.show()  