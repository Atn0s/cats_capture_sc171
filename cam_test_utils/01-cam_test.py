'''脚本用于远程连接无法外接显示器的情况下的Orbbec相机测试'''
import cv2

# 设置摄像头索引
# 外接的相机从2开始
camera_index = 2 
cap = cv2.VideoCapture(camera_index)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开索引为 {camera_index} 的摄像头。")
    # 如果失败，可以尝试其他索引
    # cap = cv2.VideoCapture(0) 
    # if not cap.isOpened():
    #    print("错误：也无法打开索引为 0 的摄像头。")
    exit()

print("摄像头已成功打开。正在尝试捕获一帧...")

# 读取一帧
ret, frame = cap.read()

if ret:
    # 将捕获到的帧保存为图片文件
    cv2.imwrite("capture_test1.jpg", frame)
    print("成功！已捕获一帧并保存为 capture_test1.jpg")
else:
    print("错误：无法从摄像头捕获帧。")

# 释放摄像头资源
cap.release()
print("摄像头已释放。")