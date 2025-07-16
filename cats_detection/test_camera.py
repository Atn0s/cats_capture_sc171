# camera_test_interactive_fps.py

import cv2
import subprocess
import time

# ==============================================================================
# --- 配置区 ---
# ==============================================================================
# [!!] 修改为您要测试的摄像头设备路径
CAMERA_DEVICE_PATH = "/dev/video2"

# [!!] 设置曝光滑动条的范围和初始值
# 重要：请先通过命令 `v4l2-ctl -d /dev/video0 -l` 查看您相机的曝光范围 (exposure_absolute)
# 并将下面的 MIN 和 MAX 设置为合理的范围值。
EXPOSURE_MIN = 10
EXPOSURE_MAX = 1000
INITIAL_EXPOSURE = 150

# --- 全局变量 ---
# 用于避免在每一帧都重复发送相同的设置命令
last_set_exposure = -1

# ==============================================================================
# --- 辅助函数：设置相机曝光 ---
# ==============================================================================

def set_manual_exposure(device_path, value):
    """使用v4l2-ctl设置曝光为手动模式并设定一个值。"""
    global last_set_exposure
    if value == last_set_exposure:
        return

    # print(f"正在设置曝光值: {value}...") # 在交互模式下，此打印会刷屏，故注释掉
    try:
        cmd_mode = ['v4l2-ctl', '-d', device_path, '-c', 'exposure_auto=1']
        subprocess.run(cmd_mode, check=True, capture_output=True, timeout=0.5)
        cmd_value = ['v4l2-ctl', '-d', device_path, '-c', f'exposure_absolute={value}']
        subprocess.run(cmd_value, check=True, capture_output=True, timeout=0.5)
        last_set_exposure = value
    except FileNotFoundError:
        print("错误: 'v4l2-ctl' 未找到。请运行 'sudo apt install v4l-utils' 来安装它。")
        last_set_exposure = value
    except subprocess.CalledProcessError as e:
        print(f"错误: 设置曝光失败。命令: '{' '.join(e.cmd)}'。")
        last_set_exposure = value
    except subprocess.TimeoutExpired:
        print("错误: v4l2-ctl 命令执行超时。")
        last_set_exposure = value

# ==============================================================================
# --- 主测试流程 ---
# ==============================================================================

def run_interactive_test():
    """启动一个带GUI的交互式相机测试。"""

    print("========= 开始交互式相机测试 ==========")
    print(f"将要打开摄像头: {CAMERA_DEVICE_PATH}")
    print("在弹出的窗口中拖动滑动条来调节曝光。")
    print("按 'q' 键退出。")

    # --- 1. 初始化摄像头 ---
    camera_id = int(CAMERA_DEVICE_PATH.replace("/dev/video", ""))
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 失败: 无法打开摄像头 {CAMERA_DEVICE_PATH}。")
        return
    print("✅ 成功: 摄像头已打开。")

    # --- 2. 创建窗口和滑动条 ---
    window_name = "Camera Test (Press 'q' to quit)"
    cv2.namedWindow(window_name)
    def dummy_callback(x): pass
    cv2.createTrackbar('Exposure', window_name, INITIAL_EXPOSURE, EXPOSURE_MAX, dummy_callback)
    cv2.setTrackbarMin('Exposure', window_name, EXPOSURE_MIN)

    # [!!] 新增：FPS 计算相关的变量
    frame_counter = 0
    fps_start_time = time.time()
    display_fps = 0

    # --- 3. 实时循环 ---
    try:
        while True:
            # 读取滑动条的当前值并设置曝光
            current_exposure_setting = cv2.getTrackbarPos('Exposure', window_name)
            set_manual_exposure(CAMERA_DEVICE_PATH, current_exposure_setting)

            # 捕获一帧画面
            ret, frame = cap.read()
            if not ret:
                print("错误：无法从摄像头捕获画面。")
                break
            
            # [!!] 新增：计算FPS
            frame_counter += 1
            # 每秒更新一次FPS值
            if (time.time() - fps_start_time) > 1.0:
                display_fps = frame_counter / (time.time() - fps_start_time)
                frame_counter = 0
                fps_start_time = time.time()

            # [!!] 修改：在画面上显示所有信息
            # 曝光值信息
            exposure_text = f"Exposure: {last_set_exposure}"
            cv2.putText(frame, exposure_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # FPS信息
            fps_text = f"FPS: {display_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 显示画面
            cv2.imshow(window_name, frame)

            # 等待按键，如果按下 'q' 则退出循环
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n'q' 被按下，正在退出...")
                break
    finally:
        # --- 4. 清理和释放资源 ---
        print("正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        print("测试结束。")

if __name__ == "__main__":
    run_interactive_test()