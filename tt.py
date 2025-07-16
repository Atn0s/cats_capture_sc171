import cv2
import time
import sys
import subprocess
import shutil

# --- 用户可配置参数 ---

# 视频设备文件路径
# 在终端使用 `ls /dev/video*` 来查看你的摄像头设备
DEVICE_PATH = "/dev/video2"
# OpenCV需要设备索引号，通常与设备路径的数字对应
DEVICE_INDEX = 2

# 视频分辨率 (宽, 高)
RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720

# 目标帧率 (FPS)
FRAME_RATE = 15

# 手动曝光值 (对应 v4l2-ctl 的 'exposure_absolute')
# !!! 重要: 这个值的范围和单位完全取决于你的摄像头型号。
# 请先在终端运行 `v4l2-ctl -d /dev/video0 --list-ctrls` 来查找 'exposure_absolute' 的 min, max, step 和 default 值。
# 常见范围可能是 3 到 2047 之间，值越大，曝光时间越长（画面越亮）。
EXPOSURE_ABSOLUTE_VALUE = 2000

# 录制时长（秒）。设置为 -1 表示手动停止 (按 Ctrl+C)。
DURATION = 60

# 输出视频文件的名称
OUTPUT_FILE = f"video_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"


def check_dependencies():
    """检查 v4l2-ctl 是否已安装"""
    if not shutil.which("v4l2-ctl"):
        print("错误: 'v4l2-ctl' 命令未找到。")
        print("这是控制摄像头的关键工具。请先安装它:")
        print("sudo apt update && sudo apt install v4l-utils")
        sys.exit(1)

def set_camera_properties_v4l2():
    """使用 v4l2-ctl 工具设置摄像头硬件参数"""
    print("--- 使用 v4l2-ctl 配置摄像头 ---")
    try:
        # 1. 设置为手动曝光模式
        # 'exposure_auto=1' 通常代表手动模式(Manual)
        subprocess.run(
            ["v4l2-ctl", "-d", DEVICE_PATH, "-c", "exposure_auto=1"],
            check=True,
            capture_output=True, text=True
        )
        print("曝光模式已设置为: 手动")

        # 2. 设置绝对曝光值
        subprocess.run(
            ["v4l2-ctl", "-d", DEVICE_PATH, "-c", f"exposure_absolute={EXPOSURE_ABSOLUTE_VALUE}"],
            check=True,
            capture_output=True, text=True
        )
        print(f"绝对曝光值已设置为: {EXPOSURE_ABSOLUTE_VALUE}")

        # 3. 设置帧率
        subprocess.run(
            ["v4l2-ctl", "--device", DEVICE_PATH, "--set-parm", str(FRAME_RATE)],
            check=True,
            capture_output=True, text=True
        )
        print(f"期望帧率已设置为: {FRAME_RATE} fps")
        
        # 短暂延时确保设置生效
        time.sleep(0.5)
        print("---------------------------------")
        return True

    except subprocess.CalledProcessError as e:
        print(f"错误: 使用 v4l2-ctl 设置参数失败。")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"返回码: {e.returncode}")
        print(f"输出: {e.stdout}")
        print(f"错误信息: {e.stderr}")
        print("请检查你的摄像头是否支持这些参数和值。")
        return False
    except FileNotFoundError:
        print(f"错误: 设备 {DEVICE_PATH} 不存在。")
        return False

def main():
    """主录制函数"""
    check_dependencies()

    if not set_camera_properties_v4l2():
        sys.exit(1)
        
    # 初始化摄像头
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"错误: 无法通过 OpenCV 打开摄像头设备索引 {DEVICE_INDEX}。")
        sys.exit(1)

    # 尽管 v4l2-ctl 已设置硬件，最好也通知 OpenCV 期望的参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # 获取摄像头实际生效的分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FRAME_RATE, (actual_width, actual_height))

    print(f"\n--- 开始使用 OpenCV 录制 ---")
    print(f"将从设备 {DEVICE_INDEX} ({DEVICE_PATH}) 录制到文件 '{OUTPUT_FILE}'")
    if DURATION > 0:
        print(f"将录制 {DURATION} 秒。")
        end_time = time.time() + DURATION
    else:
        print("将持续录制，按 Ctrl+C 停止。")
        end_time = float('inf')

    try:
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                print("警告: 无法从摄像头读取帧。")
                break
    except KeyboardInterrupt:
        print("\n检测到手动停止信号 (Ctrl+C)。")
    finally:
        print("正在停止录制并保存文件...")
        cap.release()
        out.release()
        print(f"录制完成！视频已保存为: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()