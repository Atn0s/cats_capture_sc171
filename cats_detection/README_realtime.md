# 实时猫咪检测系统

这个项目提供了使用 `/dev/video2` 摄像头进行实时猫咪检测的功能。

## 文件说明

### 主要脚本

1. **`cats_infer_yolov8_realtime.py`** - 基础实时检测脚本
2. **`cats_infer_yolov8_realtime_advanced.py`** - 高级实时检测脚本（推荐）
3. **`test_camera.py`** - 摄像头测试脚本

### 配置文件

- **`config.json`** - 检测系统配置文件

### 运行脚本

- **`run_realtime_cat_detection.sh`** - 运行基础版本
- **`run_realtime_cat_detection_advanced.sh`** - 运行高级版本（推荐）

## 使用方法

### 1. 测试摄像头

在运行猫咪检测之前，先测试摄像头是否正常工作：

```bash
# 测试 /dev/video2
python3 test_camera.py

# 测试其他摄像头设备
python3 test_camera.py /dev/video0
```

### 2. 运行实时检测

#### 方法一：使用脚本（推荐）

```bash
# 运行高级版本
./run_realtime_cat_detection_advanced.sh

# 或运行基础版本
./run_realtime_cat_detection.sh
```

#### 方法二：直接运行Python脚本

```bash
# 基础版本
python3 cats_infer_yolov8_realtime.py

# 高级版本
python3 cats_infer_yolov8_realtime_advanced.py

# 高级版本带参数
python3 cats_infer_yolov8_realtime_advanced.py --video-device /dev/video0 --no-display
```

### 3. 控制键

运行时的控制键：
- **`q`** - 退出程序
- **`s`** - 保存当前帧
- **`p`** - 暂停/恢复（仅高级版本）

## 功能特性

### 基础版本特性
- 实时猫咪检测
- 检测结果可视化
- 自动保存检测到的猫咪图片
- FPS显示
- 检测信息输出

### 高级版本特性
- 所有基础版本特性
- 配置文件支持
- 命令行参数支持
- 检测日志记录
- 详细统计信息
- 暂停/恢复功能
- 时间戳记录
- JSON格式的检测日志

## 配置说明

可以通过修改 `config.json` 文件来调整检测参数：

```json
{
  "video_device": "/dev/video2",           // 摄像头设备路径
  "conf_threshold": 0.5,                   // 置信度阈值
  "iou_threshold": 0.45,                   // IoU阈值
  "display_window": true,                  // 是否显示窗口
  "save_cropped_objects": true,            // 是否保存裁剪的猫咪图片
  "fps_limit": 30,                         // 帧率限制
  "camera_width": 640,                     // 摄像头宽度
  "camera_height": 480,                    // 摄像头高度
  "log_detections": true,                  // 是否记录检测日志
  "save_detection_log": true               // 是否保存检测日志
}
```

## 输出文件

### 裁剪的猫咪图片
- 保存位置：`cropped_objects/` 目录
- 文件名格式：`frame_XXXXXX_TIMESTAMP_cat_X_confX.XX.jpg`
- 包含检测到的每只猫的裁剪图片

### 检测日志
- 保存位置：`cropped_objects/` 目录
- 文件名格式：`detection_log_TIMESTAMP.json`
- 包含详细的检测信息和统计数据

### 保存的帧
- 按 `s` 键保存的完整帧
- 文件名格式：`detection_frame_TIMESTAMP.jpg`

## 故障排除

### 1. 摄像头问题
```bash
# 检查可用的摄像头设备
ls -la /dev/video*

# 测试摄像头
python3 test_camera.py /dev/video0
```

### 2. 权限问题
```bash
# 添加用户到video组
sudo usermod -a -G video $USER

# 重新登录后生效
```

### 3. 依赖问题
确保已安装以下依赖：
- OpenCV (cv2)
- NumPy
- SNPE框架和相关API

### 4. 模型文件
确保以下文件存在：
- `cats_detection.dlc` - 模型文件
- `api_infer.py` - SNPE推理API
- `utils.py` - 工具函数

## 性能优化

1. **调整置信度阈值**：降低 `conf_threshold` 可以检测更多猫咪，但可能增加误检
2. **调整IoU阈值**：调整 `iou_threshold` 可以控制重叠检测框的合并
3. **设置合适的帧率**：根据硬件性能调整 `fps_limit`
4. **关闭不必要的功能**：设置 `display_window: false` 可以提高性能

## 系统要求

- Linux系统
- Python 3.x
- OpenCV
- SNPE框架
- 兼容的USB摄像头（/dev/video2）

## 注意事项

1. 确保摄像头设备路径正确
2. 检查用户是否有摄像头访问权限
3. 模型文件路径要正确
4. 足够的磁盘空间用于保存图片和日志
