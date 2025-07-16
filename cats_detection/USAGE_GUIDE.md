# 实时猫咪检测系统使用指南

## 快速开始

1. **检测可用摄像头**
   ```bash
   python3 detect_camera_devices.py
   ```

2. **运行实时检测**
   ```bash
   ./run_realtime_cat_detection_advanced.sh
   ```

3. **手动指定摄像头**
   ```bash
   python3 cats_infer_yolov8_realtime_advanced.py --video-device 0
   ```

## 完整设置流程

### 1. 环境检查
```bash
# 检查Python环境
python3 --version

# 检查OpenCV
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# 检查必要文件
ls -la cats_detection.dlc api_infer.py utils.py
```

### 2. 摄像头设置
```bash
# 检测可用摄像头
python3 detect_camera_devices.py

# 测试摄像头（如果有设备）
python3 test_camera.py 0  # 使用索引
# 或
python3 test_camera.py /dev/video0  # 使用设备路径
```

### 3. 配置调整
编辑 `config.json` 文件：
```json
{
  "video_device": 0,              // 更改为你的摄像头索引或路径
  "conf_threshold": 0.5,          // 调整置信度阈值
  "display_window": true,         // 是否显示窗口
  "save_cropped_objects": true,   // 是否保存裁剪图片
  "log_detections": true          // 是否记录检测日志
}
```

### 4. 运行检测
```bash
# 使用脚本运行（推荐）
./run_realtime_cat_detection_advanced.sh

# 直接运行Python（高级用户）
python3 cats_infer_yolov8_realtime_advanced.py --config config.json
```

## 常见问题解决

### 问题1: 找不到摄像头设备
```bash
# 解决方案1: 检查USB摄像头连接
lsusb | grep -i camera

# 解决方案2: 检查权限
sudo usermod -a -G video $USER
# 重新登录后生效

# 解决方案3: 尝试不同的摄像头索引
python3 cats_infer_yolov8_realtime_advanced.py --video-device 1
```

### 问题2: 摄像头无法打开
```bash
# 检查是否被其他程序占用
sudo lsof /dev/video*

# 重启摄像头服务
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo
```

### 问题3: 检测效果不佳
```bash
# 调整置信度阈值（降低可检测更多）
# 修改config.json中的conf_threshold值，如0.3

# 调整摄像头分辨率
# 修改config.json中的camera_width和camera_height
```

### 问题4: 性能问题
```bash
# 关闭显示窗口
python3 cats_infer_yolov8_realtime_advanced.py --no-display

# 降低帧率
# 修改config.json中的fps_limit值

# 关闭保存功能
python3 cats_infer_yolov8_realtime_advanced.py --no-save
```

## 输出文件说明

### 裁剪的猫咪图片
- 位置: `cropped_objects/`
- 格式: `frame_XXXXXX_TIMESTAMP_cat_X_confX.XX.jpg`
- 说明: 每检测到一只猫就会保存一张裁剪图片

### 检测日志
- 位置: `cropped_objects/detection_log_TIMESTAMP.json`
- 内容: 
  ```json
  {
    "statistics": {
      "total_frames": 1000,
      "frames_with_cats": 150,
      "total_cats_detected": 200,
      "detection_rate": 0.15
    },
    "detections": [...]
  }
  ```

### 保存的帧
- 按 `s` 键保存当前帧
- 格式: `detection_frame_TIMESTAMP.jpg`

## 性能优化建议

1. **硬件优化**
   - 使用GPU加速（如果可用）
   - 确保充足的内存
   - 使用高质量的USB摄像头

2. **软件优化**
   - 关闭不必要的程序
   - 调整摄像头分辨率
   - 优化检测参数

3. **参数调优**
   ```json
   {
     "conf_threshold": 0.3,      // 更低=更敏感
     "iou_threshold": 0.5,       // 更高=更少重叠
     "fps_limit": 15,            // 降低帧率
     "camera_width": 320,        // 降低分辨率
     "camera_height": 240
   }
   ```

## 开发和调试

### 调试模式
```bash
# 启用详细输出
python3 cats_infer_yolov8_realtime_advanced.py --config config.json 2>&1 | tee debug.log

# 只保存检测日志，不显示窗口
python3 cats_infer_yolov8_realtime_advanced.py --no-display --config config.json
```

### 自定义配置
```bash
# 创建自定义配置文件
cp config.json my_config.json
# 编辑 my_config.json
python3 cats_infer_yolov8_realtime_advanced.py --config my_config.json
```

## 系统要求

- **操作系统**: Linux (Ubuntu, CentOS, etc.)
- **Python版本**: 3.6+
- **必要库**: OpenCV, NumPy, SNPE
- **硬件**: USB摄像头或内置摄像头
- **权限**: 摄像头访问权限

## 联系和支持

如果遇到问题，请：
1. 检查上述常见问题解决方案
2. 运行 `detect_camera_devices.py` 确认摄像头可用
3. 查看生成的日志文件
4. 提供具体的错误信息和系统环境信息
