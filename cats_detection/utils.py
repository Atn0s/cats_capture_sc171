import cv2
import numpy as np

coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box

def NMS(dets, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    [!!] 修改后：返回保留下来的框的整数索引列表
    '''
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    
    # [!!] keep 现在是我们要返回的索引列表
    keep = []
    
    # scores.argsort() 返回的是排序后的原始索引
    index = scores.argsort()[::-1] 
    
    while index.size > 0:
        # i 现在是原始dets数组中的索引
        i = index[0]
        keep.append(i)
        
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
        overlaps = w * h
        
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        # 找到需要保留的框的索引（在index数组中的位置）
        idx = np.where(ious <= thresh)[0]
        
        # 更新index数组，只保留那些与当前最大分框IOU小于阈值的框
        index = index[idx + 1] # index[1:] 的索引是 idx，所以要 +1
 
    # [!!] 返回一个整数索引的列表
    return keep

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_img(img, target_shape:tuple=None, div_num=255, means:list=[0.485, 0.456, 0.406], stds:list=[0.229, 0.224, 0.225]):
    '''
    图像预处理:
    target_shape: 目标shape
    div_num: 归一化除数
    means: len(means)==图像通道数，通道均值, None不进行zscore
    stds: len(stds)==图像通道数，通道方差, None不进行zscore
    '''
    img_processed = np.copy(img)
    # resize
    if target_shape:
        img_processed = cv2.resize(img_processed, target_shape)
        #img_processed = letterbox(img_processed, target_shape, stride=None, auto=False)[0]

    img_processed = img_processed.astype(np.float32)
    img_processed = img_processed/div_num

    # z-score
    if means is not None and stds is not None:
        means = np.array(means).reshape(1, 1, -1)
        stds = np.array(stds).reshape(1, 1, -1)
        img_processed = (img_processed-means)/stds

    # unsqueeze
    img_processed = img_processed[None, :]

    return img_processed.astype(np.float32)
    
def convert_shape(shapes:tuple or list, int8=False):
    '''
    转化为aidlite需要的格式
    '''
    if isinstance(shapes, tuple):
        shapes = [shapes]
    out = []
    for shape in shapes:
        nums = 1 if int8 else 4
        for n in shape:
            nums *= n
        out.append(nums)
    return out

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2

def detect_postprocess(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    '''
    检测输出后处理
    prediction: aidlite模型预测输出
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    '''
    h, w, _ = img1shape
    cls_num = prediction.shape[-1] - 5
    valid_condidates = prediction[prediction[..., 4] > conf_thres]
    valid_condidates[:, 0] *= w
    valid_condidates[:, 1] *= h
    valid_condidates[:, 2] *= w
    valid_condidates[:, 3] *= h
    valid_condidates[:, :4] = xywh2xyxy(valid_condidates[:, :4])
    valid_condidates = valid_condidates[(valid_condidates[:, 0] > 0) & (valid_condidates[:, 1] > 0) & (valid_condidates[:, 2] > 0) & (valid_condidates[:, 3] > 0)]
    box_cls = valid_condidates[:, 5:].argmax(1)
    cls_box = []
    for i in range(cls_num):
        temp_boxes = valid_condidates[box_cls == i]
        if(len(temp_boxes) == 0):
            cls_box.append([])
            continue
        temp_boxes = NMS(temp_boxes, iou_thres)
        temp_boxes[:, :4] = scale_coords([h, w], temp_boxes[:, :4] , img0shape).round()
        temp_boxes[:, :4] = xyxy2xywh(temp_boxes[:, :4])
        cls_box.append(temp_boxes[:, :5])
    return cls_box

def draw_detect_res(img, all_boxes, class_names=['cat']):
    """
    检测结果绘制
    img: 输入的RGB图像 (numpy array)
    all_boxes: 包含了所有类别检测结果的列表。
               每个类别的结果是一个 (N, 5) 的numpy数组，格式为 [x, y, w, h, score]
    class_names: 类别名称列表
    """
    # 将图像转换为可绘制的格式
    img_display = np.copy(img).astype(np.uint8)

    # 为不同类别的框设置不同的颜色（如果多于一个类）
    num_classes = len(class_names)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(num_classes)]
    if num_classes == 1:
        colors = [(0, 255, 0)] # 如果只有一个类，固定为亮绿色

    # 遍历每个类别的检测结果
    for class_id, class_detections in enumerate(all_boxes):
        if len(class_detections) == 0:
            continue

        # 获取该类别的颜色和名称
        color = colors[class_id]
        class_name = class_names[class_id]

        # 遍历该类别下的每一个检测框
        for box in class_detections:
            x, y, w, h = [int(t) for t in box[:4]]
            score = box[4]  # [!!] 获取置信度分数

            # --- 绘制边界框 ---
            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, thickness=2)

            # --- 准备标签文本，包含类别和置信度 ---
            # [!!] 格式化文本，例如 "cat: 0.98"
            label = f'{class_name}: {score:.2f}'

            # --- 绘制文本背景以便于阅读 ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            
            # 获取文本框的尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # 确保背景框不会超出图像顶部
            label_y = max(y, text_height + 10)
            
            # 绘制一个实心的矩形作为文本背景
            cv2.rectangle(img_display, (x, label_y - text_height - 10), (x + text_width, label_y - baseline), color, cv2.FILLED)
            
            # --- 绘制文本 ---
            # 文本颜色设为黑色以便在亮色背景上阅读
            cv2.putText(img_display, label, (x, label_y - 7), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return img_display

def detect_postprocess_yolov8(prediction, img0_shape, img1_shape, num_classes=80, conf_thres=0.25, iou_thres=0.45):
    """
    YOLOv8的检测输出后处理
    prediction: 模型原始输出, shape: [1, 4+num_classes, 8400]
    ...
    num_classes: 模型的类别数量
    ...
    """
    channels = 4 + num_classes
    expected_shape_msg = f"Prediction shape should be [1, {channels}, 8400]"

    # 检查输入维度
    if len(prediction.shape) == 3 and prediction.shape[0] == 1:
        # [!!] 关键修改：检查动态的channels数量
        if prediction.shape[1] != channels:
            raise ValueError(f"Incorrect channel size. Expected {channels}, but got {prediction.shape[1]}. {expected_shape_msg}")
        prediction = prediction[0] # 去掉batch维度, [channels, 8400]
    else:
        raise ValueError(f"Incorrect shape. {expected_shape_msg}")

    # 1. 转置输出: [channels, 8400] -> [8400, channels]
    prediction = prediction.T

    # 2. 过滤掉置信度低的框
    # [!!] 关键修改：现在只有一个类别分数，它就是置信度
    if num_classes == 1:
        scores = prediction[:, 4] # 直接取第5个元素作为分数
    else:
        scores = np.max(prediction[:, 4:], axis=1) # 如果有多个类，取最大分

    valid_candidates = prediction[scores > conf_thres]
    scores = scores[scores > conf_thres]

    if len(valid_candidates) == 0:
        # [!!] 关键修改：使用您自己的类别列表或一个通用列表
        # 为了简单起见，我们假设您的类别叫 'cat'
        your_classes = ['cat'] # 您可以从外部传入这个列表
        return [[] for _ in your_classes]

    # 3. 解码边界框 (这部分不变)
    boxes_xywh = valid_candidates[:, :4]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # 4. 获取类别ID
    # [!!] 关键修改：如果只有一个类，那么ID永远是0
    if num_classes == 1:
        class_ids = np.zeros(len(valid_candidates), dtype=int)
    else:
        class_ids = np.argmax(valid_candidates[:, 4:], axis=1)

    # 5. 按类别进行NMS (这部分逻辑不变，但循环次数会改变)
    cls_box = [[] for _ in range(num_classes)] # 创建一个大小为num_classes的列表
    for i in range(num_classes):
        mask = (class_ids == i)
        if not np.any(mask):
            continue

        class_boxes = boxes_xyxy[mask]
        class_scores = scores[mask]

        dets_for_nms = np.hstack((class_boxes, class_scores[:, np.newaxis])).astype(np.float32)
        
        keep_indices = NMS(dets_for_nms, iou_thres)
        
        if len(keep_indices) > 0:
            final_boxes_xyxy = dets_for_nms[keep_indices, :4]
            final_scores = dets_for_nms[keep_indices, 4]

            final_boxes_xyxy = scale_coords(img1_shape[:2], final_boxes_xyxy, img0_shape[:2])
            final_boxes_xywh = xyxy2xywh(final_boxes_xyxy)
            
            cls_box[i] = np.hstack((final_boxes_xywh, final_scores[:, np.newaxis]))

    return cls_box