import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import cv2

def letterbox_torch(
    img: torch.Tensor,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """YOLO letterbox — torch 텐서 기반 (CHW or BCHW)."""
    single_img = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        single_img = True

    if not img.dtype.is_floating_point:
        img = img.to(torch.float32)

    with torch.no_grad():
        max_val = img.detach().amax()
    is_unit_scale = bool(max_val <= 1.5)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    b, c, h, w = img.shape

    r = min(new_h / h, new_w / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    ratio = (r, r)

    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h

    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad_w, new_unpad_h = new_w, new_h
        ratio = (new_w / w, new_h / h)

    dw /= 2
    dh /= 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = F.interpolate(img, size=(new_unpad_h, new_unpad_w), mode="bilinear", align_corners=False)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    pad_value = float(color[0]) / 255.0 if is_unit_scale else float(color[0])
    img = F.pad(img, (left, right, top, bottom), mode="constant", value=pad_value)

    if single_img:
        img = img.squeeze(0)

    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """YOLOv5 NMS — 자체 구현."""
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    multi_label &= nc > 1
    mi = 5 + nc

    output = [torch.zeros((0, 6 + nm), device=device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True, stable=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def masks2segments(masks, strategy='largest'):
    # Convert masks(n,160,160) into segments(n,xy)
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments

def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments

def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y

def letterbox_points_to_original(points_xy, orig_shape, input_shape):
    """
    points_xy: (M,2) or list of (M,2)  (x,y) in letterboxed input coords
    orig_shape: (orig_h, orig_w)
    input_shape: (in_h, in_w)
    return: same structure with points mapped to original coords (float32)
    """
    orig_h, orig_w = orig_shape
    in_h, in_w = input_shape

    # letterbox scale & padding (must match preprocessing)
    r = min(in_w / orig_w, in_h / orig_h)
    new_w = orig_w * r
    new_h = orig_h * r
    pad_w = (in_w - new_w) / 2.0
    pad_h = (in_h - new_h) / 2.0

    def _map_one(seg):
        seg = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        seg[:, 0] = (seg[:, 0] - pad_w) / r
        seg[:, 1] = (seg[:, 1] - pad_h) / r
        seg[:, 0] = np.clip(seg[:, 0], 0, orig_w - 1)
        seg[:, 1] = np.clip(seg[:, 1], 0, orig_h - 1)
        return seg

    if isinstance(points_xy, (list, tuple)) and len(points_xy) > 0 and np.asarray(points_xy[0]).ndim >= 1:
        # list of segments
        return [_map_one(seg) for seg in points_xy]
    else:
        # single segment
        return _map_one(points_xy)

def letterbox_xyxy_to_original(
    boxes_xyxy,            # (N,4) or (4,)
    orig_shape,            # (orig_h, orig_w)
    input_shape            # (in_h, in_w)  ex) (480, 640)
):
    orig_h, orig_w = orig_shape
    in_h, in_w = input_shape

    boxes = np.array(boxes_xyxy, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]  # (1,4)

    # letterbox 비율: 가로/세로 중 constraining dimension 기준
    # (scaleup 허용 — _letterbox_torch 기본값과 일치)
    r = min(in_h / orig_h, in_w / orig_w)

    # 양방향 패딩 계산 (16:9→pad_h만, 4:3→pad 없음, 세로→pad_w만)
    pad_w = (in_w - int(round(orig_w * r))) / 2.0
    pad_h = (in_h - int(round(orig_h * r))) / 2.0

    # undo padding
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h

    # scale back up
    boxes[:, :4] /= r

    # clip to original image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    # ensure x1<=x2, y1<=y2
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2
    boxes = np.round(boxes).astype(np.int16).tolist()

    return boxes