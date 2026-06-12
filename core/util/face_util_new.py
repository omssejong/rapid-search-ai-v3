import numpy as np
import math
import cv2

class YOLOv8_face:
    def __init__(self, model_input_size_hw, conf_thres=0.2, iou_thres=0.5, gpu_index=0):
        self.gpu_index = gpu_index
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # model input size (H,W)
        self.input_height, self.input_width = map(int, model_input_size_hw)

        self.reg_max = 16
        self.project = np.arange(self.reg_max, dtype=np.float32)
        self.strides = (8, 16, 32)

        # anchors cache: key=(stride, Hf, Wf) -> (Hf*Wf,2)
        self._anchor_cache = {}

    def _make_anchors_hw(self, stride, Hf, Wf, grid_cell_offset=0.5):
        key = (int(stride), int(Hf), int(Wf), float(grid_cell_offset))
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        x = np.arange(0, Wf, dtype=np.float32) + grid_cell_offset
        y = np.arange(0, Hf, dtype=np.float32) + grid_cell_offset
        sx, sy = np.meshgrid(x, y)
        anchors = np.stack((sx, sy), axis=-1).reshape(-1, 2)  # (Hf*Wf,2)
        self._anchor_cache[key] = anchors
        return anchors

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_height, self.input_width
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    
    def resize_image_new(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_height, self.input_width
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            
            # 이미지가 목표 크기보다 작은지 확인
            if srcimg.shape[0] <= self.input_height and srcimg.shape[1] <= self.input_width:
                # 작은 이미지는 resize 없이 패딩만
                img = srcimg.copy()
                newh, neww = srcimg.shape[0], srcimg.shape[1]
                
                # 중앙 정렬을 위한 패딩 계산
                top = int((self.input_height - newh) * 0.5)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 
                                        left, self.input_width - neww - left, 
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            else:
                # 큰 이미지는 기존 로직대로 resize 후 패딩
                if hw_scale > 1:
                    newh, neww = self.input_height, int(self.input_width / hw_scale)
                    img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                    left = int((self.input_width - neww) * 0.5)
                    img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                            value=(114, 114, 114))
                else:
                    newh, neww = int(self.input_height * hw_scale), self.input_width
                    img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                    top = int((self.input_height - newh) * 0.5)
                    img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                            value=(114, 114, 114))
        else:
            # 정사각형이거나 keep_ratio=False인 경우
            if srcimg.shape[0] <= self.input_height and srcimg.shape[1] <= self.input_width:
                # 작은 이미지는 패딩만
                img = srcimg.copy()
                newh, neww = srcimg.shape[0], srcimg.shape[1]
                top = int((self.input_height - newh) * 0.5)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top,
                                        left, self.input_width - neww - left,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            else:
                # 큰 이미지는 resize
                img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
        
        
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    # def post_process(self, preds, scale_h, scale_w, padh, padw, yolov8_cnf):
    #     bboxes, scores, landmarks = [], [], []

    #     for pred in preds:
    #         # pred: torch.Tensor (B,C,Hf,Wf)
    #         pred = pred.detach().float().cpu().numpy()

    #         B, C, Hf, Wf = pred.shape

    #         # stride 계산을 H/W 둘 다로 검증
    #         stride_h = self.input_height // Hf
    #         stride_w = self.input_width  // Wf
    #         if stride_h != stride_w:
    #             raise ValueError(f"Stride mismatch: stride_h={stride_h}, stride_w={stride_w}, pred_hw={(Hf,Wf)}, input_hw={(self.input_height,self.input_width)}")
    #         stride = int(stride_h)

    #         # anchors를 pred 해상도에 맞춰 생성
    #         anchors = self._make_anchors_hw(stride, Hf, Wf)  # (Hf*Wf,2)

    #         pred = pred.transpose((0, 2, 3, 1))  # (B,Hf,Wf,C)

    #         box = pred[..., :self.reg_max * 4]
    #         cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
    #         kpts = pred[..., -15:].reshape((-1, 15))

    #         tmp = box.reshape(-1, 4, self.reg_max)
    #         bbox_pred = self.softmax(tmp, axis=-1)
    #         bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

    #         bbox = self.distance2bbox(anchors, bbox_pred, max_shape=(self.input_height, self.input_width)) * stride

    #         kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (anchors[:, 0].reshape((-1, 1)) - 0.5)) * stride
    #         kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (anchors[:, 1].reshape((-1, 1)) - 0.5)) * stride
    #         kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

    #         bbox -= np.array([[padw, padh, padw, padh]], dtype=np.float32)
    #         bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]], dtype=np.float32)

    #         kpts -= np.tile(np.array([padw, padh, 0], dtype=np.float32), 5).reshape((1, 15))
    #         kpts *= np.tile(np.array([scale_w, scale_h, 1], dtype=np.float32), 5).reshape((1, 15))

    #         bboxes.append(bbox)
    #         scores.append(cls)
    #         landmarks.append(kpts)

    #     # 이하 NMS 로직은 기존 그대로
    #     if isinstance(cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), yolov8_cnf,
    #                                self.iou_threshold), tuple):
    #         return np.array([]), np.array([]), np.array([]), np.array([])
            
    #     indices = cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), yolov8_cnf,
    #                                self.iou_threshold).flatten()
    #     if len(indices) > 0:
    #         mlvl_bboxes = bboxes[indices]
    #         confidences = confidences[indices]
    #         classIds = classIds[indices]
    #         landmarks = landmarks[indices]
    #         return mlvl_bboxes, confidences, classIds, landmarks
    #     else:
    #         return np.array([]), np.array([]), np.array([]), np.array([])

    # def post_process_renew(self, preds, scale_h, scale_w, padh, padw, yolov8_cnf):
    #     bboxes_all, scores_all, kpts_all = [], [], []

    #     for pred in preds:
    #         pred = pred.detach().float().cpu().numpy()
    #         B, C, Hf, Wf = pred.shape

    #         stride_h = self.input_height // Hf
    #         stride_w = self.input_width  // Wf
    #         if stride_h != stride_w:
    #             raise ValueError(f"Stride mismatch: stride_h={stride_h}, stride_w={stride_w}, pred_hw={(Hf,Wf)}, input_hw={(self.input_height,self.input_width)}")
    #         stride = int(stride_h)

    #         anchors = self._make_anchors_hw(stride, Hf, Wf)  # (Hf*Wf,2)
    #         pred = pred.transpose((0, 2, 3, 1))  # (B,Hf,Wf,C)

    #         box = pred[..., :self.reg_max * 4]
    #         cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,))   # (N,)
    #         kpts = pred[..., -15:].reshape((-1, 15))                                   # (N,15)

    #         tmp = box.reshape(-1, 4, self.reg_max)
    #         bbox_pred = self.softmax(tmp, axis=-1)
    #         bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

    #         bbox = self.distance2bbox(anchors, bbox_pred, max_shape=(self.input_height, self.input_width)) * stride

    #         kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (anchors[:, 0].reshape((-1, 1)) - 0.5)) * stride
    #         kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (anchors[:, 1].reshape((-1, 1)) - 0.5)) * stride
    #         kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

    #         # letterbox 역변환
    #         bbox -= np.array([[padw, padh, padw, padh]], dtype=np.float32)
    #         bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]], dtype=np.float32)

    #         kpts -= np.tile(np.array([padw, padh, 0], dtype=np.float32), 5).reshape((1, 15))
    #         kpts *= np.tile(np.array([scale_w, scale_h, 1], dtype=np.float32), 5).reshape((1, 15))

    #         bboxes_all.append(bbox.astype(np.float32))     # (N,4) xyxy
    #         scores_all.append(cls.astype(np.float32))      # (N,)
    #         kpts_all.append(kpts.astype(np.float32))       # (N,15)

    #     # ---- concat ----
    #     bboxes = np.concatenate(bboxes_all, axis=0) if bboxes_all else np.zeros((0, 4), np.float32)
    #     confidences = np.concatenate(scores_all, axis=0) if scores_all else np.zeros((0,), np.float32)
    #     landmarks = np.concatenate(kpts_all, axis=0) if kpts_all else np.zeros((0, 15), np.float32)

    #     if bboxes.shape[0] == 0:
    #         return np.array([]), np.array([]), np.array([]), np.array([])

    #     # ---- threshold ----
    #     keep = confidences >= float(yolov8_cnf)
    #     bboxes = bboxes[keep]
    #     confidences = confidences[keep]
    #     landmarks = landmarks[keep]

    #     if bboxes.shape[0] == 0:
    #         return np.array([]), np.array([]), np.array([]), np.array([])

    #     # ---- cv2 NMSBoxes는 보통 xywh 기대 -> 변환 ----
    #     xywh = bboxes.copy()
    #     xywh[:, 2] = xywh[:, 2] - xywh[:, 0]  # w = x2-x1
    #     xywh[:, 3] = xywh[:, 3] - xywh[:, 1]  # h = y2-y1

    #     boxes_list = xywh.tolist()
    #     conf_list = confidences.tolist()

    #     idxs = cv2.dnn.NMSBoxes(
    #         boxes_list, conf_list,
    #         float(yolov8_cnf),
    #         float(self.iou_threshold)
    #     )

    #     if idxs is None or len(idxs) == 0:
    #         return np.array([]), np.array([]), np.array([]), np.array([])

    #     if isinstance(idxs, tuple):
    #         # 드물게 tuple로 오는 케이스 방어
    #         idxs = np.array(idxs).reshape(-1)
    #     else:
    #         idxs = np.array(idxs).reshape(-1)

    #     mlvl_bboxes = bboxes[idxs]
    #     confidences = confidences[idxs]
    #     landmarks = landmarks[idxs]

    #     # face 단일 클래스라면 classIds는 0으로 통일
    #     classIds = np.zeros((mlvl_bboxes.shape[0],), dtype=np.int32)

    #     return mlvl_bboxes, confidences, classIds, landmarks
    
    def post_process_trt(self, preds65, kpt_out_15x25200, scale_h, scale_w, padh, padw, yolov8_cnf):
        """
        preds65: [P8,P16,P32] each (1,65,Hf,Wf)
        kpt_out_15x25200: (1,15,25200)  # TRT output
        """
        bboxes, scores, landmarks = [], [], []

        # (1,15,25200) -> (25200,15)
        kpt = kpt_out_15x25200.detach().float().cpu().numpy()[0].transpose(1, 0).astype(np.float32)

        # split sizes from feature maps
        n_list = [int(p.shape[2] * p.shape[3]) for p in preds65]
        if sum(n_list) != kpt.shape[0]:
            raise ValueError(f"kpt split mismatch: sum={sum(n_list)} vs kpt={kpt.shape[0]}")

        splits = []
        s = 0
        for n in n_list:
            splits.append(kpt[s:s+n])
            s += n

        for i, pred in enumerate(preds65):
            pred = pred.detach().float().cpu().numpy()  # (1,65,Hf,Wf)
            B, C, Hf, Wf = pred.shape

            stride_h = self.input_height // Hf
            stride_w = self.input_width  // Wf
            if stride_h != stride_w:
                raise ValueError(f"Stride mismatch: stride_h={stride_h}, stride_w={stride_w}, pred_hw={(Hf,Wf)}, input_hw={(self.input_height,self.input_width)}")
            stride = int(stride_h)

            # anchors from (stride, Hf, Wf)
            anchors = self._make_anchors_hw(stride, Hf, Wf)  # (Hf*Wf,2)

            pred = pred.transpose((0, 2, 3, 1))  # (1,Hf,Wf,65)

            # box + cls
            box = pred[..., :self.reg_max * 4]          # (1,Hf,Wf,64)
            cls_logit = pred[..., self.reg_max * 4]     # (1,Hf,Wf)
            cls = 1 / (1 + np.exp(-cls_logit))          # (1,Hf,Wf)
            cls = cls.reshape((-1, 1)).astype(np.float32)  # (Hf*Wf,1)

            # DFL decode
            tmp = box.reshape(-1, 4, self.reg_max).astype(np.float32)
            bbox_prob = self.softmax(tmp, axis=-1).astype(np.float32)
            bbox_pred = np.dot(bbox_prob, self.project).reshape((-1, 4)).astype(np.float32)

            bbox = self.distance2bbox(anchors, bbox_pred, max_shape=(self.input_height, self.input_width)).astype(np.float32)
            bbox *= float(stride)

            # kpts decode using split + anchors
            kpts = splits[i].copy()  # (Hf*Wf,15)
            ax = anchors[:, 0:1]
            ay = anchors[:, 1:2]
            st = float(stride)

            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (ax - 0.5)) * st
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (ay - 0.5)) * st
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            # letterbox inverse to original
            bbox -= np.array([[padw, padh, padw, padh]], dtype=np.float32)
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]], dtype=np.float32)

            kpts -= np.tile(np.array([padw, padh, 0], dtype=np.float32), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1], dtype=np.float32), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        # concat
        bboxes = np.concatenate(bboxes, axis=0) if bboxes else np.zeros((0,4), np.float32)
        scores = np.concatenate(scores, axis=0) if scores else np.zeros((0,1), np.float32)
        landmarks = np.concatenate(landmarks, axis=0) if landmarks else np.zeros((0,15), np.float32)

        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # xywh for cv2 NMS
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2] = bboxes_wh[:, 2] - bboxes_wh[:, 0]
        bboxes_wh[:, 3] = bboxes_wh[:, 3] - bboxes_wh[:, 1]

        confidences = scores.reshape(-1)
        classIds = np.zeros((confidences.shape[0],), dtype=np.int32)

        # threshold
        mask = confidences > float(yolov8_cnf)
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        if bboxes_wh.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        idxs = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), float(yolov8_cnf), float(self.iou_threshold))
        if idxs is None or len(idxs) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        idxs = np.array(idxs).reshape(-1)
        return bboxes_wh[idxs], confidences[idxs], classIds[idxs], landmarks[idxs]


def rotate_img(img, angle, x, y):
    image_center = tuple((x, y))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def face_alignment(face_image, landmark, cropped_box):
    lm_point_list = list()
    # for lm in landmark:
    #     for k in range(5):
    #         lm_point_list.append([lm[3*k+0], lm[3*k+1]])
    
    # for lm in landmark:
    #     points = [(lm[i], lm[i+1]) for i in range(0, 15, 3)]
    #     lm_point_list.extend(points)
    
    points = [(landmark[i], landmark[i+1]) for i in range(0, 15, 3)]
    lm_point_list.extend(points)
    
    le_pt = [lm_point_list[0][i] - cropped_box[i] for i in range(len(lm_point_list[0]))]
    re_pt = [lm_point_list[1][i] - cropped_box[i] for i in range(len(lm_point_list[1]))]
    n_pt = [lm_point_list[2][i] - cropped_box[i] for i in range(len(lm_point_list[2]))]
    lm_pt = [lm_point_list[3][i] - cropped_box[i] for i in range(len(lm_point_list[3]))]
    rm_pt = [lm_point_list[4][i] - cropped_box[i] for i in range(len(lm_point_list[4]))]
    
    # temp = []
    # for i in range(1, len(landmark), 2):
    #     temp.append([landmark[i-1], landmark[i]])
        
    #le_pt, re_pt, n_pt, lm_pt, rm_pt = landmark[:5]
    #le_pt, re_pt, n_pt, lm_pt, rm_pt = temp[:5]
    
    margin = int(abs(re_pt[0] - le_pt[0]))
    h, w = face_image.shape[:2]
    # diagonal = int(math.sqrt(h**2 + w**2))
    # margin = diagonal
    #margin = max(h, w)
    # diagonal = int(math.sqrt(h**2 + w**2))
    # margin = (diagonal - min(h, w)) // 2 + int(abs(re_pt[0] - le_pt[0]))
    margin_img = cv2.copyMakeBorder(face_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, 0)
    
    le_pt = (le_pt[0] + margin, le_pt[1]+margin)
    re_pt = (re_pt[0] + margin, re_pt[1]+margin)
    n_pt = (n_pt[0] + margin, n_pt[1]+margin)
    lm_pt = (lm_pt[0] + margin, lm_pt[1]+margin)
    rm_pt = (rm_pt[0] + margin, rm_pt[1]+margin)
    
    face_original_distance = math.sqrt(math.pow((le_pt[0]+re_pt[0])/2 - (lm_pt[0] + rm_pt[0])/2, 2)
                                  + math.pow((le_pt[1]+re_pt[1])/2 - (lm_pt[1] + rm_pt[1])/2, 2))
    face_scaler_factor = abs(48./face_original_distance)
    
    center_eye_point = ((le_pt[0]+re_pt[0])/2, (le_pt[1]+re_pt[1])/2)
    center_mouth_point = ((lm_pt[0]+rm_pt[0])/2, (lm_pt[1]+rm_pt[1])/2)
    
    eye_x_distance = re_pt[0] - le_pt[0]
    eye_y_distance = re_pt[1] - le_pt[1]
    
    # 좌측 우측 눈의 x,y 좌표를 이용한 두 눈의 유클리디안 거리 계산
    eye_euclidean_distance = math.hypot(re_pt[0] - le_pt[0], 
                                        re_pt[1] - le_pt[1]) 
    
    # box width에 대한 얼굴 내 양눈 유클리디안 거리의 비율
    face_ratio = eye_euclidean_distance / (cropped_box[2] - cropped_box[0])
    PI = math.pi
    
    if face_ratio > 0.2:
        radian = math.atan2(re_pt[1]-le_pt[1], re_pt[0]-le_pt[0])    
        degree = (radian * 180)/PI
    else:
        radian = math.atan2(center_eye_point[1] - center_mouth_point[1], center_eye_point[0] - center_mouth_point[1])
        degree = (radian * 180)/PI + 90
    
    aligned_face = rotate_img(margin_img, degree, center_eye_point[0], center_eye_point[1])
    scale_img = cv2.resize(aligned_face, (int(float(margin_img.shape[1])*face_scaler_factor), int(float(margin_img.shape[0])*face_scaler_factor)))
    h, w, c = aligned_face.shape
    
    aligned_face_half = aligned_face[:int(n_pt[1]), :]
    half_face_ratio = n_pt[1]/h
    scale_img_half = cv2.resize(aligned_face_half, (abs(int(float(margin_img.shape[1]) * face_scaler_factor)), abs(int(float(margin_img.shape[0])*face_scaler_factor * half_face_ratio))))
    
    
    
    new_center_x = int(float(center_eye_point[0]) * face_scaler_factor)
    new_center_y = int(float(center_eye_point[1]) * face_scaler_factor)
    
    roi_img = scale_img[
                        new_center_y-40 if new_center_y-40 > 0 else 0:new_center_y+88 if new_center_y+88 > 0 else 0, 
                        new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
                        ].copy()
    
    roi_img_half = scale_img_half[
                    new_center_y-40 if new_center_y-40 > 0 else 0:, 
                    new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
                    ].copy()
    
    # print(f"============================================={frame_idx}_{i}====================================================")
    # print(f"le_pt local: {le_pt}")
    # print(f"re_pt local: {re_pt}")
    # print(f"margin: {margin}")
    # print(f"margin_img shape: {margin_img.shape}")
    # print(f"center_eye_point: {center_eye_point}")
    # print(f"degree: {degree}")
    # print(f"scale factor: {face_scaler_factor}")
    # print(f"new_center: ({new_center_x}, {new_center_y})")
    # print(f"scale_img shape: {scale_img.shape}")
    # print(f"================================================================================================================")
    
    # cv2.imwrite(f"./yolov8_face_crop_ratio_1/frame_{frame_idx}/debug_{i}_aligned.png", aligned_face)
    # cv2.imwrite(f"./yolov8_face_crop_ratio_1/frame_{frame_idx}/debug_{i}_scale.png", scale_img)
    # cv2.imwrite(f"./yolov8_face_crop_ratio_1/frame_{frame_idx}/debug_{i}_roi.png", roi_img)
    

    return roi_img, roi_img_half

def face_alignment_renew(face_image, landmarks, cropped_box):
    # lm_point_list = list()
    # for lm in landmark:
    #     for k in range(5):
    #         lm_point_list.append([lm[3*k+0], lm[3*k+1]])
    
    # for lm in landmark:
    #     points = [(lm[i], lm[i+1]) for i in range(0, 15, 3)]
    #     lm_point_list.extend(points)
    
    # points = [(landmark[i], landmark[i+1]) for i in range(0, 15, 3)]
    # lm_point_list.extend(points)
        
    
    le_pt = [landmarks[0][i] - cropped_box[i] for i in range(len(landmarks[0]))]
    re_pt = [landmarks[1][i] - cropped_box[i] for i in range(len(landmarks[1]))]
    n_pt = [landmarks[2][i] - cropped_box[i] for i in range(len(landmarks[2]))]
    lm_pt = [landmarks[3][i] - cropped_box[i] for i in range(len(landmarks[3]))]
    rm_pt = [landmarks[4][i] - cropped_box[i] for i in range(len(landmarks[4]))]
    

    #le_pt, re_pt, n_pt, lm_pt, rm_pt = landmark[:5]
    #le_pt, re_pt, n_pt, lm_pt, rm_pt = landmarks[:5]
    
    
    margin = int(abs(re_pt[0] - le_pt[0]))
    margin_img = cv2.copyMakeBorder(face_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, 0)
    
    le_pt = (le_pt[0] + margin, le_pt[1]+margin)
    re_pt = (re_pt[0] + margin, re_pt[1]+margin)
    n_pt = (n_pt[0] + margin, n_pt[1]+margin)
    lm_pt = (lm_pt[0] + margin, lm_pt[1]+margin)
    rm_pt = (rm_pt[0] + margin, rm_pt[1]+margin)
    
    face_original_distance = math.sqrt(math.pow((le_pt[0]+re_pt[0])/2 - (lm_pt[0] + rm_pt[0])/2, 2)
                                  + math.pow((le_pt[1]+re_pt[1])/2 - (lm_pt[1] + rm_pt[1])/2, 2))
    face_scaler_factor = abs(48./face_original_distance)
    
    center_eye_point = ((le_pt[0]+re_pt[0])/2, (le_pt[1]+re_pt[1])/2)
    center_mouth_point = ((lm_pt[0]+rm_pt[0])/2, (lm_pt[1]+rm_pt[1])/2)
    
    eye_x_distance = re_pt[0] - le_pt[0]
    eye_y_distance = re_pt[1] - le_pt[1]
    
    # 좌측 우측 눈의 x,y 좌표를 이용한 두 눈의 유클리디안 거리 계산
    eye_euclidean_distance = math.hypot(re_pt[0] - le_pt[0], 
                                        re_pt[1] - le_pt[1]) 
    
    # box width에 대한 얼굴 내 양눈 유클리디안 거리의 비율
    #face_ratio = eye_euclidean_distance / (cropped_box[2] - cropped_box[0])
    #face_ratio = eye_euclidean_distance / face_image.shape[1]
    face_ratio = eye_euclidean_distance / (cropped_box[2] - cropped_box[0])
    PI = math.pi
    
    if face_ratio > 0.2:
        radian = math.atan2(re_pt[1]-le_pt[1], re_pt[0]-le_pt[0])    
        degree = (radian * 180)/PI
    else:
        radian = math.atan2(center_eye_point[1] - center_mouth_point[1], center_eye_point[0] - center_mouth_point[1])
        degree = (radian * 180)/PI + 90
    
    aligned_face = rotate_img(margin_img, degree, center_eye_point[0], center_eye_point[1])
    scale_img = cv2.resize(aligned_face, (int(float(margin_img.shape[1])*face_scaler_factor), int(float(margin_img.shape[0])*face_scaler_factor)))
    h, w, c = aligned_face.shape
    
    aligned_face_half = aligned_face[:int(n_pt[1]), :]
    half_face_ratio = n_pt[1]/h
    scale_img_half = cv2.resize(aligned_face_half, (abs(int(float(margin_img.shape[1]) * face_scaler_factor)), abs(int(float(margin_img.shape[0])*face_scaler_factor * half_face_ratio))))
    
    
    
    new_center_x = int(float(center_eye_point[0]) * face_scaler_factor)
    new_center_y = int(float(center_eye_point[1]) * face_scaler_factor)
    
    roi_img = scale_img[
                        new_center_y-40 if new_center_y-40 > 0 else 0:new_center_y+88 if new_center_y+88 > 0 else 0, 
                        new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
                        ].copy()
    
    roi_img_half = scale_img_half[
                    new_center_y-40 if new_center_y-40 > 0 else 0:, 
                    new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
                    ].copy()
    
    return roi_img, roi_img_half

def face_alignment_advise(face_image, landmark, cropped_box):
    """
    개선된 얼굴 정렬 함수 (RetinaFace, YOLOv8 모두 적용)
    """
    lm_point_list = list()
    points = [(landmark[i], landmark[i+1]) for i in range(0, 15, 3)]
    lm_point_list.extend(points)
    
    le_pt = [lm_point_list[0][i] - cropped_box[i] for i in range(len(lm_point_list[0]))]
    re_pt = [lm_point_list[1][i] - cropped_box[i] for i in range(len(lm_point_list[1]))]
    n_pt = [lm_point_list[2][i] - cropped_box[i] for i in range(len(lm_point_list[2]))]
    lm_pt = [lm_point_list[3][i] - cropped_box[i] for i in range(len(lm_point_list[3]))]
    rm_pt = [lm_point_list[4][i] - cropped_box[i] for i in range(len(lm_point_list[4]))]
    
    # ========== 핵심 개선: margin 계산 ==========
    h, w = face_image.shape[:2]
    
    # 1. 랜드마크 중심 계산
    all_pts = [le_pt, re_pt, n_pt, lm_pt, rm_pt]
    lm_center_x = sum([pt[0] for pt in all_pts]) / 5
    lm_center_y = sum([pt[1] for pt in all_pts]) / 5
    
    # 2. face_image 중심 대비 offset
    img_center_x = w / 2
    img_center_y = h / 2
    offset_x = lm_center_x - img_center_x
    offset_y = lm_center_y - img_center_y
    
    # 3. 충분한 base margin (얼굴 크기의 50%)
    base_margin = int(max(h, w) * 0.5)
    
    # 4. 비대칭 margin으로 center 보정
    margin_top = base_margin + max(0, -int(offset_y))
    margin_bottom = base_margin + max(0, int(offset_y))
    margin_left = base_margin + max(0, -int(offset_x))
    margin_right = base_margin + max(0, int(offset_x))
    
    # 5. margin 적용
    margin_img = cv2.copyMakeBorder(face_image, 
                                    margin_top, margin_bottom, 
                                    margin_left, margin_right, 
                                    cv2.BORDER_CONSTANT, 0)
    
    # 6. 랜드마크 좌표 보정
    le_pt = (le_pt[0] + margin_left, le_pt[1] + margin_top)
    re_pt = (re_pt[0] + margin_left, re_pt[1] + margin_top)
    n_pt = (n_pt[0] + margin_left, n_pt[1] + margin_top)
    lm_pt = (lm_pt[0] + margin_left, lm_pt[1] + margin_top)
    rm_pt = (rm_pt[0] + margin_left, rm_pt[1] + margin_top)
    # ========== 개선 끝 ==========
    
    # ========== 이후 기존 로직 완전히 동일 ==========
    face_original_distance = math.sqrt(
        math.pow((le_pt[0]+re_pt[0])/2 - (lm_pt[0] + rm_pt[0])/2, 2)
        + math.pow((le_pt[1]+re_pt[1])/2 - (lm_pt[1] + rm_pt[1])/2, 2))
    face_scaler_factor = abs(48./face_original_distance)
    
    center_eye_point = ((le_pt[0]+re_pt[0])/2, (le_pt[1]+re_pt[1])/2)
    center_mouth_point = ((lm_pt[0]+rm_pt[0])/2, (lm_pt[1]+rm_pt[1])/2)
    
    eye_euclidean_distance = math.hypot(re_pt[0] - le_pt[0], re_pt[1] - le_pt[1])
    face_ratio = eye_euclidean_distance / (cropped_box[2] - cropped_box[0])
    PI = math.pi
    
    if face_ratio > 0.2:
        radian = math.atan2(re_pt[1]-le_pt[1], re_pt[0]-le_pt[0])    
        degree = (radian * 180)/PI
    else:
        radian = math.atan2(center_eye_point[1] - center_mouth_point[1], 
                           center_eye_point[0] - center_mouth_point[1])
        degree = (radian * 180)/PI + 90
    
    aligned_face = rotate_img(margin_img, degree, center_eye_point[0], center_eye_point[1])
    scale_img = cv2.resize(aligned_face, 
                          (int(float(margin_img.shape[1])*face_scaler_factor), 
                           int(float(margin_img.shape[0])*face_scaler_factor)))
    h, w, c = aligned_face.shape
    
    aligned_face_half = aligned_face[:int(n_pt[1]), :]
    half_face_ratio = n_pt[1]/h
    scale_img_half = cv2.resize(aligned_face_half, 
                                (abs(int(float(margin_img.shape[1]) * face_scaler_factor)), 
                                 abs(int(float(margin_img.shape[0])*face_scaler_factor * half_face_ratio))))
    
    new_center_x = int(float(center_eye_point[0]) * face_scaler_factor)
    new_center_y = int(float(center_eye_point[1]) * face_scaler_factor)
    
    roi_img = scale_img[
        new_center_y-40 if new_center_y-40 > 0 else 0:new_center_y+88 if new_center_y+88 > 0 else 0, 
        new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
    ].copy()
    
    roi_img_half = scale_img_half[
        new_center_y-40 if new_center_y-40 > 0 else 0:, 
        new_center_x-64 if new_center_x-64 > 0 else 0:new_center_x+64 if new_center_x+64 > 0 else 0
    ].copy()
    
    # save_debug_info(frame_idx, i, le_pt, re_pt, base_margin, margin_img, 
    #                 center_eye_point, degree, face_scaler_factor, 
    #                 new_center_x, new_center_y, scale_img,
    #                 aligned_face, roi_img
    #                 )
    
    return roi_img, roi_img_half


def xyxy_1080_to_1152(
    boxes_xyxy,                  # list[(x1,y1,x2,y2)] or list[list]
    src_shape=(1080, 1920),
    dst_shape=(1152, 1920),
    mode="scale",                # "scale" or "pad"
    pad_align="center",          # "center" | "top" | "bottom"
):
    """
    1080x1920 기준 bbox(list)를 1152x1920 기준 bbox(list)로 변환
    return: List[List[int]]
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []

    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape

    # ---- numpy 변환 ----
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]   # (1,4)

    # ---- 좌표 정렬 보장 ----
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2

    # ---- 변환 ----
    if mode == "scale":
        sx = dst_w / src_w
        sy = dst_h / src_h
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

    elif mode == "pad":
        pad_w = max(0.0, dst_w - src_w)
        pad_h = max(0.0, dst_h - src_h)

        if pad_align == "center":
            off_x = pad_w / 2.0
            off_y = pad_h / 2.0
        elif pad_align == "top":
            off_x = pad_w / 2.0
            off_y = 0.0
        elif pad_align == "bottom":
            off_x = pad_w / 2.0
            off_y = pad_h
        else:
            raise ValueError("pad_align must be center | top | bottom")

        boxes[:, [0, 2]] += off_x
        boxes[:, [1, 3]] += off_y

    else:
        raise ValueError("mode must be 'scale' or 'pad'")

    # ---- dst 영역 clip ----
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, dst_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, dst_h - 1)

    # ---- list[int]로 반환 ----
    return boxes.round().astype(int).tolist()     

def save_debug_info(frame_idx, i, le_pt, re_pt, margin, margin_img, center_eye_point, 
                    degree, face_scaler_factor, new_center_x, new_center_y, scale_img,
                    aligned_face, roi_img, base_path="./yolov8_face_advise"):
    """
    얼굴 정렬 디버그 정보를 저장하는 함수
    
    Args:
        frame_idx: 프레임 인덱스
        i: 얼굴 인덱스
        le_pt, re_pt: 왼쪽/오른쪽 눈 좌표
        margin: 마진 크기
        margin_img: 마진 추가된 이미지
        center_eye_point: 눈 중심점
        degree: 회전 각도
        face_scaler_factor: 스케일 팩터
        new_center_x, new_center_y: 새로운 중심 좌표
        scale_img: 스케일된 이미지
        aligned_face: 정렬된 얼굴
        roi_img: ROI 이미지
        base_path: 저장 기본 경로
    """
    import os
    
    # 디렉토리 생성
    save_dir = f"{base_path}/frame_{frame_idx}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 텍스트 파일로 저장
    txt_path = f"{save_dir}/debug_{i}_info.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"le_pt local: {le_pt}\n")
        f.write(f"re_pt local: {re_pt}\n")
        f.write(f"margin: {margin}\n")
        f.write(f"margin_img shape: {margin_img.shape}\n")
        f.write(f"center_eye_point: {center_eye_point}\n")
        f.write(f"degree: {degree}\n")
        f.write(f"scale factor: {face_scaler_factor}\n")
        f.write(f"new_center: ({new_center_x}, {new_center_y})\n")
        f.write(f"scale_img shape: {scale_img.shape}\n")
    
    # 이미지 저장
    cv2.imwrite(f"{save_dir}/debug_{i}_aligned.png", aligned_face)
    cv2.imwrite(f"{save_dir}/debug_{i}_scale.png", scale_img)
    cv2.imwrite(f"{save_dir}/debug_{i}_roi.png", roi_img)