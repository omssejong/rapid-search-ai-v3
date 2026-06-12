import cv2
from .fd_utils import *

class OMSFace:
    def __init__(self):
        self.rt = [-1, -1, -1, -1]
        self.sID = ""
        self.fvScore = -1.
        self.ptLE = [-1, -1]
        self.ptRE = [-1, -1]
        self.ptN = [-1, -1]
        self.ptLM = [-1, -1]
        self.ptRM = [-1, -1]


class RetinaFace:
    def __init__(self,
                 model_input_size = (1080, 1920),
                 iou_threshold = 0.4,
                 conf_threshold = 0.9,
                 device = "cuda:0"
                 ):
        
        
        self.model_input_height, self.model_input_width = model_input_size
        self.device =device # cuda:index
        
        self.low_conf_threshold = 0.02
        self.nms_threshold = iou_threshold
        self.vis_thresh = conf_threshold
        
        self.first_top_k = 5000
        self.second_top_k = 750
        
        self.crop_ratio = 1.1
        self.cfg_re50 = cfg_re50
        
        self.priors = self._init_priors(cfg_re50, input_size=model_input_size, device=device)
    
    def _init_priors(self, cfg, input_size, device):
        priorbox = PriorBox(cfg, image_size=input_size)
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.to(device)
        return priors
    
    # def letterbox(self, img_raw):
        
    #     ratio = 1
        
    #     ori_h, ori_w, _ = img_raw.shape
    #     ratio = min(self.model_input_width / ori_w, self.model_input_height / ori_h)
        
    #     new_w = int(ori_w * ratio)
    #     new_h = int(ori_h * ratio)
        
        
    #     img_resize = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
    #     if ratio > 1:
    #         ratio = 1.0 / ratio
    #     #     pad_left = 0
    #     #     pad_top = 0
            
    #     # else:
    #         # padding
    #     pad_w = self.model_input_width - new_w
    #     pad_h = self.model_input_height - new_h

    #     pad_left   = pad_w // 2
    #     pad_right  = pad_w - pad_left
    #     pad_top    = pad_h // 2
    #     pad_bottom = pad_h - pad_top

    #     img_new = cv2.copyMakeBorder(
    #         img_resize,
    #         pad_top, pad_bottom, pad_left, pad_right,
    #         borderType=cv2.BORDER_CONSTANT,
    #         value=(114, 114, 114)
    #     )

    #     return img_new, pad_left, pad_top, ratio
    
    def letterbox(self, img_raw):
    
        ori_h, ori_w, _ = img_raw.shape
        
        # 캔버스보다 큰 경우에만 리사이즈 (비율 유지)
        if ori_w > self.model_input_width or ori_h > self.model_input_height:
            ratio = min(self.model_input_width / ori_w, self.model_input_height / ori_h)
            new_w = int(ori_w * ratio)
            new_h = int(ori_h * ratio)
            img_raw = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            ratio = 1.0
            new_w, new_h = ori_w, ori_h

        pad_w = self.model_input_width - new_w
        pad_h = self.model_input_height - new_h

        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top

        img_new = cv2.copyMakeBorder(
            img_raw,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        return img_new, pad_left, pad_top, ratio
    
    def retina_postprocess(self, img_tensor, loc, conf, landms, scale, resize):
        _, _, im_height, im_width = img_tensor.shape
        #priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
        #priors = priorbox.forward()
        #priors = priors.to(self.device)
        prior_data = self.priors.data  # data: tensor
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_re50['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg_re50['variance'])
        scale1 = torch.Tensor([img_tensor.shape[3], img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[2],img_tensor.shape[3], img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[2],img_tensor.shape[3], img_tensor.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > self.low_conf_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.first_top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.second_top_k, :]
        landms = landms[:self.second_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def point_post_process(self, dets, ratio, pad_left, pad_top):
        
        list_OMSFace = list()
        
        faceIdx = 0
        for b in dets:
            
            #if pixel under 28, remove face detection result
            if (b[4] < self.vis_thresh or int(abs(b[0] - b[2])) < 28):
                continue
            ef = OMSFace()
            b = list(map(int, b))
            list_OMSFace.append(ef)
            face_center_point_x = (b[0]+b[2])/2
            face_center_point_y = (b[1]+b[3])/2
            face_w = abs(b[0]-b[2])
            face_h = abs(b[1]-b[3])

            crop_ratio = 1.1      #! default 1.25 -> 1
            tmp_lenth = face_w if face_w >= face_h else face_h
            b[0] = int(face_center_point_x - tmp_lenth/2*crop_ratio) if int(face_center_point_x - tmp_lenth/2*crop_ratio) > 0 else 0
            b[1] = int(face_center_point_y - tmp_lenth/2*crop_ratio) if int(face_center_point_y - tmp_lenth/2*crop_ratio) > 0 else 0
            b[2] = int(face_center_point_x + tmp_lenth/2*crop_ratio) if int(face_center_point_x + tmp_lenth/2*crop_ratio) < 1920 else 1920
            b[3] = int(face_center_point_y + tmp_lenth/2*crop_ratio) if int(face_center_point_y + tmp_lenth/2*crop_ratio) < 1080 else 1080
            
            ef.rt = [int(ratio * (b[0] - pad_left)), int(ratio * (b[1] - pad_top)), int(ratio * (b[2] - pad_left)), int(ratio * (b[3] - pad_top))]
            
            list_OMSFace[faceIdx].ptLE = [int(ratio * (b[5] - pad_left)), int(ratio * (b[6] - pad_top))]
            list_OMSFace[faceIdx].ptRE = [int(ratio * (b[7] - pad_left)), int(ratio * (b[8] - pad_top))]
            list_OMSFace[faceIdx].ptN = [int(ratio * (b[9] - pad_left)), int(ratio * (b[10] - pad_top))]
            list_OMSFace[faceIdx].ptLM = [int(ratio * (b[11] - pad_left)), int(ratio * (b[12] - pad_top))]
            list_OMSFace[faceIdx].ptRM = [int(ratio * (b[13] - pad_left)), int(ratio * (b[14] - pad_top))]
            faceIdx += 1

        list_OMSFace = [a for a in list_OMSFace if len(list_OMSFace) > 0]
        
        return list_OMSFace
            
        
    
    # def forward(self, img_raw, model):
    #     img_new, pad_left, pad_top, ratio = self.retina_letterbox(img_raw)
        
    #     img = np.float32(img_new)
    #     img -= (104, 117, 123)
    #     img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,h,w), uint8
        
    #     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #     scale = scale.to("cuda:0")
        
    #     # === GPU 이동 ===
    #     img_tensor = img_tensor.to("cuda:0", non_blocking=True)  # (3,h,w) on GPU
    #     img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
        
    #     output = model(img_tensor) # engine model
    #     loc, conf, landms = output[0][0], output[1][0], output[2][0]
        
    #     dets = self.retina_postprocess(img_tensor, loc, conf, scale, 1)
    #     list_OMS_Face = self.point_post_process(dets, ratio, pad_left, pad_top)