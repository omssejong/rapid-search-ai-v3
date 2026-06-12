"""
배치 엔진 기반 타겟 feature 로더 — lib/core 참조 없이 완전 독립 동작.

Target(target.py)과 동일 기능을 pkg/batch_engine.py의 TRTBatchEngine +
RetinaFace/face_alignment 유틸리티와 결합하여 구현.

사용법:
    from pkg.batch_target import BatchTarget

    target = BatchTarget(target_type="person", gpu_idx=0, cfg_path="prod_cfg.yaml")
    target_feature_dict = target.set_target("/path/to/target.json")
"""

import json
import logging
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from shapely.geometry import Polygon
from typing import Dict, List, Tuple

from .util.engine import TRTBatchEngine
#from pkg.batch_engine import TRTBatchEngine
# (
#     TRTBatchEngine,
#     RETINAFACE_CFG,
#     _retinaface_priorbox,
#     _retinaface_decode,
#     _retinaface_decode_landm,
#     _py_cpu_nms,
#     _face_alignment,
# )


logger = logging.getLogger(__name__)


class BatchTarget:
    """TRTBatchEngine 기반 타겟 feature 추출기.

    lib/core/target.py의 Target 클래스와 동일 기능을 수행하되,
    pkg/batch_engine.py의 TRTBatchEngine을 사용하여 모델 추론.
    """

    SUPPORTED_TYPES = ("person", "attribute", "carplate", "face", "area", "exportvideo")

    def __init__(
        self,
        target_type: str,
        gpu_idx: int,
        cfg: dict = None,
        log: logging.Logger = None,
    ):
        if target_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"not exist {target_type} type!!!")

        self.target_type = target_type
        self.logger = log or logger
        self.config = cfg

        torch.cuda.set_device(gpu_idx)
        self.device = torch.device(f"cuda:{gpu_idx}")

        if target_type == "person":
            engine_path = self.config["model_path"]["person_analysis"]
            try:
                self.analysis_engine = TRTBatchEngine(engine_path, device=self.device)
            except Exception as e:
                self.logger.error(f"{target_type} : Engine Load Error => {e}")
                raise
            self.analysis_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).view(1, 3, 1, 1)
            self.analysis_std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)

        elif target_type == "face":
            
            from .util.engine import AnalysisEngine_v4
            from .util.retinaface import RetinaFace
            
            ############ gpu 설정 먼저 하고 engine load 하는 파일들 import 해야 gpu index 설정 오류 없이 작동함
            self.image_size = tuple(self.config["image_size"]["face_frame"])
            
            self.half_transform = T.Compose([T.Grayscale(), T.CenterCrop((64, 128))])
            self.full_transform = T.Compose([T.CenterCrop(128)])
            
            self.retina = RetinaFace(model_input_size=self.image_size, device=self.device)
            try:
                #self.detect_model = DetectEngine(config.model_path.face_detect, device=self.torch_device)
                self.detect_model = AnalysisEngine_v4(self.config["model_path"]["face_detect"], fp16=False, output_count=3, device=self.device)
                self.analysis_half_model = AnalysisEngine_v4(self.config["model_path"]["face_half_analysis"], fp16=False, output_count=1, device=self.device)
                self.analysis_full_model = AnalysisEngine_v4(self.config["model_path"]["face_full_analysis"], fp16=False, output_count=1, device=self.device)
            except Exception as e:
                self.logger.error(f"{target_type} : Engine Load Error => {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 공통 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def set_target(self, json_path: str) -> dict:
        """JSON 파일에서 타겟 feature 추출. Target.set_target() 호환."""
        target_feature_dict = None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                target_info = json.load(f)

            dispatch = {
                "person": self._set_person_target,
                "attribute": self._set_attribution_target,
                "carplate": self._set_carplate_target,
                "face": self._set_face_target,
                "area": self._set_area_target,
                "exportvideo": self._set_exportvideo_target,
            }
            target_feature_dict = dispatch[self.target_type](target_info)
            self._empty_target_check(target_feature_dict)

        except Exception as e:
            self.logger.exception(f"{self.target_type} : Set Target Error => {e}")

        return target_feature_dict

    def _empty_target_check(self, target_feature_dict):
        if self.target_type != "face":
            if target_feature_dict is None or len(target_feature_dict) == 0:
                self.logger.warning(f"{self.target_type}'s target_feature is empty!")
        else:
            if (target_feature_dict["full"] is None or len(target_feature_dict["full"]) == 0) and \
               (target_feature_dict["half"] is None or len(target_feature_dict["half"]) == 0):
                self.logger.warning(f"{self.target_type}'s target_feature is empty!")

    # ─────────────────────────────────────────────────────────────────────────
    # person — PLR-OSNet feature 추출 (TRTBatchEngine)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_person_target(self, target_info: Dict) -> Dict:
        target_feature_dict = {}

        for obj in target_info["objectList"]:
            target_id = str(obj["id"])

            img_bgr = cv2.imread(obj["imagePath"])
            assert img_bgr is not None, "이미지 로드 실패"

            # BGR → RGB → CHW → float [0,1] → GPU
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
            img_tensor = img_tensor.float().div_(255.0)
            img_tensor = img_tensor.to(self.device, non_blocking=True)

            # Resize (128, 64) + Normalize
            img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
            img_tensor = F.interpolate(img_tensor, size=(128, 64), mode="bilinear", align_corners=False)
            img_tensor = (img_tensor - self.analysis_mean) / self.analysis_std
            img_tensor = img_tensor.contiguous()

            out = self.analysis_engine.infer(img_tensor)[0]
            out = out.cpu().detach().float().numpy().squeeze()

            target_feature_dict[target_id] = out

        return target_feature_dict

    # ─────────────────────────────────────────────────────────────────────────
    # attribute — JSON → attr ID 리스트 (모델 추론 불필요)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_attribution_target(self, target_info: Dict) -> Dict:
        from .util.target_util import get_attr_id_from_json

        target_feature_dict = {}
        for obj in target_info["objectList"]:
            target_id = str(obj["id"])
            target_attr = obj["attrParam"]
            target_feature_dict[target_id] = get_attr_id_from_json(target_attr)
        return target_feature_dict

    # ─────────────────────────────────────────────────────────────────────────
    # carplate — OCR 인덱스 + 텍스트
    # ─────────────────────────────────────────────────────────────────────────

    def _set_carplate_target(self, target_info: Dict) -> Dict:
        target_feature_dict = {}
        for obj in target_info["objectList"]:
            target_id = str(obj["id"])
            target_ocr_idx = [i for i, item in enumerate(list(obj["ocr"])) if item != "*"]
            target_ocr = list(obj["ocr"])
            target_feature_dict[target_id] = (target_ocr_idx, target_ocr)
        return target_feature_dict

    def _set_face_target(self, target_info: Dict) -> Dict:
        from .util.face_util_new import face_alignment_renew
        target_feature_dict = {
                                    "half": [],
                                    "full": []
                                    }
        
        for idx, obj in enumerate(target_info['objectList']):
            
            target_id = str(obj["id"])
            img_bgr = cv2.imread(obj["imagePath"])
            mask = obj["mask"] # 마스크 착용 유/무
            
            img_new, pad_left, pad_top, ratio = self.retina.letterbox(img_bgr)
            
            img_tensor = torch.from_numpy(img_new).to(self.device, non_blocking=True)
            img_tensor = img_tensor.permute(2, 0, 1).float()
            img_tensor.sub_(torch.tensor([104, 117, 123], device=self.device).view(3,1,1))
            img_tensor = img_tensor.unsqueeze(0).contiguous()
            
            scale = torch.Tensor([img_new.shape[1], img_new.shape[0], img_new.shape[1], img_new.shape[0]])
            scale = scale.to(self.device)
            
            # img_tensor = img_tensor.to(self.device, non_blocking=True)  # (3,h,w) on GPU
            # img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
            
            # face alignment 때문에 lanmark 좌표가 필요함에 따라 crop image face detect 다시 수행!
            output = self.detect_model(img_tensor) # engine model
            loc, conf, landms = output[0][0], output[1][0], output[2][0]
            
            dets = self.retina.retina_postprocess(img_tensor, loc, conf, landms, scale, 1)
            list_OMS_Face = self.retina.point_post_process(dets, ratio, pad_left, pad_top)
            self.logger.info(list_OMS_Face)
            
            if list_OMS_Face is not None and len(list_OMS_Face) > 0:
                
                for face in list_OMS_Face:
                    face_boxing = [pt if pt > 0 else 0 for pt in face.rt]
                    cropped_face = img_bgr[face_boxing[1]:face_boxing[3], face_boxing[0]:face_boxing[2]]
                
                    landmarks = [face.ptLE, face.ptRE, face.ptLM, face.ptRM, face.ptN]
                    roi_img, roi_img_half = face_alignment_renew(cropped_face, landmarks, face_boxing)
                    
                    if mask:
                        img_half_tensor = torch.from_numpy(roi_img_half).permute(2, 0, 1).contiguous()  # (3,h,w), uint8
                        img_half_tensor = img_half_tensor.float().div_(255.0)
                        img_half_tensor = img_half_tensor.to(self.device, non_blocking=True)
                        img_half_tensor = self.half_transform(img_half_tensor)
                        img_half_tensor = img_half_tensor.unsqueeze(0)  # (1,3,H,W)
                        out = self.analysis_half_model(img_half_tensor)[0]
                        out = out[0].cpu().detach().numpy()
                        target_feature_dict["half"].append({target_id: out.copy()})
                    else:
                        img_tensor = torch.from_numpy(roi_img).permute(2, 0, 1).contiguous()  # (3,h,w), uint8
                        img_tensor = img_tensor.float().div_(255.0)
                        img_tensor = img_tensor.to(self.device, non_blocking=True)  # (3,h,w) on GPU
                        img_tensor = self.full_transform(img_tensor)
                        img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
                        out = self.analysis_full_model(img_tensor)[0]
                        out = out[0].cpu().detach().numpy()
                        target_feature_dict["full"].append({target_id: out.copy()})
        return target_feature_dict      
    
    
    # ─────────────────────────────────────────────────────────────────────────
    # area
    # ─────────────────────────────────────────────────────────────────────────

    def _set_area_target(self, target_info: Dict) -> Dict:
        target_feature_dict = {}
        area_list = []
        for obj in target_info["objectList"]:
            for roi_point in obj["point"]:
                area_list.append(Polygon(roi_point))
            target_feature_dict["points"] = area_list
            target_feature_dict["classes"] = obj["classes"]
        return target_feature_dict

    # ─────────────────────────────────────────────────────────────────────────
    # exportvideo
    # ─────────────────────────────────────────────────────────────────────────

    def _set_exportvideo_target(self, target_info: Dict) -> Dict:
        return {"masking": target_info["options"]["masking"]}

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 유틸리티
    # ─────────────────────────────────────────────────────────────────────────

    # @staticmethod
    # def _center_crop(tensor: torch.Tensor, output_size: tuple) -> torch.Tensor:
    #     """(C,H,W) 텐서 center crop. output_size=(crop_h, crop_w)."""
    #     _, h, w = tensor.shape
    #     crop_h, crop_w = output_size
    #     top = max((h - crop_h) // 2, 0)
    #     left = max((w - crop_w) // 2, 0)
    #     return tensor[:, top:top + crop_h, left:left + crop_w]

    # def _retinaface_postproc(self, loc, conf, landms, pad_info, orig_shape):
    #     """단일 프레임 RetinaFace 후처리 → list of face dicts.

    #     FaceBatchInference._retinaface_postproc()과 동일 로직.
    #     """
    #     pad_left, pad_top, ratio = pad_info
    #     orig_h, orig_w = orig_shape
    #     target_h, target_w = self.image_size
    #     variance = RETINAFACE_CFG["variance"]

    #     priors_cpu = self.priors.cpu()
    #     boxes = _retinaface_decode(loc.squeeze(0).cpu(), priors_cpu, variance)
    #     landm = _retinaface_decode_landm(landms.squeeze(0).cpu(), priors_cpu, variance)

    #     scale = torch.tensor([target_w, target_h, target_w, target_h], dtype=torch.float32)
    #     boxes = boxes * scale

    #     scores = conf.squeeze(0).cpu().numpy()[:, 1]

    #     scale_landm = torch.tensor([target_w, target_h] * 5, dtype=torch.float32)
    #     landm = landm * scale_landm

    #     boxes = boxes.numpy()
    #     landm = landm.numpy()

    #     # confidence filter
    #     inds = np.where(scores > 0.02)[0]
    #     boxes, landm, scores = boxes[inds], landm[inds], scores[inds]

    #     # top-k before NMS
    #     order = scores.argsort()[::-1][:5000]
    #     boxes, landm, scores = boxes[order], landm[order], scores[order]

    #     # NMS
    #     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    #     keep = _py_cpu_nms(dets, 0.4)
    #     dets, landm = dets[keep][:750], landm[keep][:750]

    #     # vis_thresh filter + inverse letterbox
    #     faces = []
    #     for i in range(len(dets)):
    #         if dets[i, 4] < self.vis_thresh:
    #             continue

    #         x1, y1, x2, y2 = dets[i, :4]
    #         face_w = x2 - x1
    #         if face_w < 28:
    #             continue

    #         # inverse letterbox
    #         bx1 = (x1 - pad_left) / ratio
    #         by1 = (y1 - pad_top) / ratio
    #         bx2 = (x2 - pad_left) / ratio
    #         by2 = (y2 - pad_top) / ratio

    #         # bbox를 정사각형으로 확장 (crop_ratio=1.1)
    #         cw = bx2 - bx1
    #         ch = by2 - by1
    #         face_size = max(cw, ch) * 1.1
    #         cx = (bx1 + bx2) / 2
    #         cy = (by1 + by2) / 2
    #         bx1 = max(0, cx - face_size / 2)
    #         by1 = max(0, cy - face_size / 2)
    #         bx2 = min(orig_w, cx + face_size / 2)
    #         by2 = min(orig_h, cy + face_size / 2)

    #         bbox = [int(bx1), int(by1), int(bx2), int(by2)]

    #         # landmarks inverse letterbox
    #         lm = landm[i].reshape(5, 2)
    #         lm_orig = []
    #         for pt in lm:
    #             px = (pt[0] - pad_left) / ratio
    #             py = (pt[1] - pad_top) / ratio
    #             lm_orig.append([px, py])

    #         faces.append({
    #             "bbox": bbox,
    #             "landmarks": lm_orig,
    #             "confidence": float(dets[i, 4]),
    #         })

    #     return faces
