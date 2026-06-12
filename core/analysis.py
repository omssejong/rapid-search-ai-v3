import time
from typing import List
import numpy as np
import torch
import logging
import yaml
import cv2
import torch.nn.functional as F
import torchvision.transforms as T
from shapely.geometry import Polygon

from .util.engine import TRTBatchEngine, AnalysisEngine_ORT, AnalysisEngine_v4, SegDetectEngineTRT_v2
from .util.yolo_util import *
from .util.common import to_dotdict
from .util.scoring import *
from .util.face_util_new import face_alignment_renew

class BatchInference:
    """4프레임 배치 추론 파이프라인.

    trt_batch_test.py / plr_osnet_trt.py 패턴 기반. lib/core 참조 없이 완전 독립 동작.

    파이프라인:
        1. N프레임 letterbox → (N,3,H,W) 배치 텐서
        2. TRT detect 배치 추론 → NMS per frame
        3. 검출된 crop 전처리 → 배치 텐서
        4. TRT analysis 배치 추론 (PLR-OSNet feature 추출)
        5. feature 비교 (코사인 유사도) → 매칭 결과 반환
    """

    def __init__(
        self,
        target_type: str, # 분석 타입
        target_feature: dict = None, # 타겟 데이터
        gpu_idx: int = 0,
        batch_size: int = 4, # Detect 모델 배치 사이즈
        analysis_batch_size: int = None, # 분석 모델 배치 사이즈
        cfg: dict = {}, # config 데이터
        logger: logging.Logger = None,
    ):
        
        # 로거 설정
        self.logger = logger or logging.getLogger(__name__)
        
        self.cfg = cfg
        #type_cfg = getattr(cfg["type"], target_type, None)
        type_cfg = cfg["type"].get(str(target_type))
        if type_cfg is None:
            self.logger.error(f"지정된 타입이 아닙니다. 현재 타겟 타입 : {target_type}")
        
        
        self.target_type = target_type
        
        '''
            분석 타입별 모델 input size 와 모델 경로 설정
        '''
        if target_type == "person" or target_type == "attribute":
            self.image_size = cfg["image_size"]["person_frame"]
            detect_engine_path = cfg["model_path"]["person_detect"]
            
            if target_type == "person":
                analysis_engine_path = cfg["model_path"]["person_analysis"]
            elif target_type == "attribute":
                analysis_engine_path = cfg["model_path"]["attribute_analysis"]
                
        else:
            self.image_size = cfg["image_size"]["plate_frame"]
            detect_engine_path = cfg["model_path"]["plate_detect"]
            analysis_engine_path = cfg["model_path"]["plate_ocr_analysis"]
            
            
        self.target_feature = target_feature
        #self.origin_size = origin_size  # (H, W) 원본 프레임 크기 — bbox 좌표 역변환용
        self.threshold = getattr(type_cfg, "conf_thresh", 0.5) if type_cfg else 0.5
        self.iou = getattr(type_cfg, "iou_thresh",  0.5) if type_cfg else 0.5
        self.max_det = cfg["yolo"]["max_det"]
        self.stride = cfg["yolo"]["stride"]
        self.batch_size = batch_size
        self.analysis_batch_size = getattr(type_cfg, "batch_size", 8) if type_cfg else 8
        
        torch.cuda.set_device(gpu_idx)
        self.device = torch.device(f"cuda:{gpu_idx}")

        self.detect_class_filter = type_cfg["cls_list"]

        
        self.detect_engine = TRTBatchEngine(detect_engine_path, device=self.device)

        self.logger.info(f"[BatchInference] analysis engine: {analysis_engine_path}")
        if target_type == "person":
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            if any(x in gpu_name for x in ["5080", "5090"]):
                self.analysis_engine = AnalysisEngine_ORT(cfg["model_path"]["person_analysis_onnx"], fp16=False, output_count=1, device=self.device)
            else:
                self.analysis_engine = TRTBatchEngine(analysis_engine_path, device=self.device)
            self.logger.info("analysis 모델 load 완료")        
        else:
            self.analysis_engine = TRTBatchEngine(analysis_engine_path, device=self.device)

        # 배치 크기 안전장치: 엔진 max_batch 초과 시 자동 클램핑
        detect_max = self.detect_engine.max_batch
        analysis_max = self.analysis_engine.max_batch
        if self.batch_size > detect_max:
            self.logger.warning(
                f"[BatchInference] detect batch_size({self.batch_size}) > "
                f"engine max_batch({detect_max}), 분할 추론 적용"
            )
        if self.analysis_batch_size > analysis_max:
            self.logger.warning(
                f"[BatchInference] analysis_batch_size({self.analysis_batch_size}) > "
                f"engine max_batch({analysis_max}), "
                f"클램핑: {self.analysis_batch_size} → {analysis_max}"
            )
            self.analysis_batch_size = analysis_max

        # 타겟 feature → numpy 정규화 캐싱 (코사인 유사도 고속화, person/attribute 전용)
        # carplate는 target_feature를 그대로 사용 (OCR 매칭용 튜플)
        self._target_norms = {}
        if target_feature and target_type in ("person",):
            for key, feat in target_feature.items():
                feat_np = np.asarray(feat, dtype=np.float32).flatten()
                norm = np.linalg.norm(feat_np)
                self._target_norms[key] = feat_np / norm if norm > 0 else feat_np

        # 분석 모델 전처리 파라미터
        self.analysis_input_size = tuple(type_cfg["crop_resize"])
        if target_type == "person":
            
            # 전신 분석에 필요한 상수값 #
            self.min_crop_size = tuple(cfg["type"]["person"]["min_crop_size"])
            self.analysis_input_size = tuple(cfg["type"]["person"]["crop_resize"])
            self.match_score_thresh = float(cfg["type"]["person"]["match_score_thresh"])
            # 전신 분석에 필요한 상수값 #
            
            self.analysis_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).view(1, 3, 1, 1)
            self.analysis_std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)
        elif target_type == "attribute":
            
            # 속성 분석에 필요한 상수값 #
            self.min_crop_size = tuple(cfg["type"]["attribute"]["min_crop_size"]) # 최소 크기 필터 임계값
            self.analysis_input_size = tuple(cfg["type"]["attribute"]["crop_resize"]) # 분석 모델 입력 사이즈
            self.match_score_thresh = float(cfg["type"]["attribute"]["match_score_thresh"]) # 분석 모델 prob 임계값
            self.match_color_filter_thresh = float(cfg["type"]["attribute"]["match_color_filter_thresh"]) # 컬러 필터 임계값
            self.match_attr_avg_score_thresh = float(cfg["type"]["attribute"]["match_attr_avg_score_thresh"]) # 객체 후보 임계값
            # 속성 분석에 필요한 상수값 #
            
            # 전처리에 필요한 평균, 표준편차 #
            self.analysis_mean = torch.tensor(
                [0.46190931, 0.43684858, 0.43709673], device=self.device
            ).view(1, 3, 1, 1)
            self.analysis_std = torch.tensor(
                [0.26607546, 0.25202867, 0.25236069], device=self.device
            ).view(1, 3, 1, 1)
            # 전처리에 필요한 평균, 표준편차 #
            
            # 속성 칼라 로직에 필요한 변수 #
            self.outer_color_list = list(range(31, 44))
            self.bottom_color_list = list(range(74, 87))
            self.outer_color_map  = {v: i for i, v in enumerate(self.outer_color_list)}
            self.bottom_color_map = {v: i for i, v in enumerate(self.bottom_color_list)}
            # 속성 칼라 로직에 필요한 변수 #
            
        elif target_type == "carplate":
            
            # ocr 분석에 필요한 상수값 #
            self.min_crop_size = tuple(cfg["type"]["carplate"]["min_crop_size"])
            self.analysis_input_size = tuple(cfg["type"]["carplate"]["crop_resize"])
            # ocr 분석에 필요한 상수값 #
            
            # grayscale 1ch: mean=0.5, std=0.5
            self.analysis_mean = torch.tensor(
                [0.5], device=self.device
            ).view(1, 1, 1, 1)
            self.analysis_std = torch.tensor(
                [0.5], device=self.device
            ).view(1, 1, 1, 1)
            # CTC 디코더 초기화
            from .util.car_plate_util import CTCLabelConverter
            
            char_string = cfg["etc_info"]["carLPLModelNameString"]
            self.converter = CTCLabelConverter(char_string, self.device)
        else:
            self.analysis_input_size = None
            self.analysis_mean = None
            self.analysis_std = None

        self.logger.info(
            f"[BatchInference] ready: type={target_type} "
            f"image_size={self.image_size} detect_batch={batch_size} "
            f"analysis_batch={self.analysis_batch_size} "
            f"targets={len(self._target_norms)}"
        )

    def _preproc_frames(self, frames: list[np.ndarray]) -> tuple[torch.Tensor, list[tuple]]:
        """N개 프레임 letterbox 전처리 → 배치 텐서.

        동일 크기 프레임이면 np.stack → GPU 전송 1회 + letterbox 배치 1회로 최적화.
        """
        orig_shapes = [(f.shape[0], f.shape[1]) for f in frames]

        # 모든 프레임이 동일 크기인지 확인
        first_shape = orig_shapes[0]
        all_same = all(s == first_shape for s in orig_shapes)

        if all_same:
            # 동일 크기 → np.stack 후 GPU 전송 1회
            stacked = np.stack(frames, axis=0)  # (N, H, W, 3)
            batch_gpu = torch.from_numpy(stacked).to(self.device, non_blocking=True)
            # (N,H,W,3) → (N,3,H,W), fp16, 0~1
            batch_gpu = batch_gpu.permute(0, 3, 1, 2).half().div_(255.0)
            # letterbox 배치 처리 (BCHW 지원)
            batch_gpu = letterbox_torch(
                img=batch_gpu, new_shape=self.image_size,
                stride=self.stride, auto=False,
            )[0]
            # BGR → RGB (채널 스왑)
            batch_tensor = batch_gpu[:, [2, 1, 0], ...]
        else:
            # 다른 크기 → 프레임별 개별 처리
            preprocessed = []
            for frame_bgr in frames:
                im = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
                im = im.permute(2, 0, 1).half().div_(255.0)
                im = letterbox_torch(
                    img=im, new_shape=self.image_size,
                    stride=self.stride, auto=False,
                )[0]
                im = im[[2, 1, 0], ...]
                preprocessed.append(im)
            batch_tensor = torch.stack(preprocessed, dim=0)

        return batch_tensor, orig_shapes

    def _batch_detect(self, batch_tensor: torch.Tensor, orig_shapes: list[tuple]) -> list:
        """배치 detect 추론 + fp32 NMS.

        detect 엔진 max_batch 초과 시 자동 분할 처리.
        부분 배치(N < batch_size)는 batch_size로 패딩하여 추론 안정성 보장.
        """
        n = batch_tensor.shape[0]
        in_h, in_w = batch_tensor.shape[-2:]
        max_b = min(self.batch_size, self.detect_engine.max_batch)

        # max_batch 이하 → 단일 추론
        if n <= max_b:
            pred = self._detect_infer_padded(batch_tensor, max_b)
        else:
            # max_batch 초과 → 분할 추론 후 결합
            pred_chunks = []
            for start in range(0, n, max_b):
                chunk = batch_tensor[start:start + max_b]
                chunk_pred = self._detect_infer_padded(chunk, max_b)
                if chunk_pred is not None:
                    pred_chunks.append(chunk_pred)
            if pred_chunks:
                pred = torch.cat(pred_chunks, dim=0)
                # self.logger.info(
                #     f"[BatchInference] detect split: total={n} max_batch={max_b} "
                #     f"chunks={len(pred_chunks)}"
                # )
            else:
                pred = None

        if pred is None:
            return [None] * n

        # fp16 → fp32 (bbox 좌표 정밀도 보장)
        if pred.dtype == torch.float16:
            pred = pred.float()

        preds = non_max_suppression(
            pred, self.threshold, self.iou,
            self.detect_class_filter, False,
            max_det=self.max_det,
        )

        detections = []
        for i in range(n):
            if i >= len(preds):
                detections.append(None)
                continue
            det = preds[i]
            if det is None or len(det) == 0:
                detections.append(None)
            else:
                det = det.float()
                #orig_h, orig_w = orig_shapes[i]
                #raw_boxes = det[:, :4].clone()
                det[:, :4] = scale_boxes(
                    (in_h, in_w), det[:, :4], (in_h, in_w)
                ).round()
                # self.logger.debug(
                #     f"[detect_bbox] frame={i} infer_size=({in_h},{in_w}) "
                #     f"resized=({orig_h},{orig_w}) "
                #     f"raw={raw_boxes[0].tolist()} → scaled={det[0, :4].tolist()}"
                # )
                detections.append(det)

        return detections

    def _detect_infer_padded(self, batch_tensor: torch.Tensor, pad_to: int) -> torch.Tensor:
        """detect 추론: 부분 배치 패딩 → 추론 → 실제 프레임만 반환."""
        n = batch_tensor.shape[0]

        if n < pad_to:
            pad_tensor = torch.zeros(
                (pad_to - n, *batch_tensor.shape[1:]),
                dtype=batch_tensor.dtype, device=batch_tensor.device,
            )
            batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)

        pred_outputs = self.detect_engine.infer(batch_tensor)
        pred = pred_outputs[0] if pred_outputs else None
        # if pred is None:
        #     return None

        # 패딩 제거: 실제 프레임 수만 반환
        return pred[:n] if pred is not None else None


    def _extract_crops(self, frame_bgr: np.ndarray, det: torch.Tensor) -> tuple[list, list, list, list]:
        """단일 프레임에서 detection 기반 crop 추출.

        detection bbox는 letterbox 좌표계(image_size, 예: 480x640).
        frame_bgr은 리사이즈된 프레임(예: 360x640).
        crop 시 letterbox 패딩 오프셋을 보정하고,
        valid_boxes에는 letterbox 좌표를 유지 (후속 letterbox_xyxy_to_original용).

        Returns:
            crops: list[np.ndarray] — RGB HWC uint8
            boxes: list[[x1,y1,x2,y2]] — letterbox 좌표 (원본 역변환용)
            confs: list[float] — detection confidence
            cls_ids: list[int] — class id
        """
        if det is None or len(det) == 0:
            return [], [], [], []

        H, W = frame_bgr.shape[:2]

        # letterbox 패딩 오프셋 계산: letterbox 좌표 → 프레임 좌표 변환용
        # _preproc_frames에서 auto=False, scaleup=True letterbox 적용 시 발생하는 패딩
        lb_h, lb_w = self.image_size  # letterbox 크기 (예: 480, 640)
        r = min(lb_h / H, lb_w / W)
        # scaleup=True: r > 1.0 허용 (_letterbox_torch 기본값과 일치)
        pad_w = (lb_w - int(round(W * r))) / 2.0
        pad_h = (lb_h - int(round(H * r))) / 2.0

        # crop용 좌표: letterbox 좌표에서 패딩을 빼고 스케일 역변환
        crop_x1 = ((det[:, 0] - pad_w) / r).clamp(0, W - 1).int()
        crop_y1 = ((det[:, 1] - pad_h) / r).clamp(0, H - 1).int()
        crop_x2 = ((det[:, 2] - pad_w) / r).clamp(1, W).int()
        crop_y2 = ((det[:, 3] - pad_h) / r).clamp(1, H).int()
        conf = det[:, 4].float()
        cls_id = det[:, 5].int()

        valid = (crop_x2 > crop_x1) & (crop_y2 > crop_y1)
        crop_x1 = crop_x1[valid]
        crop_y1 = crop_y1[valid]
        crop_x2 = crop_x2[valid]
        crop_y2 = crop_y2[valid]
        conf = conf[valid]
        cls_id = cls_id[valid]

        # letterbox 좌표 유지 (후속 letterbox_xyxy_to_original 역변환용)
        lb_x1 = det[:, 0][valid].int()
        lb_y1 = det[:, 1][valid].int()
        lb_x2 = det[:, 2][valid].int()
        lb_y2 = det[:, 3][valid].int()

        if len(crop_x1) == 0:
            return [], [], [], []

        crop_boxes = torch.stack([crop_x1, crop_y1, crop_x2, crop_y2], dim=1).cpu().tolist()
        lb_boxes = torch.stack([lb_x1, lb_y1, lb_x2, lb_y2], dim=1).cpu().tolist()
        boxes = [[int(v) for v in b] for b in lb_boxes]
        confs = conf.cpu().tolist()
        cls_ids = cls_id.cpu().tolist()

        # 최소 crop 크기 (리사이즈 프레임 기준)
        # 너무 작은 crop은 128×64 업스케일 시 feature가 붕괴되어 오탐 발생
        # person: ReID feature 품질 보장을 위해 최소 48×32 필요
        # attribute: 224×224 리사이즈이므로 32×16
        if self.target_type in ("person",):
            # min_crop_h = 40
            # min_crop_w = 21
            min_crop_h, min_crop_w = self.min_crop_size
        elif self.target_type in ("attribute",):
            # min_crop_h = 40
            # min_crop_w = 21
            min_crop_h, min_crop_w = self.min_crop_size
        elif self.target_type in ("carplate",):
            # 번호판: OCR 인식 품질 보장을 위해 최소 크기 필터링
            # 190×60 리사이즈 대상이므로 너무 작으면 문자 판독 불가
            # min_crop_h = 21
            # min_crop_w = 40
            min_crop_h, min_crop_w = self.min_crop_size
        else:
            min_crop_h = 0
            min_crop_w = 0

        crops = []
        valid_boxes = []
        valid_confs = []
        valid_cls_ids = []
        for idx, (bx1, by1, bx2, by2) in enumerate(crop_boxes):
            #if confs[idx] < min_det_conf:
                # self.logger.debug(
                #     f"[crop_filter] conf={confs[idx]:.3f} < {min_det_conf} "
                #     f"box=[{bx1},{by1},{bx2},{by2}] skipped"
                # )
                #continue
            crop = frame_bgr[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue
            crop_h, crop_w = crop.shape[:2]
            if crop_h < min_crop_h or crop_w < min_crop_w:
                # self.logger.debug(
                #     f"[crop_filter] size={crop_w}x{crop_h} < min={min_crop_w}x{min_crop_h} "
                #     f"box=[{bx1},{by1},{bx2},{by2}] skipped"
                # )
                continue
            
            if self.target_type in ["person", "attribute"] and crop_h <= crop_w:
                continue
            elif self.target_type == "carplate" and crop_h >= crop_w:
                continue
            
            # # 면적 필터: 너무 작은 bbox는 ReID feature 품질 보장 불가
            # crop_area = crop_h * crop_w
            # if self.target_type in ("person", "attribute") and crop_area < 2000:
            #     self.logger.debug(
            #         f"[crop_filter] area={crop_area} < 2000 "
            #         f"box=[{bx1},{by1},{bx2},{by2}] skipped"
            #     )
            #     continue
            # # 종횡비 필터: 사람은 세로가 가로보다 충분히 길어야 함 (h/w >= 1.2)
            # if self.target_type in ("person", "attribute") and crop_h < crop_w * 1.2:
            #     self.logger.debug(
            #         f"[crop_filter] aspect={crop_h/crop_w:.2f} < 1.2 "
            #         f"box=[{bx1},{by1},{bx2},{by2}] skipped"
            #     )
            #     continue
            
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
            valid_boxes.append(boxes[idx])  # letterbox 좌표 유지
            valid_confs.append(confs[idx])
            valid_cls_ids.append(cls_ids[idx])

        return crops, valid_boxes, valid_confs, valid_cls_ids
    
    def _preproc_crops_person(self, all_crops: list[np.ndarray]) -> torch.Tensor:
        """person crop 배치 전처리: 개별 resize(128,64) → normalize.

        각 crop을 개별적으로 (128,64)에 맞춰 resize한 뒤 stack.
        패딩 없이 resize하므로 crop 크기 차이에 의한 feature 왜곡이 없음.
        """
        if not all_crops:
            return None

        resized = []
        for crop_rgb in all_crops:
            x = torch.from_numpy(
                np.ascontiguousarray(crop_rgb)
            ).to(self.device, non_blocking=True)
            x = x.permute(2, 0, 1).float().div_(255.0)
            x = x.unsqueeze(0)
            x = F.interpolate(
                x, size=self.analysis_input_size,
                mode="bilinear", align_corners=False,
            )
            resized.append(x)

        batch = torch.cat(resized, dim=0)
        batch = (batch - self.analysis_mean) / self.analysis_std
        batch = batch.half()
        return batch

    def _preproc_crops_attribute(self, all_crops: list[np.ndarray]) -> torch.Tensor:
        """attribute crop 배치 전처리: resize(224,224) → normalize."""
        if not all_crops:
            return None

        resized = []
        for crop_rgb in all_crops:
            x = torch.from_numpy(
                np.ascontiguousarray(crop_rgb)
            ).to(self.device, non_blocking=True)
            x = x.permute(2, 0, 1).float().div_(255.0)
            x = x.unsqueeze(0)
            x = F.interpolate(
                x, size=self.analysis_input_size,
                mode="bilinear", align_corners=False,
            )
            resized.append(x)

        batch = torch.cat(resized, dim=0)
        del resized # 메모리 해제
        
        batch = (batch - self.analysis_mean) / self.analysis_std
        batch = batch.half()
        return batch

    def _preproc_crops_carplate(self, all_crops: list[np.ndarray]) -> torch.Tensor:
        """carplate crop 배치 전처리: RGB→Grayscale→resize(60,190)→normalize.

        OCR 모델 입력: (N, 1, 60, 190) FP16
        """
        if not all_crops:
            return None

        resized = []
        for crop_rgb in all_crops:
            x = torch.from_numpy(
                np.ascontiguousarray(crop_rgb)
            ).to(self.device, non_blocking=True)
            x = x.permute(2, 0, 1).float().div_(255.0)  # (3,H,W)
            # RGB → Grayscale: ITU-R 601 가중 평균
            x = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]  # (1,H,W) 
            x = x.unsqueeze(0)  # (1,1,H,W)
            x = F.interpolate(
                x, size=self.analysis_input_size,
                mode="bicubic", align_corners=False,
            )
            resized.append(x)

        batch = torch.cat(resized, dim=0)  # (N,1,60,190)
        del resized
        
        batch = (batch - self.analysis_mean) / self.analysis_std
        batch = batch.half()
        return batch

    def _match_carplate(self, features_gpu: torch.Tensor, boxes: list, confs: list, cls_ids: list) -> list[dict]:
        """carplate OCR 배치 디코딩 + CTC decode + 필터링 + plate_match.

        features_gpu: (N, 48, 55) GPU float32 텐서 — OCR 모델 raw logits (softmax 전)
        조건: score >= 0.7, 길이 7~8, 한글 포함, 뒤 4글자 매칭
        """
        import re
        

        results = []
        #pattern = re.compile('[ㄱ-ㅎ가-힣]+')

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            logit_t = features_gpu[idx:idx + 1]  # (1, 48, 55) GPU — 복사 없음

            if not torch.isfinite(logit_t).all():
                self.logger.warning(f"[match_plate] NaN/Inf in logits, skipping box={box}")
                continue
            
            # greedy argmax
            preds_size = torch.IntTensor([logit_t.size(1)])
            _, preds_index = logit_t.max(2)
            preds_str = self.converter.decode(preds_index, preds_size)

            # confidence: softmax → max prob 평균 / 26
            preds_prob = F.softmax(logit_t, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            score = float(preds_max_prob[0].sum().item() / 26)

            pred = preds_str[0]

            # 필터링: score >= 0.7, 길이 7~8, 한글 포함
            # if score < 0.7:
            #     continue
            # if len(pred) < 7 or len(pred) > 8:
            #     continue
            # if not pattern.search(pred):
            #     continue
            
            # 필터링: score >= 0.7, 길이 7~8, 한글 포함
            if score < 0.7:
                continue
            
            # 글자 존재 유무에 따른 로직 분기
            # match = re.search(r'[ㄱ-ㅎ가-힣](\d{4,})', pred) # 마지막 한글 뒤 숫자 4자이상
            # if match:
            #     pred = match.group(1)            
            # else:
            #     if len(pred) < 4:
            #         continue
            
            # result_ocr = list(pred[-4:])
            
            if re.search(r'[ㄱ-ㅎ가-힣]', pred):
                match = re.search(r'[ㄱ-ㅎ가-힣](\d{4})\D*$', pred)
                if not match:
                    continue
                result_ocr = list(match.group(1))

            # 한글이 하나도 없으면: 마지막 숫자 4개
            else:
                digits = re.findall(r'\d', pred)
                if len(digits) < 4:
                    continue
                result_ocr = digits[-4:]

            if self.target_feature:
                scores_dict = plate_match(self.target_feature, result_ocr, box)
                for target_id, items in scores_dict.items():
                    results.append({
                        "targetId": str(target_id),
                        "accuracy": float(items["score"]),
                        "infer_ocr": pred,
                        "confidence": conf,
                        "classId": cls_id,
                        "points": box,
                    })
                    
                    # 번호판 후보 로그
                    self.logger.info(
                            f"[type] type={self.target_type}"
                            f"[match] target={target_id} score={score:.4f} result_ocr={result_ocr}"
                            f"box={box} conf={conf:.3f}"
                        )
            else:
                # target 없으면 OCR 결과만 반환
                results.append({
                    "targetId": "0",
                    "accuracy": score,
                    "infer_ocr": pred,
                    "confidence": conf,
                    "classId": cls_id,
                    "points": box,
                })

        return results

    def _match_person(self, features_np: np.ndarray, boxes: list, confs: list, cls_ids: list) -> list[dict]:
        """person feature 코사인 유사도 비교 → 매칭 결과.

        plr_osnet_trt.py 기반: PLR-OSNet feature를 코사인 유사도로 비교.
        target_feature가 없으면 검출 결과만 반환 (score=0).
        """
        results = []
        if not self._target_norms:
            # 타겟 feature가 없으면 매칭 불가 → 빈 결과 반환 (오탐 방지)
            return results

        for feat, box, conf, cls_id in zip(features_np, boxes, confs, cls_ids):
            feat_flat = feat.flatten().astype(np.float32)

            # NaN/Inf 방어: FP16 추론 결과에 NaN이 포함될 수 있음
            if not np.isfinite(feat_flat).all():
                # self.logger.warning(f"[match] NaN/Inf in feature vector, skipping box={box}")
                continue

            feat_norm = np.linalg.norm(feat_flat)
            if feat_norm > 0:
                feat_flat = feat_flat / feat_norm
            else:
                continue  # zero vector → 매칭 불가

            for target_id, target_norm in self._target_norms.items():
                score = float(np.dot(target_norm, feat_flat))
                if score < self.match_score_thresh:
                    continue
                
                # 전신 후보 로그
                self.logger.info(
                    f"[type] type={self.target_type}"
                    f"[match] target={target_id} score={score:.4f} "
                    f"thresh={self.match_score_thresh} box={box} conf={conf:.3f}"
                )
                
                results.append({
                    "targetId": str(target_id),
                    "accuracy": score,
                    "confidence": conf,
                    "classId": cls_id,
                    "points": box,
                })

        return results

    def _match_attribute(self, features_np: np.ndarray, boxes: list, confs: list, cls_ids: list) -> list[dict]:
        """attribute sigmoid → attr_match_score 기반 속성 분류 매칭.

        lib/core/main.py process_attribute + scores.py attr_match_score 로직 기반.
        ConvFormer 출력(89차원) → sigmoid → 속성 인덱스별 매칭 → attribution_value 구조 반환.
        """
        from .util.attr_config import attribution_config, attribution_key, color_list

        results = []
        if not self.target_feature:
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                results.append({
                    "targetId": "0",
                    "accuracy": 0.0,
                    "confidence": conf,
                    "classId": cls_id,
                    "points": box,
                })
            return results
        
        # NaN/Inf 방어: FP16 추론 결과 보호
        if not np.isfinite(features_np).all():
            self.logger.warning(f"[match_attr] NaN/Inf in features, replacing with 0")
            features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)

        features_sigmoid = 1.0 / (1.0 + np.exp(-features_np))
        for output, box, conf, cls_id in zip(features_sigmoid, boxes, confs, cls_ids):
            # 비인체 입력 필터링: 실제 사람이면 sigmoid 출력이 0/1 근처에 분산되지만
            # 관목/배경 등 비인체 입력이면 대부분 0.5 근처에 몰림
            # confidence_spread = 전체 89차원의 |sigmoid - 0.5| 평균
            confidence_spread = float(np.mean(np.abs(output - 0.5)))
            if confidence_spread < 0.08:
                self.logger.info(
                    f"[attr_filter] non-person crop skipped: "
                    f"spread={confidence_spread:.3f} conf={conf:.3f} box={box}"
                )
                continue

            for target_id, target_indices in self.target_feature.items():
                score = 0.0
                return_true_att_list = []
                
                outer_score = 0.0
                bottom_score = 0.0
                
                outer_filtered_indices = [
                    v for v in self.outer_color_list
                    if v in target_indices
                ]

                bottom_filtered_indices = [
                    v for v in self.bottom_color_list
                    if v in target_indices
                ]
                
                if outer_filtered_indices:
                    result = output[self.outer_color_list]  # shape: [len(outer_color_list)]

                    # ✅ outer_filtered_indices가 outer_color_list 내에서 몇 번째 위치인지
                    outer_color_set = {v: i for i, v in enumerate(self.outer_color_list)}
                    gt_local = [outer_color_set[v] for v in outer_filtered_indices]
                    
                    #outer_best_gt, outer_best_score, outer_delete = best_color_score_logits(result, gt_local)
                    outer_best_local, outer_best_score, outer_delete = best_color_score_logits(result, gt_local, self.match_color_filter_thresh)
                    outer_best_gt = self.outer_color_list[outer_best_local]
                            
                else:
                    outer_best_score = 0
                    outer_delete = False
                
                if bottom_filtered_indices:
                    
                    result = output[self.bottom_color_list]
                    
                    bottom_color_set = {v: i for i, v in enumerate(self.bottom_color_list)}
                    gt_local = [bottom_color_set[v] for v in bottom_filtered_indices]
                    

                    #bottom_best_gt, bottom_best_score, bottom_delete = best_color_score_logits(result, gt_local)
                    bottom_best_local, bottom_best_score, bottom_delete = best_color_score_logits(result, gt_local, self.match_color_filter_thresh)
                    bottom_best_gt = self.bottom_color_list[bottom_best_local]
                                
                else:
                    bottom_best_score = 0
                    bottom_delete = False

                
                if bottom_delete or outer_delete:
                    continue

                for idx in target_indices:
                    if idx not in attribution_config:
                        continue
                    if output[idx] > self.match_score_thresh:
                        accuracy_weight = attribution_config[idx]['accuracy']
                        weighted = float(output[idx]) * accuracy_weight
                        
                        if idx in self.outer_color_list:
                            if idx == outer_best_gt:
                                score += (outer_best_score * accuracy_weight)
                                att_label = attribution_config[idx]['attrInfo']
                                return_true_att_list.append({
                                    "label": att_label,
                                    "score": (outer_best_score * accuracy_weight),
                                })
                        elif idx in self.bottom_color_list:
                            if idx == bottom_best_gt:
                                score += (bottom_best_score * accuracy_weight)
                                att_label = attribution_config[idx]['attrInfo']
                                return_true_att_list.append({
                                    "label": att_label,
                                    "score": (bottom_best_score * accuracy_weight),
                                })
                        else:
                            score += weighted
                            att_label = attribution_config[idx]['attrInfo']
                            return_true_att_list.append({
                                "label": att_label,
                                "score": weighted,
                            })
                        
                        # if idx in outer_color_list:
                        #     outer_score += weighted
                        
                        # if idx in bottom_color_list:
                        #     bottom_score += weighted
                        
                    self.logger.info(f"[attr_each_attr] attr_score: {score:.3f}, outer_score: {outer_score:.3f}, bottom_score: {bottom_score:.3f}. output: {output[idx]:.3f}")        

                #non_color_score = score - (outer_score + bottom_score)
                self.logger.info(f"[attr] attr_score: {score:.3f}, outer_score: {outer_score:.3f}, bottom_score: {bottom_score:.3f}")
                
                if len(return_true_att_list) == 0:
                    continue
                
                if len(bottom_filtered_indices) > 0 or len(outer_filtered_indices) > 0:
                    
                    if len(bottom_filtered_indices) > 0 and len(outer_filtered_indices) > 0:
                        score = (score / len(return_true_att_list)) * (len(return_true_att_list) / (len(target_indices) - (len(outer_filtered_indices) + len(bottom_filtered_indices)) + 2))
                    else:
                        score = (score / len(return_true_att_list)) * (len(return_true_att_list) / (len(target_indices) - (len(outer_filtered_indices) + len(bottom_filtered_indices)) + 1))
                
                else:
                    score = (score / len(return_true_att_list)) * (len(return_true_att_list) / (len(target_indices)))
                    
                
                ####################################################### 색상 정확도 개선 로직 #######################################################
                # if non_color_score != 0:
                #     if bottom_color_avg_score != 0 and outer_color_avg_score != 0:
                #         color_score = (outer_color_avg_score + bottom_color_avg_score) / 2
                #         score = score * (color_score ** 2)
                #     elif bottom_color_avg_score == 0 and outer_color_avg_score == 0:
                #         pass
                #     elif outer_color_avg_score == 0:
                #         color_score = bottom_color_avg_score
                #         score = score * (color_score ** 2)
                #     elif bottom_color_avg_score == 0:
                #         color_score = outer_color_avg_score
                #         score = score * (color_score ** 2)
                    
                ####################################################### 색상 정확도 개선 로직 #######################################################
                
                # 속성 수에 따른 동적 threshold: 속성이 적을수록 높은 확신 요구
                # 속성 1개("짧은 머리"만)로 50채널 검색 시 대량 오탐 방지
                #n_target_attrs = len(target_indices)
                # if n_target_attrs <= 1:
                #     attr_threshold = 0.85
                # elif n_target_attrs <= 2:
                #     attr_threshold = 0.7
                # elif n_target_attrs <= 3:
                #     attr_threshold = 0.6
                # else:
                #     attr_threshold = self.match_score_thresh
                if score < self.match_attr_avg_score_thresh:
                    continue

                # attribution_value 구조 생성
                attribution_value = {
                    "general": {
                        "hair": {"value": None, "score": 0},
                        "gender": {"value": None, "score": 0},
                    },
                    "outer": {
                        "type": {"value": None, "score": 0},
                        "shape": {"value": None, "score": 0},
                        "pattern": {"value": None, "score": 0},
                        "color": [],
                    },
                    "inner": {
                        "pattern": {"value": None, "score": 0},
                        "color": [],
                    },
                    "bottom": {
                        "type": {"value": None, "score": 0},
                        "pattern": {"value": None, "score": 0},
                        "color": [],
                    },
                    "shoes": {
                        "type": {"value": None, "score": 0},
                    },
                    "etc": {
                        "mask": {"value": None, "score": 0},
                        "glasses": {"value": None, "score": 0},
                        "bag": {"value": None, "score": 0},
                        "walkingaids": {"value": None, "score": 0},
                    },
                }

                for att_value in return_true_att_list:
                    parts = att_value["label"].split('_')
                    first_key, value = parts[0], parts[1]
                    if value not in attribution_key:
                        continue
                    second_key = attribution_key[value]
                    if first_key not in attribution_value:
                        continue
                    if second_key not in attribution_value[first_key]:
                        continue
                    if value in color_list:
                        attribution_value[first_key][second_key].append({
                            "value": value,
                            "score": att_value["score"],
                        })
                    else:
                        attribution_value[first_key][second_key]["value"] = value
                        attribution_value[first_key][second_key]["score"] = att_value["score"]


                # 인상착의 후보 로그
                self.logger.info(
                    f"[type] type={self.target_type}"
                    f"[match] target={target_id} score={score:.4f} "
                    f"thresh={self.match_attr_avg_score_thresh} box={box} attr_value={attribution_value} "
                    f"match_attr_count={len(return_true_att_list)}"
                )
                
                results.append({
                    "targetId": str(target_id),
                    "accuracy": score,
                    "attr_value": attribution_value,
                    "matched_attr_count": len(return_true_att_list),
                    "confidence": conf,
                    "classId": cls_id,
                    "points": box,
                })

        return results

    def run_batch(
        self,
        frames: list[np.ndarray],
        image_ids: list = None,
        device_ids: list = None,
        origin_size: list[tuple] = None
    ) -> list[tuple]:
        """N프레임 배치 추론 실행.

        Returns:
            list[tuple(inference_data, image_id, device_id)]
            - inference_data: list[dict] — 매칭 결과 (targetId, accuracy, points)
            - image_id, device_id: 그대로 전달
        """
        n = len(frames)
        if image_ids is None:
            image_ids = [None] * n
        if device_ids is None:
            device_ids = [None] * n

        t_start = time.perf_counter()

        if origin_size is None:
            default_origin_sizes = []
            for i in range(n):
                default_origin_sizes.append(tuple(self.origin_size))
        
        # 1. 배치 전처리 (letterbox)
        batch_tensor, orig_shapes = self._preproc_frames(frames)
        t_preproc = time.perf_counter()

        # 2. 배치 detection
        detections = self._batch_detect(batch_tensor, orig_shapes)
        t_detect = time.perf_counter()

        # 3. 프레임별 crop 추출
        all_crops = []
        all_boxes = []
        all_confs = []
        all_cls_ids = []
        frame_crop_counts = []

        for i in range(n):
            crops, boxes, confs, cls_ids = self._extract_crops(frames[i], detections[i])

            # person/attribute: w > h 가로형 박스 필터링 (사람이 아닌 오브젝트 제거)
            # if self.target_type in ("person", "attribute"):
            #     filtered_crops = []
            #     filtered_boxes = []
            #     filtered_confs = []
            #     filtered_cls_ids = []
            #     for crop, box, conf, cls_id in zip(crops, boxes, confs, cls_ids):
            #         bx1, by1, bx2, by2 = box
            #         if (bx2 - bx1) > (by2 - by1):
            #             continue
            #         filtered_crops.append(crop)
            #         filtered_boxes.append(box)
            #         filtered_confs.append(conf)
            #         filtered_cls_ids.append(cls_id)
            #     crops, boxes, confs, cls_ids = filtered_crops, filtered_boxes, filtered_confs, filtered_cls_ids

            all_crops.extend(crops)
            all_boxes.extend(boxes)
            all_confs.extend(confs)
            all_cls_ids.extend(cls_ids)
            frame_crop_counts.append(len(crops))

        t_crop = time.perf_counter()

        # 4. 배치 analysis (crop이 있는 경우만, max_batch 분할 처리)
        all_features_np = None
        all_features_gpu = None  # carplate용 GPU 텐서 (D2H 회피)
        if all_crops and self.target_type in ("person", "attribute", "carplate"):
            if self.target_type == "person":
                crop_batch = self._preproc_crops_person(all_crops)
            elif self.target_type == "attribute":
                crop_batch = self._preproc_crops_attribute(all_crops)
            else:
                crop_batch = self._preproc_crops_carplate(all_crops)

            t_crop_preproc = time.perf_counter()

            max_b = min(self.analysis_batch_size, self.analysis_engine.max_batch)
            total = crop_batch.shape[0]

            if total <= max_b:
                # max_batch 이내 → 패딩하여 고정 배치 추론 (버퍼 재할당 방지)
                if total < max_b:
                    pad = torch.zeros(
                        (max_b - total, *crop_batch.shape[1:]),
                        dtype=crop_batch.dtype, device=crop_batch.device,
                    )
                    padded = torch.cat([crop_batch, pad], dim=0)
                    del pad # 메모리 해제
                else:
                    padded = crop_batch
                
                if self.target_type == "person" and any(x in torch.cuda.get_device_name(0) for x in ["5080", "5090"]):
                    analysis_outputs = self.analysis_engine(padded)
                else:
                    analysis_outputs = self.analysis_engine.infer(padded)
                
                raw_out = analysis_outputs[0][:total]
                del padded, crop_batch # 메모리 해제
                
                if self.target_type == "carplate":
                    all_features_gpu = raw_out.float()
                else:
                    all_features_np = raw_out.cpu().detach().float().numpy()
            else:
                # max_batch 초과 → 분할 추론 후 결합 (각 청크를 max_b로 패딩)
                feat_chunks = []
                for start in range(0, total, max_b):
                    chunk = crop_batch[start:start + max_b]
                    actual = chunk.shape[0]
                    if actual < max_b:
                        pad = torch.zeros(
                            (max_b - actual, *chunk.shape[1:]),
                            dtype=chunk.dtype, device=chunk.device,
                        )
                        chunk = torch.cat([chunk, pad], dim=0)
                        del pad # 메모리 해제
                    
                    #chunk_out = self.analysis_engine.infer(chunk)
                    
                    if self.target_type == "person" and any(x in torch.cuda.get_device_name(0) for x in ["5080", "5090"]):
                        chunk_out = self.analysis_engine(chunk)
                    else:
                        chunk_out = self.analysis_engine.infer(chunk)
                    feat_chunks.append(chunk_out[0][:actual])
                
                combined = torch.cat(feat_chunks, dim=0)
                del feat_chunks # 메모리 해제
                
                if self.target_type == "carplate":
                    all_features_gpu = combined.float()
                else:
                    all_features_np = combined.cpu().detach().float().numpy()
                # self.logger.info(
                #     f"[BatchInference] analysis split: total={total} max_batch={max_b} "
                #     f"chunks={len(feat_chunks)}"
                # )

            t_analysis = time.perf_counter()
        else:
            t_crop_preproc = t_crop
            t_analysis = t_crop

        # 5. 프레임별 매칭 + bbox 좌표 원본 스케일 역변환
        has_features = all_features_np is not None or all_features_gpu is not None
        output = []
        offset = 0
        for i in range(n):
            count = frame_crop_counts[i]
            if count == 0 or not has_features:
                output.append(([], image_ids[i], device_ids[i]))
            else:
                frame_boxes = all_boxes[offset:offset + count]
                frame_confs = all_confs[offset:offset + count]
                frame_cls_ids = all_cls_ids[offset:offset + count]

                if self.target_type == "carplate":
                    # GPU 텐서 직접 사용 (D2H→H2D 왕복 회피)
                    frame_feats_gpu = all_features_gpu[offset:offset + count]
                    matched = self._match_carplate(frame_feats_gpu, frame_boxes, frame_confs, frame_cls_ids)
                else:
                    frame_feats = all_features_np[offset:offset + count]
                    if self.target_type == "person":
                        matched = self._match_person(frame_feats, frame_boxes, frame_confs, frame_cls_ids)
                    elif self.target_type == "attribute":
                        matched = self._match_attribute(frame_feats, frame_boxes, frame_confs, frame_cls_ids)
                    else:
                        matched = []
                
                final_matched = []
                actual_orig = origin_size[i] if origin_size is not None else self.origin_size
                frame_h, frame_w = self.image_size if self.image_size else orig_shapes[i]
                if matched:
                    if self.image_size is not None:
                        for det in matched:
                            pts = det.get("points")
                            det["points"] = letterbox_xyxy_to_original(
                                pts, actual_orig, (frame_h, frame_w)
                            )[0]
                            
                            final_matched.append(det)
                            
                            
                            # self.logger.info(
                            #     f"[coord] letterbox→orig: {pts} → {det['points']}, "
                            #     f"origin={actual_orig} lb=({frame_h},{frame_w})"
                            # )
                    # 최종 필터
                    # filtered_matched = []
                    # for det in matched:
                    #     filtered_matched.append(det)
                    #matched = filtered_matched

                output.append((final_matched, image_ids[i], device_ids[i]))
                offset += count

        t_end = time.perf_counter()

        # self.logger.info(
        #     f"[BatchInference] frames={n} "
        #     f"preproc:{(t_preproc-t_start)*1000:.1f}ms "
        #     f"detect:{(t_detect-t_preproc)*1000:.1f}ms "
        #     f"crop:{(t_crop-t_detect)*1000:.1f}ms "
        #     f"crop_preproc:{(t_crop_preproc-t_crop)*1000:.1f}ms "
        #     f"analysis:{(t_analysis-t_crop_preproc)*1000:.1f}ms "
        #     f"total:{(t_end-t_start)*1000:.1f}ms "
        #     f"crops:{sum(frame_crop_counts)}"
        # )

        return output

    def person_auto_detect(
        self,
        frames: list[np.ndarray],
        origin_sizes: list[tuple] = None
    ) -> dict:
    
    
        batch_tensor, orig_shapes = self._preproc_frames(frames)
        # 2. 배치 detection
        detections = self._batch_detect(batch_tensor, orig_shapes)
        det = detections[0]
        det = det.float()
        
        
        in_h, in_w = batch_tensor.shape[-2:]
        boxes = []
        cls_list = []
        for di in range(len(det)):
            cls_id = int(det[di, 5])
            conf = float(det[di, 4])
            raw_box = det[di, :4].tolist()
            box = letterbox_xyxy_to_original(raw_box, origin_sizes[0], (in_h, in_w))[0]
            if cls_id == 0:
                boxes.append([int(v) for v in box])
                cls_list.append(cls_id)
        
        if len(boxes) == 0:
            return None
        
        res_dict ={
            "boxes": boxes,
            "cls": cls_list
        }
        
        return res_dict

        
        
    
    

class FaceInference:
    def __init__(
        self,
        target_feature: dict = None,
        gpu_idx: int = 0,
        detect_batch_size: int = 1,
        analysis_batch_size: int = 32,
        vis_thresh: float = 0.9,
        nms_thresh: float = 0.4,
        min_face_width: int = 28,
        score_threshold: float = 0.3,
        cfg: dict = {},
        logger: logging.Logger = None,
    ):
        
        self.image_size = cfg["image_size"]["face_frame"]
        self.target_feature = target_feature
        self.detect_batch_size = detect_batch_size
        self.analysis_batch_size = analysis_batch_size
        self.vis_thresh = vis_thresh
        self.nms_thresh = nms_thresh
        self.min_face_width = min_face_width
        self.score_threshold = score_threshold
        self.logger = logger or logging.getLogger(__name__)

        torch.cuda.set_device(gpu_idx)
        self.device = torch.device(f"cuda:{gpu_idx}")
        

        from .util.retinaface import RetinaFace

        self.retina = RetinaFace(model_input_size=self.image_size, device=self.device)
        self.detect_engine = AnalysisEngine_v4(cfg["model_path"]["face_detect"], fp16=False, output_count=3, device=self.device)
        
        self.analysis_half_engine = AnalysisEngine_v4(cfg["model_path"]["face_half_analysis"], fp16=False, output_count=1, device=self.device)
        self.analysis_full_engine = AnalysisEngine_v4(cfg["model_path"]["face_full_analysis"], fp16=False, output_count=1, device=self.device)

        self.half_transform = T.Compose([T.Grayscale(), T.CenterCrop((64, 128))])
        self.full_transform = T.Compose([T.CenterCrop(128)])
    
    def _detect_face(self, frame: np.ndarray):
        
        img_new, pad_left, pad_top, ratio = self.retina.letterbox(frame)
        
        img_tensor = torch.from_numpy(img_new).to(self.device, non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).float()
        img_tensor.sub_(torch.tensor([104, 117, 123], device=self.device).view(3,1,1))
        img_tensor = img_tensor.unsqueeze(0).contiguous()
        
        scale = torch.Tensor([img_new.shape[1], img_new.shape[0], img_new.shape[1], img_new.shape[0]])
        scale = scale.to(self.device)

        outputs = self.detect_engine(img_tensor)
        # RetinaFace 출력: loc(B,N,4), conf(B,N,2), landms(B,N,10)
        loc = outputs[0][:1].float()
        conf = outputs[1][:1].float()
        landms = outputs[2][:1].float()
        
        dets = self.retina.retina_postprocess(img_tensor, loc, conf, landms, scale, 1)
        list_OMS_Face = self.retina.point_post_process(dets, ratio, pad_left, pad_top)
        
        return list_OMS_Face

    
    def _retinaface_preproc_to_second_model(self, frame:np.ndarray, list_face):
        candid_face_list = list()
        candid_half_face_list = list()
        candid_bbox_list = list()
        
        for face in list_face:
            face_boxing = [pt if pt > 0 else 0 for pt in face.rt]
            cropped_face = frame[face_boxing[1]:face_boxing[3], face_boxing[0]:face_boxing[2]]
            landmarks = [face.ptLE, face.ptRE, face.ptLM, face.ptRM, face.ptN]
            roi_img, roi_img_half = face_alignment_renew(cropped_face, landmarks, face_boxing)
            
            # === HWC -> CHW + Tensor ===
            img_tensor = torch.from_numpy(roi_img).permute(2, 0, 1).contiguous()  # (3,h,w), uint8
            img_half_tensor = torch.from_numpy(roi_img_half).permute(2, 0, 1).contiguous()  # (3,h,w), uint8

            # === float / 255 ===
            img_tensor = img_tensor.float().div_(255.0)
            img_half_tensor = img_half_tensor.float().div_(255.0)

            # === GPU 이동 ===
            img_tensor = img_tensor.to(self.device, non_blocking=True)  # (3,h,w) on GPU
            img_half_tensor = img_half_tensor.to(self.device, non_blocking=True)
            
            candid_face_list.append(img_tensor)
            candid_half_face_list.append(img_half_tensor)
            candid_bbox_list.append(face_boxing)
        
        return candid_face_list, candid_half_face_list, candid_bbox_list
        
    # def _match_face_process(self, crops, theta_params, half_params, bbox_list):
    #     outputs = []
        
    #     N = len(crops)
    #     if N == 0:
    #         return outputs

    #     for start in range(0, N, self.batch_size):
    #         chunk_crops  = crops       [start:start + self.batch_size]
    #         chunk_theta  = theta_params[start:start + self.batch_size]
    #         chunk_half_p = half_params [start:start + self.batch_size]
    #         chunk_boxes  = bbox_list   [start:start + self.batch_size]
    #         chunk_n      = len(chunk_crops)

            

    #         # ── 1. pinned buffer 재사용 → H2D 1회 ────────────────────────────
    #         max_h = max(c.shape[0] for c in chunk_crops)
    #         max_w = max(c.shape[1] for c in chunk_crops)

    #         #self._face_np_buf[:chunk_n, :max_h, :max_w] = 0
    #         batch_np    = np.zeros((chunk_n, max_h, max_w, 3), dtype=np.uint8)
    #         extra_tops  = []
    #         extra_lefts = []

    #         for i, crop in enumerate(chunk_crops):
    #             h, w = crop.shape[:2]
    #             et = (max_h - h) // 2
    #             el = (max_w - w) // 2
    #             batch_np[i, et:et+h, el:el+w] = crop
    #             extra_tops.append(et)
    #             extra_lefts.append(el)

    #         src_tensor = (
    #             torch.from_numpy(batch_np)
    #             .to(self.device, non_blocking=True)
    #             .permute(0, 3, 1, 2).float().div_(255.0)      # (chunk_n,3,max_h,max_w)
    #             .contiguous()
    #         )

    #         t1 = time.perf_counter()

    #         # ── 2. affine warp (GPU 1회) ──────────────────────────────────────
    #         sw_f = 2.0 / (max_w - 1)
    #         sh_f = 2.0 / (max_h - 1)

    #         thetas_np = np.empty((chunk_n, 2, 3), dtype=np.float32)
    #         for i, (tx_base, ty_base, cos_d, sin_d, s) in enumerate(chunk_theta):
    #             tx = tx_base + extra_lefts[i]
    #             ty = ty_base + extra_tops[i]
    #             thetas_np[i, 0] = [ sw_f * self._face_affine_half_w * cos_d/s,  sw_f * self._face_affine_half_w * sin_d/s,  sw_f*tx - 1.0]
    #             thetas_np[i, 1] = [-sh_f * self._face_affine_half_w * sin_d/s,  sh_f * self._face_affine_half_w * cos_d/s,  sh_f*ty - 1.0]

    #         theta_t  = torch.from_numpy(thetas_np).to(self.device, non_blocking=True)
    #         grid     = torch.nn.functional.affine_grid(
    #             theta_t, (chunk_n, 3, self._face_aligned_h, self._face_aligned_w), align_corners=True
    #         )
    #         roi_batch = torch.nn.functional.grid_sample(
    #             src_tensor, grid,
    #             mode='bilinear', padding_mode='zeros', align_corners=True
    #         )                                                   # (chunk_n, 3, 128, 128)

    #         t2 = time.perf_counter()

    #         # ── 3. full model ─────────────────────────────────────────────────
    #         if len(self.target_feature["full"]) > 0:
    #             fh, fw   = self._full_crop_h, self._full_crop_w
    #             sh_start = (self._face_aligned_h - fh) // 2
    #             sw_start = (self._face_aligned_w - fw) // 2

    #             full_input = (
    #                 roi_batch[:chunk_n, :, sh_start:sh_start+fh, sw_start:sw_start+fw]
    #                 .to(dtype=self._full_input_dtype)
    #                 .contiguous()
    #             )

    #             full_result = self.analysis_full_model(full_input)
    #             full_np     = full_result[0][:chunk_n].cpu().numpy()

    #             t3 = time.perf_counter()

    #             for i, feat in enumerate(full_np):
    #                 scores_dict = face_match_score(self.target_feature["full"], feat)
    #                 for target_id, score in scores_dict.items():
    #                     outputs.append({
    #                         'targetId': target_id,
    #                         'accuracy': float(score),
    #                         'points':   chunk_boxes[i]
    #                     })
            

    #         # ── 4. half model ─────────────────────────────────────────────────
    #         if len(self.target_feature["half"]) > 0:
    #             half_hs = []
    #             for cy_crop, n_pt_y_crop, s in chunk_half_p:
    #                 cut_s   = int(n_pt_y_crop * s)
    #                 start_s = max(0, int(cy_crop * s) - 40)
    #                 half_h  = max(self._half_crop_h, min(cut_s - start_s, self._face_aligned_h))
    #                 half_hs.append(half_h)

    #             # full 추론 완료 후 roi_batch 재사용 (clone 제거)
    #             for i, hh in enumerate(half_hs):
    #                 if hh < self._face_aligned_h:
    #                     roi_batch[i, :, hh:, :] = 0.0

    #             # Grayscale: GPU 배치 weighted sum
    #             gray_batch = (roi_batch[:chunk_n] * self._gray_w).sum(dim=1, keepdim=True)

    #             # CenterCrop
    #             hh_c = self._half_crop_h
    #             hw_c = self._half_crop_w
    #             hsh  = (self._face_aligned_h - hh_c) // 2
    #             hsw  = (self._face_aligned_w - hw_c) // 2

    #             half_input = (
    #                 gray_batch[:, :, hsh:hsh+hh_c, hsw:hsw+hw_c]
    #                 .to(dtype=self._half_input_dtype)
    #                 .contiguous()
    #             )

    #             half_result = self.analysis_half_model(half_input)
    #             half_np     = half_result[0][:chunk_n].cpu().numpy()

    #             t4 = time.perf_counter()

    #             for i, feat in enumerate(half_np):
    #                 scores_dict = face_match_score(self.target_feature["half"], feat)
    #                 for target_id, score in scores_dict.items():
    #                     outputs.append({
    #                         'targetId': target_id,
    #                         'accuracy': float(score),
    #                         'points':   chunk_boxes[i]
    #                     })    
    #     return outputs
    
    def _match_face_process(self, candidate_face_tensor_list, candidate_half_face_list, candidate_bbox_list) -> List:
        outputs = []
        self.logger.info(f"target_feature : {self.target_feature}")
        if len(self.target_feature["full"]) > 0:
            # for feature_dict in self.target_feature["full"]:
            #     for target_id, feature in feature_dict.items():
            for idx, align_face_tensor_img in enumerate(candidate_face_tensor_list):
                img_tensor = self.full_transform(align_face_tensor_img)
                
                # === batch 차원 추가 ===
                img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
                #img_tensor = img_tensor.contiguous()
                
                out = self.analysis_full_engine(img_tensor)[0]
                out = out[0].cpu().detach().numpy()
                scores_dict = face_match_score(self.target_feature["full"], out)

                for target_id, score in scores_dict.items():
                    result_dict = {
                        'targetId': target_id,
                        'accuracy': float(score),
                        'points': candidate_bbox_list[idx]
                    }            
                    outputs.append(result_dict)
        
        if len(self.target_feature["half"]) > 0:
            # for feature_dict in self.target_feature["full"]:
            #     for target_id, feature in feature_dict.items():
            for idx, align_face_tensor_img in enumerate(candidate_half_face_list):
                img_tensor = self.half_transform(align_face_tensor_img)
                
                # === batch 차원 추가 ===
                img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
                #img_tensor = img_tensor.contiguous()
                
                out = self.analysis_half_engine(img_tensor)[0]
                out = out[0].cpu().detach().numpy()
                scores_dict = face_match_score(self.target_feature["half"], out)

                for target_id, score in scores_dict.items():
                    result_dict = {
                        'targetId': target_id,
                        'accuracy': float(score),
                        'points': candidate_bbox_list[idx]
                    }            
                    outputs.append(result_dict)
        
        return outputs

    def run_single(
        self,
        frame: np.ndarray,
        image_id: int = None,
        device_id: int = None,
        origin_size: tuple = None
    ) -> list[tuple]:
        """N프레임 얼굴 탐지 + 검증 배치 실행.

        Returns:
            list[tuple(inference_data, image_id, device_id)]
        """

        face_list = self._detect_face(frame)
        full_face_list, half_face_list, bbox_list = self._retinaface_preproc_to_second_model(frame, face_list)
        if bbox_list is not None and len(bbox_list) > 0:
            output = self._match_face_process(full_face_list, half_face_list, bbox_list)
            if origin_size is not None and tuple(frame.shape[:2]) != origin_size:
                points_list = [o["points"] for o in output]
                converted = letterbox_xyxy_to_original(points_list, origin_size, tuple(frame.shape[:2]))
                for o, c in zip(output, converted):
                    o["points"] = c
            return output, image_id, device_id
        else:
            return None, image_id, device_id

    def auto_face_detect(
        self,
        frame: np.ndarray,
        origin_size: tuple = (1080, 1920)
    ):
        
        face_list = self._detect_face(frame)
        if face_list is not None and len(face_list) > 0:
            face_boxing = [[pt if pt > 0 else 0 for pt in face.rt] for face in face_list]
            if tuple(frame.shape[:2]) != origin_size and origin_size is not None:
                 face_boxing = letterbox_xyxy_to_original(face_boxing, origin_size, tuple(frame.shape[:2]))
        else:
            face_boxing = []
            
        return face_boxing
        
        
class AreaInference:
    def __init__(
        self,
        target_type: str,
        target_feature: dict = None,
        gpu_idx: int = 0,
        batch_size: int = 1,
        cfg: dict = {},
        logger: logging.Logger = None,
    ):
        
        self.logger = logger or logging.getLogger(__name__)
        
        torch.cuda.set_device(gpu_idx)
        self.device = torch.device(f"cuda:{gpu_idx}")
        

        type_cfg = cfg["type"].get(str(target_type))
        if type_cfg is None:
            self.logger.warning("!!!!!!!!!!!!!!!!!!!!!!")
        
        self.target_type = target_type
        self.image_size = cfg["image_size"]["area_frame"]
        self.target_feature = target_feature
        #self.origin_size = origin_size  # (H, W) 원본 프레임 크기 — bbox 좌표 역변환용
        self.threshold = getattr(type_cfg, "conf_thresh", 0.5) if type_cfg else 0.5
        self.iou = getattr(type_cfg, "iou_thresh",  0.5) if type_cfg else 0.5
        self.max_det = cfg["yolo"]["max_det"]
        self.stride = cfg["yolo"]["stride"]
        self.batch_size = batch_size
        self.segment_model = SegDetectEngineTRT_v2(cfg["model_path"]["area_detect"], device=self.device)
    
    def _segment(self, img_bgr:np.ndarray) -> torch.Tensor:
        try:
            orig_h, orig_w = img_bgr.shape[:2]

            # numpy -> torch (HWC uint8) on GPU
            im = torch.from_numpy(img_bgr).to(self.device, non_blocking=True)

            # HWC -> CHW, fp16, 0~1
            im = im.permute(2, 0, 1).contiguous()          # (3,H,W)
            im = im.half().div_(255.0)
            
            # letterbox (torch)
            im = letterbox_torch(
                img=im,
                new_shape=self.image_size,
                stride=self.stride,
                auto=False
            )[0]                                           # (3,H_in,W_in)

            # BGR->RGB (torch에서 채널 스왑)
            im = im[[2, 1, 0], ...]                         # (3,H_in,W_in)

            # batch dim
            im = im.unsqueeze(0)                            # (1,3,H_in,W_in)

            # yolo inference
            pred, proto = self.segment_model(im)
            pred = non_max_suppression(pred, self.threshold, self.iou, self.target_feature["classes"], False, max_det=self.max_det, nm=32)

            if pred is None or len(pred) == 0:
                return None
            
            detected_object_list = list()
            
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    scores = det[:, 4].cpu().numpy().tolist()
                    classes = det[:, 5].cpu().numpy().tolist()
                    segments = [scale_segments((im.shape[2],im.shape[3]), x, img_bgr.shape, normalize=False)
                                for x in reversed(masks2segments(masks))]
                    det_seg = list(zip(scores, classes, segments))
                    detected_object_list.append(det_seg)
        
        except Exception as e:
            self.logger.error(f"{self.target_type} : Segment Error => {e}")
        return detected_object_list

    def _match_segment(self, detected_object_list, frame_bgr, origin_image_size=(1080, 1920)):
        outputs = []
        if len(detected_object_list) == 0 or detected_object_list is None:
            return outputs
        else:
            #try:
            for det_seg in detected_object_list:
                for conf, cls_idx, segment in det_seg:
                    seg_orig = letterbox_points_to_original(
                                                            segment,                       # (M,2) : 바로 이걸 넣는다
                                                            orig_shape=origin_image_size,   # 원본 이미지 크기
                                                            input_shape=tuple(frame_bgr.shape[:2])       # letterbox 입력 크기 (예: 480,640)
                                                        )
                    if len(seg_orig) < 4:
                        self.logger.warning(f"points len should over 4 but now {len(segment)}")
                        continue
                    segment_polygon = Polygon(seg_orig)
                    if not segment_polygon.is_valid:         #! 문제가 있는 폴리곤 수리
                        segment_polygon = segment_polygon.buffer(0)
                    draw_poly = np.array(seg_orig, dtype=np.int32)
                    for roi in self.target_feature["points"]:
                        if roi.intersects(segment_polygon):
                            result_dict = {
                                'points' : draw_poly.tolist(),
                                'class' : int(cls_idx)
                            }
                            outputs.append(result_dict)
            #except Exception as e:
            #    self.logger.error(f"{self.target_type} : Analysis Error => {e}")
            return outputs

    def run_single(
        self,
        frame: np.ndarray,
        image_id: list = None,
        device_id: list = None,
        origin_size: tuple = (1920, 1080)
    ) -> list[tuple]:
        
        
        segmented_object_list = self._segment(frame)
        output = self._match_segment(segmented_object_list, frame, origin_size)
        if output is not None and len(output) > 0:
            return output, image_id, device_id
        else:
            return None, image_id, device_id     
                
                
# class AutoDetect:
#     def __init__(
#         self,
#         detect_engine_path: str,
#         target_type: str,
#         image_size: tuple,
#         gpu_idx: int = 0,
#         batch_size: int = 1,
#         origin_size: tuple = None,
#         cfg_path: str = "",
#         logger: logging.Logger = None
#     ):
#         torch.cuda.set_device(gpu_idx)
#         self.device = torch.device(f"cuda:{gpu_idx}")
        
#         with open(cfg_path, "r") as f:
#             cfg = to_dotdict(yaml.safe_load(f))

#         self.logger = logger or logging.getLogger(__name__)
        
#         type_cfg = getattr(cfg.type, target_type, None)
#         if type_cfg is None:
#             self.logger.warning("!!!!!!!!!!!!!!!!!!!!!!")
        
#         self.target_type = target_type
#         self.image_size = image_size
#         self.origin_size = origin_size  # (H, W) 원본 프레임 크기 — bbox 좌표 역변환용
#         self.threshold = getattr(type_cfg, "conf_thresh", 0.5) if type_cfg else 0.5
#         self.iou = getattr(type_cfg, "iou_thresh",  0.5) if type_cfg else 0.5
#         self.max_det = cfg.yolo.max_det
#         self.stride = cfg.yolo.stride
#         self.batch_size = batch_size
        
#         self.logger.info(f"[AreaSingle] detect engine: {detect_engine_path}")
#         self.detect_engine = TRTBatchEngine(detect_engine_path, device=self.device)
        
#         if target_type == "face":
#             from core.util.retinaface import RetinaFace
#             self.retina = RetinaFace(model_input_size=self.detect_image_size, device=self.device)
            
    
#     def _detect(self, img_bgr: np.ndarray, model=None) -> torch.Tensor:
#         """
#         img_bgr: numpy (H,W,3) uint8 BGR
#         return: det (torch.Tensor, Nx6) on GPU, in original image coords (xyxy conf cls)
#                 없으면 None
#         """
#         try:
#             orig_h, orig_w = img_bgr.shape[:2]

#             # ① BGR→RGB numpy 단계 처리 (CPU view, 복사 없음)
#             #    + ascontiguousarray로 contiguous 보장
#             img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])  # (H,W,3) RGB

#             # ② H2D (RGB로 올림 → GPU 채널 스왑 커널 제거)
#             im = torch.from_numpy(img_rgb).to(self.device, non_blocking=True)  # (H,W,3)

#             # ③ HWC→CHW + fp16 + 0~1 정규화
#             #    .contiguous() 제거 (half()가 내부적으로 처리)
#             im = im.permute(2, 0, 1).half().div_(255.0)   # (3,H,W)

#             # ④ letterbox_torch (BGR→RGB 이전에 처리 → im[[2,1,0]] 제거)
#             im = letterbox_torch(
#                 img=im,
#                 new_shape=self.image_size,
#                 stride=self.stride,
#                 auto=False
#             )[0]                                           # (3,H_in,W_in) RGB

#             # ⑤ im[[2,1,0]] 완전 제거 (① 에서 이미 RGB 처리 완료)

#             # ⑥ batch dim 추가
#             im = im.unsqueeze(0)                           # (1,3,H_in,W_in)

#             # ⑦ 추론
#             if model is None:
#                 model = self.detect_model

#             pred = model(im)
#             pred = non_max_suppression(
#                 pred,
#                 self.threshold,
#                 self.iou,
#                 self.detect_class_filster,
#                 False,
#                 max_det=self.max_det
#             )

#             det = pred[0]
#             if det is None or len(det) == 0:
#                 return None

#             # ⑧ scale boxes → original coords
#             in_h, in_w = im.shape[-2:]
#             det[:, :4] = scale_boxes(
#                 (in_h, in_w), det[:, :4], (orig_h, orig_w)
#             ).round()

#         except Exception as e:
#             self.logger.error(f"{self.target_type} : Detect Error => {e}")
#             return None

#         return det  # GPU tensor (Nx6)
    
#     def _auto_face_detect(self, frame_bgr, origin_image_size=(1080, 1920)):
#         img_new, pad_left, pad_top, ratio = self.retina.letterbox(frame_bgr)
        
#         # img = np.float32(img_new)
#         # img -= (104, 117, 123)
#         # img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,h,w), uint8
        
#         img_tensor = torch.from_numpy(img_new).to(self.device, non_blocking=True)
#         img_tensor = img_tensor.permute(2, 0, 1).float()
#         img_tensor.sub_(torch.tensor([104, 117, 123], device=self.device).view(3,1,1))
#         img_tensor = img_tensor.unsqueeze(0).contiguous()
        
#         scale = torch.Tensor([img_new.shape[1], img_new.shape[0], img_new.shape[1], img_new.shape[0]])
#         scale = scale.to(self.device)
        
#         # img_tensor = img_tensor.to(self.device, non_blocking=True)  # (3,h,w) on GPU
#         # img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
        
#         output = self.detect_model(img_tensor) # engine model
#         loc, conf, landms = output[0][0], output[1][0], output[2][0]
        
#         dets = self.retina.retina_postprocess(img_tensor, loc, conf, landms, scale, 1)
#         list_OMS_Face = self.retina.point_post_process(dets, ratio, pad_left, pad_top)
        
#         if list_OMS_Face is not None or len(list_OMS_Face) > 0:
#             face_boxing = [[pt if pt > 0 else 0 for pt in face.rt] for face in list_OMS_Face]
#             # if not (origin_image_size == self.origin_image_size and tuple(frame_bgr.shape[:2]) == self.origin_image_size):
#             #     face_boxing = letterbox_xyxy_to_original(face_boxing, origin_image_size, tuple(frame_bgr.shape[:2]))
#         else:
#             face_boxing = []
        
#         return face_boxing
    
#     def _post_process_auto_detect(self, img_bgr, det):
        
#         if det is None or len(det) == 0:
#             return None

#         H, W = img_bgr.shape[:2]
#         det_cpu = det.detach().to("cpu")

#         boxes = []
#         confs = []
#         clses = []

#         for *xyxy, conf, cls in det_cpu.tolist():
#             if cls == 0:
#                 x1, y1, x2, y2 = map(int, xyxy)

#                 # clamp
#                 x1 = max(0, min(x1, W - 1))
#                 x2 = max(0, min(x2, W))
#                 y1 = max(0, min(y1, H - 1))
#                 y2 = max(0, min(y2, H))
#                 if x2 <= x1 or y2 <= y1:
#                     continue
                
#                 boxes.append([x1, y1, x2, y2])
#                 confs.append(float(conf))
#                 clses.append(int(cls))

#         if len(boxes) == 0:
#             return None

#         meta = {
#             "boxes": boxes, 
#             "cls":   clses
#         }
        
#         return meta
    
#     def run_auto(
#         self,
#         frame: np.ndarray,
#         image_id: int = None,
#         device_id: int = None,
#     ) -> list[list]:
        
#         if self.target_type in ("person", "attribute"):
#             det = self._detect(frame)
#             meta = self._post_process_auto_detect(frame, det)
#             boxes = meta["boxes"]
#         else:
#             boxes = self._auto_face_detect(frame)
        
#         return boxes


class ExportVideo:
    def __init__(
        self,
        detect_engine_path: str,
        target_type: str,
        image_size: tuple,
        target_feature: dict = None,
        gpu_idx: int = 0,
        batch_size: int = 1,
        origin_size: tuple = None,
        cfg_path: str = "",
        logger: logging.Logger = None
    ):

        torch.cuda.set_device(gpu_idx)
        self.device = torch.device(f"cuda:{gpu_idx}")
        
        with open(cfg_path, "r") as f:
            cfg = to_dotdict(yaml.safe_load(f))
        
        self.logger = logger or logging.getLogger(__name__)
        
        type_cfg = getattr(cfg.type, target_type, None)
        if type_cfg is None:
            self.logger.warning("!!!!!!!!!!!!!!!!!!!!!!")
        
        self.target_type = target_type
        self.image_size = image_size
        self.origin_size = origin_size  # (H, W) 원본 프레임 크기 — bbox 좌표 역변환용
        self.threshold = getattr(type_cfg, "conf_thresh", 0.5) if type_cfg else 0.5
        self.iou = getattr(type_cfg, "iou_thresh",  0.5) if type_cfg else 0.5
        self.max_det = cfg.yolo.max_det
        self.stride = cfg.yolo.stride
        self.batch_size = batch_size
        self.target_feature = target_feature
        
        if "head" in target_feature["masking"]:
            self.head_cfg = getattr(cfg.type, "head", None)
            self.head_detect_model = TRTBatchEngine(cfg.model_path.head_detect, device=self.device)
        
        if "carplate" in target_feature["masking"]:
            self.carplate_detect_model = TRTBatchEngine(cfg.model_path.single_plate_detect, device=self.deive)
            self.carplate_cfg = getattr(cfg.type, "carplate", None)
    
    def _detect(self, img_bgr:np.ndarray, model=None) -> torch.Tensor:
        """
        img_bgr: numpy (H,W,3) uint8 BGR
        return: det (torch.Tensor, Nx6) on GPU, in original image coords (xyxy conf cls)
                없으면 None
        """
        try:
            orig_h, orig_w = img_bgr.shape[:2]

            # numpy -> torch (HWC uint8) on GPU
            im = torch.from_numpy(img_bgr).to(self.device, non_blocking=True)

            # HWC -> CHW, fp16, 0~1
            #im = im.permute(2, 0, 1).contiguous()          # (3,H,W)
            im = im.permute(2, 0, 1).half().div_(255.0)

            # letterbox (torch)
            im = letterbox_torch(
                img=im,
                new_shape=self.image_size,
                stride=self.stride,
                auto=False
            )[0]                                           # (3,H_in,W_in)

            # BGR->RGB (torch에서 채널 스왑)
            im = im[[2, 1, 0], ...]                         # (3,H_in,W_in)

            # batch dim
            im = im.unsqueeze(0)                            # (1,3,H_in,W_in)

            
            # 특정 모델 주입해서 사용 하는경우 해당 모델
            if model is None:
                model = self.detect_model
            
            # yolo inference
            pred = model(im)
            pred = non_max_suppression(pred, self.threshold, self.iou, self.detect_class_filster, False, max_det=self.max_det)

            # 단일 이미지 기준이므로 pred[0]만 처리
            det = pred[0]
            self.logger.info(
                f"[detect_diag] orig=({orig_h},{orig_w}) infer={tuple(im.shape[-2:])} "
                f"thresh={self.threshold} iou={self.iou} "
                f"det_count={0 if det is None or len(det)==0 else len(det)}"
            )
            if det is None or len(det) == 0:
                return None

            # model input size (letterbox 후)
            in_h, in_w = im.shape[-2:]

            # scale boxes -> original coords
            det[:, :4] = scale_boxes((in_h, in_w), det[:, :4], (orig_h, orig_w)).round()
            for di in range(len(det)):
                self.logger.info(
                    f"[detect_diag] det[{di}] box={det[di, :4].tolist()} "
                    f"conf={det[di, 4]:.3f} cls={int(det[di, 5])}"
                )
        except Exception as e:
            self.logger.error(f"{self.target_type} : Detect Error => {e}")
            return None

        return det  # GPU tensor (Nx6)

    def _common_preproc(self, det, img_bgr):
        t0 = time.perf_counter()

        if det is None or len(det) == 0:
            return None, None

        H, W = img_bgr.shape[:2]

        x1 = det[:, 0].clamp(0, W - 1).int()
        y1 = det[:, 1].clamp(0, H - 1).int()
        x2 = det[:, 2].clamp(0, W).int()
        y2 = det[:, 3].clamp(0, H).int()

        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]

        if len(x1) == 0:
            return None, None

        t1 = time.perf_counter()

        boxes = torch.stack([x1, y1, x2, y2], dim=1).cpu().tolist()
        boxes = [[int(v) for v in b] for b in boxes]

        t2 = time.perf_counter()

        #img_rgb = img_bgr[:, :, ::-1]
        #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        crops = []
        valid_boxes = []

        crop_extract_time = 0.0
        contiguous_time = 0.0

        for bx1, by1, bx2, by2 in boxes:
            s0 = time.perf_counter()
            crop = img_bgr[by1:by2, bx1:bx2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            s1 = time.perf_counter()

            if crop.size == 0:
                continue

            #crop = np.ascontiguousarray(crop)
            #s2 = time.perf_counter()

            crops.append(crop)
            valid_boxes.append([bx1, by1, bx2, by2])

            crop_extract_time += (s1 - s0)

        t3 = time.perf_counter()

        self.logger.info(
            f"[common_preproc] "
            f"box_filter:{(t1-t0)*1000:.3f}ms "
            f"boxes_cpu:{(t2-t1)*1000:.3f}ms "
            f"crop_extract:{crop_extract_time*1000:.3f}ms "
            #f"contiguous:{contiguous_time*1000:.3f}ms "
            f"total:{(t3-t0)*1000:.3f}ms "
            f"crops:{len(valid_boxes)}"
        )

        if len(crops) == 0:
            return None, None

        return crops, {"boxes": valid_boxes}
    
    
    def run_auto(
        self,
        frame_bgr: np.ndarray,
        origin_image_size=(1080, 1920)
    ) -> list[list]:
        feature_map = {
                        "head": (
                            "head_boxes",
                            getattr(self, "head_detect_model", None)
                        ),
                        "carplate": (
                            "carplate_boxes",
                            getattr(self, "carplate_detect_model", None)
                        ),
                    }

        result = {key: [] for key, _ in feature_map.values()}

        for feature, (result_key, model) in feature_map.items():
            if feature in self.target_feature["masking"] and model is not None:
                
                if feature == "head" and self.head_cfg is not None:
                    self.threshold = self.head_cfg.conf_thresh
                    self.iou = self.head_cfg.iou_thresh

                elif feature == "carplate" and self.carplate_cfg is not None:
                    self.threshold = self.carplate_cfg.conf_thresh
                    self.iou = self.carplate_cfg.iou_thresh
                
                det = self.detect(frame_bgr, model=model)
                meta = self.post_process_auto_detect(frame_bgr, det)
                if meta is not None:    
                    boxes = letterbox_xyxy_to_original(
                        meta['boxes'],
                        origin_image_size,
                        tuple(frame_bgr.shape[:2])
                    )
                    result[result_key] = boxes

        return result