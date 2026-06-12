import os
import sys
import json
import argparse
import cv2
import torch

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

cv2.setNumThreads(2)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from core.analysis import BatchInference, FaceInference, AreaInference, ExportVideo
from core.target import BatchTarget

CONFIG_PATH = os.path.join("../", "prod_cfg.yaml")
DEFAULT_BATCH_SIZE = 4

# Batch Enine 용 Model Inferecne Adapter
class BatchEngineAdapter:
    def __init__(self, engine, detect_type: str):
        self.engine = engine
        self.detect_type = detect_type
        
    def run_frames(self, frames, frame_ids, device_ids, origin_sizes):
        
        return self.engine.run_batch(
            frames,
            frame_ids,
            device_ids,
            origin_sizes
        )
        

class SinleEngineAdapter:
    def __init__(self, engine, detect_type: str):
        self.engine = engine
        self.detect_type = detect_type
        
    def run_frames(self, frames, frame_ids, device_ids, origin_sizes):
        results = []
        
        for frame, frame_id, device_id, origin_size in zip(frames, frame_ids, device_ids, origin_sizes):
            
            run_result = self.engine.run_single(
                frame, frame_id, device_id, origin_size
            )
            
            if isinstance(run_result, tuple) and len(run_result) == 3:
                inference_data, image_idx, result_device_id = run_result
            else:
                raise RuntimeError(
                    f"invalid run_single return format: {run_result}"
                )

            results.append((inference_data or [], image_idx, result_device_id))

        return results
            
    

class VideoInferenceTester:
    def __init__(
        self,
        detect_type: str,
        video_path: str,
        target_json_path: str = "",
        output_path: str = "/tmp/ai-core-video-test.mp4",
        gpu_idx: int = 0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_frames: int = 0,
    ):
        self.detect_type = detect_type
        self.video_path = video_path
        self.target_json_path = target_json_path
        self.output_path = output_path
        self.gpu_idx = gpu_idx
        self.batch_size = batch_size
        self.max_frames = max_frames

        self.cfg = self._load_cfg()
        self.target_feature = self._load_target_feature()
        self.engine = self._create_engine()

        self.total_frames = 0
        self.total_result_count = 0
        self.result_frames = 0
        self.cfg = self._load_cfg()
        
        if self.detect_type in ["person", "attribute", "carplate"]:
            self.engine_adapter = BatchEngineAdapter(self.engine, detect_type)
        else:
            self.engine_adapter = SinleEngineAdapter(self.engine, detect_type)

    def _load_cfg(self):
        import yaml

        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _load_target_feature(self):
        if not self.target_json_path:
            return None

        return BatchTarget(
            target_type=self.detect_type,
            gpu_idx=self.gpu_idx,
            cfg=self.cfg,
            log=None,
        ).set_target(self.target_json_path)

    # 모델 input 용 size로 frame resize 하는 함수
    def _resize_keep_ratio(self, frame, max_size):
        if max_size is None:
            return frame

        max_h, max_w = max_size
        h, w = frame.shape[:2]

        scale = min(max_w / w, max_h / h, 1.0)

        if scale >= 1.0:
            return frame

        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _create_engine(self):
        cfg = self.cfg
        type_cfg = cfg["type"].get(self.detect_type, {})

        threshold = type_cfg.get("conf_thresh", 0.5)
        iou = type_cfg.get("iou_thresh", 0.45)
        stride = cfg["yolo"]["stride"]
        max_det = cfg["yolo"]["max_det"]

        if self.detect_type == "person":
            self.image_size = tuple(cfg["image_size"]["person_frame"])
            analysis_batch_size = self.batch_size
        elif self.detect_type == "attribute":
            self.image_size = tuple(cfg["image_size"]["person_frame"])
            analysis_batch_size = 8
        elif self.detect_type == "carplate":
            self.image_size = tuple(cfg["image_size"]["plate_frame"])
            analysis_batch_size = 32
        elif self.detect_type == "face":
            self.image_size = tuple(cfg["image_size"]["face_frame"])
        elif self.detect_type == "area":
            self.image_size = tuple(cfg["image_size"]["area_frame"])
        else:
            raise ValueError(f"지원하지 않는 BatchInference type: {self.detect_type}")
        
        if self.detect_type == "face":
            return FaceInference(
                target_feature=self.target_feature,
                gpu_idx=self.gpu_idx,
                detect_batch_size=self.batch_size,
                analysis_batch_size=32,
                vis_thresh=type_cfg.get("conf_thresh", 0.9),
                nms_thresh=type_cfg.get("iou_thresh", 0.4),
                score_threshold=0.5,
                cfg=cfg,
                logger=None,
            )
        
        if self.detect_type == "area":
            return AreaInference(
                target_type=self.detect_type,
                target_feature=self.target_feature,
                gpu_idx=self.gpu_idx,
                batch_size=1,
                cfg=cfg,
                logger=None,
            )

        return BatchInference(
            target_type=self.detect_type,
            target_feature=self.target_feature,
            gpu_idx=self.gpu_idx,
            batch_size=self.batch_size,
            analysis_batch_size=analysis_batch_size,
            cfg=cfg,
            logger=None,
        )

    
    # 테스트 메인 함수
    def run(self) -> bool:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"영상 파일을 열 수 없습니다: {self.video_path}")

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # if fps <= 0:
        #     fps = 30

        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # writer = cv2.VideoWriter(
        #     self.output_path,
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     (width, height),
        # )

        frames = []
        frame_ids = []
        device_ids = []
        origin_sizes = []

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if self.max_frames > 0 and frame_idx >= self.max_frames:
                break
            
            origin_size = frame.shape[:2]
            
            
            infer_frame = self._resize_keep_ratio(frame, self.image_size)
            
            frames.append(infer_frame)
            frame_ids.append(frame_idx)
            device_ids.append(0)
            origin_sizes.append(origin_size)
            
            

            if len(frames) >= self.batch_size:
                self._process_frames(frames, frame_ids, device_ids, origin_sizes)
                frames, frame_ids, device_ids, frame_sizes = [], [], [], []

            frame_idx += 1

        if frames:
            self._process_frames(frames, frame_ids, device_ids, frame_sizes)

        cap.release()

        # passed = self.total_result_count > 0

        # summary = {
        #     "passed": passed,
        #     "type": self.detect_type,
        #     "video": self.video_path,
        #     "output": self.output_path,
        #     "totalFrames": self.total_frames,
        #     "resultFrames": self.result_frames,
        #     "totalResultCount": self.total_result_count,
        # }

        # print(json.dumps(summary, ensure_ascii=False, indent=2))

        return {
            "passed": self.total_result_count > 0,
            "type": self.detect_type,
            "result_count": self.total_result_count,
            "error": None
            
        }
        

    # AI 분석 결과 후보 수 누적하여 합산하는 함수
    def _process_frames(self, frames, frame_ids, device_ids, origin_sizes):

        results = self.engine_adapter.run_frames(
            frames, frame_ids, device_ids, origin_sizes
        )
        
        
        
        if results is None:
            raise RuntimeError("engine returned None")

        if len(results) != len(frames):
            raise RuntimeError(
                f"engine result size mismatch: input={len(frames)} output={len(results)}"
            )

        frame_result_count = 0

        for i, result in enumerate(results):
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(
                    f"invalid engine result format at index={i}: {result}"
                )

            inference_data, image_idx, device_id = result

            if inference_data is None:
                count = 0
            elif isinstance(inference_data, list):
                count = len(inference_data)
            else:
                raise RuntimeError(
                    f"invalid inference_data type at index={i}: "
                    f"{type(inference_data).__name__}"
                )

            self.total_frames += 1
            self.total_result_count += count
            frame_result_count += count

            if count > 0:
                self.result_frames += 1

        return frame_result_count

    #? 현재 시각화 불필요
    # def _draw_results(self, frame, inference_data, frame_id: int, count: int):
    #     out = frame.copy()

    #     for item in inference_data:
    #         if not isinstance(item, dict):
    #             continue

    #         points = item.get("points")
    #         if points is None:
    #             continue

    #         if hasattr(points, "tolist"):
    #             points = points.tolist()

    #         if len(points) != 4:
    #             continue

    #         x1, y1, x2, y2 = [int(v) for v in points]

    #         label = self._make_label(item)

    #         cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(
    #             out,
    #             label,
    #             (x1, max(y1 - 8, 0)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             (0, 255, 0),
    #             2,
    #             cv2.LINE_AA,
    #         )

    #     status = f"type={self.detect_type} frame={frame_id} count={count} total={self.total_result_count}"
    #     cv2.putText(
    #         out,
    #         status,
    #         (20, 35),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.8,
    #         (0, 255, 255),
    #         2,
    #         cv2.LINE_AA,
    #     )

    #     return out

    def _make_label(self, item: dict) -> str:
        label_parts = []

        if "targetId" in item:
            label_parts.append(str(item["targetId"]))

        if "accuracy" in item:
            try:
                label_parts.append(f'{float(item["accuracy"]):.2f}')
            except Exception:
                pass

        if "cls" in item:
            label_parts.append(str(item["cls"]))

        if "text" in item:
            label_parts.append(str(item["text"]))

        return " ".join(label_parts) if label_parts else self.detect_type


def main():
    parser = argparse.ArgumentParser(description="Video BatchInference simple tester")
    parser.add_argument("--type", choices=["person", "attribute", "face", "carplate", "area"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--target_json_path", default="")
    parser.add_argument("--output", default="/tmp/ai-core-video-test.mp4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    tester = VideoInferenceTester(
        detect_type=args.type,
        video_path=args.video,
        target_json_path=args.target_json_path,
        output_path=args.output,
        gpu_idx=args.gpu,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )

    try:
        summary = tester.run()
        print(json.dumps(summary, ensure_ascii=False), flush=True)

        if summary["result_count"] > 0:
            raise SystemExit(0)

        raise SystemExit(1)

    except SystemExit:
        raise
    
    except Exception as e:
        print(json.dumps({
            "passed": False,
            "type": args.type,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            },
        }, ensure_ascii=False), flush=True)

        raise SystemExit(2)


if __name__ == "__main__":
    main()