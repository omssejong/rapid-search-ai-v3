import os, sys, argparse, logging
from logging.handlers import RotatingFileHandler, SysLogHandler
import threading, queue

# 멀티프로세스 환경 CPU 스레드 폭발 방지
# torch/numpy import 전에 설정해야 스레드 풀 초기화에 반영됨
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json as _json_mod
import math as _math_mod

def _sanitize_for_json(obj):
    """NaN/Infinity를 JSON 호환 값으로 치환 (재귀)."""
    if isinstance(obj, float):
        if _math_mod.isnan(obj) or _math_mod.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj

try:
    import orjson
    def _json_dumps(obj):
        return or_json_mod.dumps(_sanitize_for_json(obj))
except ImportError:
    def _json_dumps(obj):
        return _json_mod.dumps(_sanitize_for_json(obj)).encode("utf-8")

import cv2, struct
import numpy as np
import torch

# cv2 스레드 제한 (import 후 설정)
cv2.setNumThreads(2)
# torch CPU 스레드 제한 (환경변수와 별개로 명시적 설정)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# 추론 결정성 보장: 같은 입력 → 같은 출력
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from pkg.unix_socket import UnixSocketSender
from contextlib import contextmanager

from pkg.shm_reader import SharedMemoryReader, ShmSlotReader, ShmErrorCode

# 로깅 설정
_LOG_DIR = "/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/logs"
os.makedirs(_LOG_DIR, exist_ok=True)

def _setup_logger(name: str, gpu_idx: int = 0) -> logging.Logger:
    """프로세스별 로그 파일을 생성하는 로거 설정.

    로그 파일: {_LOG_DIR}/ai-core-{name}-gpu{gpu_idx}.log
    최대 50MB × 3개 로테이션 유지.
    """
    logger = logging.getLogger(f"ai-core-{name}-{gpu_idx}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    log_file = os.path.join(_LOG_DIR, f"ai-core-{name}-gpu{gpu_idx}.log")
    fh = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    # syslog: CRITICAL 에러만 시스템 로그에 기록
    try:
        syslog_handler = SysLogHandler(address='/dev/log', facility=SysLogHandler.LOG_DAEMON)
        syslog_handler.setLevel(logging.CRITICAL)
        syslog_handler.setFormatter(logging.Formatter("ai-core[%(process)d]: %(message)s"))
        logger.addHandler(syslog_handler)
    except Exception:
        pass  # syslog 미사용 환경에서도 정상 동작

    return logger

log: logging.Logger = None
#ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = "/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai"
sys.path.insert(0, ROOT)
from lib import Total_AI  #, Target  # Target → BatchTarget로 교체
from pkg.batch_engine import BatchInference, FaceBatchInference
from pkg.batch_target import BatchTarget
import torch

config_path = os.path.join(ROOT, "prod_cfg.yaml")
ai_core: Total_AI = None

# detect_type → 비율 유지 리사이즈 상한 크기 (h, w)
# origin_frame과 동일하면 리사이즈 불필요 → None
def _get_max_size(detect_type: str):
    """prod_cfg.yaml image_size 기반 detect_type별 리사이즈 상한 크기 반환"""
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    origin = tuple(cfg["image_size"]["origin_frame"])  # (1080, 1920)
    _map = {
        "person":      tuple(cfg["image_size"]["person_frame"]),
        "attribute":   tuple(cfg["image_size"]["person_frame"]),
        "area":        tuple(cfg["image_size"]["area_frame"]),
        "carplate":    tuple(cfg["image_size"]["plate_frame"]),
        "face":        tuple(cfg["image_size"]["face_frame"]),
        "exportvideo": None,
    }
    size = _map.get(detect_type)
    if size is None or size == origin:
        return None
    return size

# 백그라운드 디스크 I/O 라이터 — 추론 루프 블로킹 방지
_write_queue = queue.Queue(maxsize=1024)

def _bg_writer():
    """백그라운드 스레드: 큐에서 (path, data, event) 꺼내서 디스크에 저장"""
    while True:
        item = _write_queue.get()
        if item is None:
            break
        path, data, done_event = item
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            pass
        finally:
            if done_event is not None:
                done_event.set()

_writer_thread = threading.Thread(target=_bg_writer, daemon=True)
_writer_thread.start()

def save_frame_async(path: str, data: bytes) -> threading.Event:
    """프레임을 백그라운드 스레드로 비동기 저장. 완료 Event 반환."""
    done = threading.Event()
    try:
        _write_queue.put_nowait((path, data, done))
    except queue.Full:
        done.set()  # 큐 가득 차면 드롭하되 Event는 set (대기 무한블로킹 방지)
    return done

# 외부 라이브러리 출력 억제
@contextmanager
def quiet_mode():
    """외부 라이브러리 출력만 억제"""
    with open(os.devnull, 'w') as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

# input: image output: target crop data list
def read_target_data(image_shm_key: str, image_shm_idx: int, detect_type: str, batch_engine=None):
    """SHM에서 프레임 1장 읽기 → detect → stdout JSON 출력.

    batch_engine이 주어지면 BatchInference/FaceBatchInference의 detect 파이프라인 사용.
    없으면 레거시 Total_AI 경로 (하위 호환).
    """
    shm_image = SharedMemoryReader(image_shm_key)
    image = shm_image.read_frame_safe(image_shm_idx)
    if image is None:
        sys.exit(3)
    shm_image.set_read_index(image_shm_idx + 1)
    np_arr = np.frombuffer(image, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        log.error(f"[detect] cv2.imdecode failed, raw_size={len(np_arr)}")
        sys.exit(4)
    log.info(f"[detect] frame decoded: shape={frame_bgr.shape} dtype={frame_bgr.dtype}")
    shm_image.close()

    if batch_engine is not None and detect_type in ("person", "attribute"):
        # BatchInference 파이프라인으로 detect
        batch_tensor, orig_shapes = batch_engine._preproc_frames([frame_bgr])
        detections = batch_engine._batch_detect(batch_tensor, orig_shapes)
        det = detections[0]

        if det is None or len(det) == 0:
            log.warning(f"[detect] batch_detect returned no detections (exit 4)")
            sys.exit(4)

        det = det.float()
        # letterbox(480x640) → 원본 프레임 좌표 역변환
        from pkg.batch_engine import letterbox_xyxy_to_original
        orig_h, orig_w = frame_bgr.shape[:2]
        in_h, in_w = batch_tensor.shape[-2:]
        boxes = []
        cls_list = []
        for di in range(len(det)):
            cls_id = int(det[di, 5])
            conf = float(det[di, 4])
            raw_box = det[di, :4].tolist()
            box = letterbox_xyxy_to_original(raw_box, (orig_h, orig_w), (in_h, in_w))[0]
            log.info(f"[detect] det[{di}] box={box} conf={conf:.3f} cls={cls_id}")
            if cls_id == 0:
                boxes.append([int(v) for v in box])
                cls_list.append("PERSON")

        if len(boxes) == 0:
            log.warning(f"[detect] no person class (cls=0) in detections (exit 4)")
            sys.exit(4)

        res_dict = {"points": boxes, "verify": cls_list}
        log.info(f"[detect] result: {res_dict}")
        sys.stdout.buffer.write(_json_mod.dumps(res_dict).encode("utf-8"))
        sys.stdout.buffer.flush()
        return

    # face 또는 레거시 fallback
    if detect_type == "face" and batch_engine is not None:
        # FaceBatchInference로 detect
        batch_results = batch_engine.run_batch([frame_bgr])
        inference_data, _, _ = batch_results[0]
        if not inference_data:
            log.warning(f"[detect] face batch_detect returned no faces (exit 4)")
            sys.exit(4)
        crop_points = []
        for face_res in inference_data:
            crop_points.append(face_res["points"])
        face_detect_result = {"crop": crop_points}
        log.info(f"[detect] face result: {face_detect_result}")
        sys.stdout.buffer.write(_json_mod.dumps(face_detect_result).encode("utf-8"))
        sys.stdout.buffer.flush()
        return

    # 레거시 Total_AI fallback
    with quiet_mode():
        if detect_type == "face":
            auto_detect_res = ai_core.auto_face_detect(frame_bgr)
        else:
            auto_detect_res = ai_core.auto_detect(frame_bgr)
    if auto_detect_res is None:
        log.warning(f"[detect] auto_detect returned None (exit 4)")
        sys.exit(4)
    if detect_type == "face":
        face_detect_result = {"crop": auto_detect_res}
        sys.stdout.buffer.write(_json_mod.dumps(face_detect_result).encode("utf-8"))
        sys.stdout.buffer.flush()
        return
    res_dict = {"points": auto_detect_res["boxes"]}
    cls_list = ["PERSON" for cls in auto_detect_res["cls"] if cls == 0]
    res_dict["verify"] = cls_list
    sys.stdout.buffer.write(_json_mod.dumps(res_dict).encode("utf-8"))
    sys.stdout.buffer.flush()

BATCH_SIZE = 4  # 배치 추론 크기

def _create_batch_engine(detect_type: str, target_feature: dict, gpu_idx: int) -> BatchInference:
    """prod_cfg.yaml 기반 BatchInference 인스턴스 생성."""
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    type_cfg = cfg["type"].get(detect_type, {})
    threshold = type_cfg.get("conf_thresh", 0.5)
    iou = type_cfg.get("iou_thresh", 0.45)
    stride = cfg["yolo"]["stride"]
    max_det = cfg["yolo"]["max_det"]

    # detect_type별 엔진 경로 & 이미지 크기
    if detect_type == "face":
        detect_path = cfg["model_path"]["face_detect"]
        analysis_path = cfg["model_path"]["face_full_analysis"]
        face_size = tuple(cfg["image_size"]["face_frame"])
        return FaceBatchInference(
            detect_engine_path=detect_path,
            analysis_engine_path=analysis_path,
            detect_image_size=face_size,
            target_feature=target_feature,
            gpu_idx=gpu_idx,
            detect_batch_size=BATCH_SIZE,
            analysis_batch_size=32,
            vis_thresh=type_cfg.get("conf_thresh", 0.9),
            nms_thresh=type_cfg.get("iou_thresh", 0.4),
            score_threshold=0.5,
            logger=log,
        )
    elif detect_type == "person":
        detect_path = cfg["model_path"]["person_detect"]
        analysis_path = cfg["model_path"]["person_analysis"]
        image_size = tuple(cfg["image_size"]["person_frame"])
    elif detect_type == "attribute":
        detect_path = cfg["model_path"]["attribute_detect"]
        analysis_path = cfg["model_path"]["attribute_analysis"]
        image_size = tuple(cfg["image_size"]["person_frame"])
    elif detect_type == "carplate":
        detect_path = cfg["model_path"]["plate_detect"]
        analysis_path = cfg["model_path"]["plate_ocr_analysis"]
        image_size = tuple(cfg["image_size"]["plate_frame"])
    else:
        raise ValueError(f"BatchInference 미지원 타입: {detect_type}")

    # RTX 4080 벤치마크 최적 배치 (타입별)
    # ┌──────────┬──────────────────┬──────────┬─────────┐
    # │  타입    │     모델         │ 최적배치 │  비고   │
    # ├──────────┼──────────────────┼──────────┼─────────┤
    # │ person   │ YOLO detect      │    4     │ opt=4   │
    # │ person   │ ReID analysis    │    4     │         │
    # │ attr     │ YOLO detect(공용)│    4     │ opt=4   │
    # │ attr     │ ConvFormer       │    8     │ 효율75% │
    # │ carplate │ YOLO detect      │    4     │         │
    # │ carplate │ Plate OCR        │   32     │ FPS=4924│
    # └──────────┴──────────────────┴──────────┴─────────┘
    # detect_batch: person_detect 엔진은 opt=4 프로파일 — person/attribute 공용이므로
    # detect_batch는 엔진 opt와 일치시켜야 함 (불일치 시 bbox 오탐 발생 확인됨)
    # analysis_batch: 각 analysis 엔진의 벤치마크 최적값 적용
    if detect_type == "attribute":
        detect_batch = BATCH_SIZE      # attribute_detect 엔진: opt=4, max=4 (단일 SHM 다채널)
        analysis_batch = 8             # ConvFormer 엔진: opt=8 (crop 배치)
    elif detect_type == "carplate":
        detect_batch = BATCH_SIZE      # plate_detect
        analysis_batch = 32            # Plate OCR: FPS=4924
    else:
        detect_batch = BATCH_SIZE      # person: 4
        analysis_batch = BATCH_SIZE

    origin_size = tuple(cfg["image_size"]["origin_frame"])  # (1080, 1920)

    return BatchInference(
        detect_engine_path=detect_path,
        analysis_engine_path=analysis_path,
        target_type=detect_type,
        image_size=image_size,
        target_feature=target_feature,
        gpu_idx=gpu_idx,
        threshold=threshold,
        iou=iou,
        max_det=max_det,
        stride=stride,
        batch_size=detect_batch,
        analysis_batch_size=analysis_batch,
        origin_size=origin_size,
        cfg_path=config_path,
        logger=log,
    )


def analyze_images(image_shm_key:str, detect_type:str, sock: UnixSocketSender, analyze_id: int, frame_path: str="/opt/oms/omeye/filestorage/frame", batch_engine: BatchInference = None, ai_core=None):
    import time
    max_size = _get_max_size(detect_type)
    reader = ShmSlotReader(image_shm_key, max_size=max_size)
    log.info(f"ShmSlotReader max_size={max_size} for type={detect_type}")
    if not reader.connect():
        log.error(f"SHM 연결 실패: {image_shm_key}")
        sys.exit(3)

    # face + Total_AI: 1프레임씩 처리
    use_legacy = (ai_core is not None and batch_engine is None)
    detect_batch = 1 if use_legacy else batch_engine.batch_size
    log.info(f"analyze polling started (detect_batch={detect_batch} legacy={use_legacy}): {image_shm_key}")
    get_result_count = 0
    infer_count = 0
    consecutive_errors = 0
    no_data_count = 0
    last_stats_time = time.time()
    pending_frames = []  # 배치 누적 버퍼
    while True:
        try:
            batch = reader.read_batch_ex(max_batch=detect_batch)

            if not batch:
                if pending_frames:
                    if use_legacy:
                        _process_analyze_legacy(pending_frames, ai_core, sock, analyze_id, frame_path)
                    else:
                        _process_analyze_batch(pending_frames, batch_engine, sock, analyze_id, frame_path)
                    infer_count += len(pending_frames)
                    for pf in pending_frames:
                        if pf.get('_has_result'):
                            get_result_count += 1
                    pending_frames = []

                no_data_count += 1
                if no_data_count < 10:
                    time.sleep(0.0001)
                elif no_data_count < 100:
                    time.sleep(0.001)
                else:
                    time.sleep(0.005)
                continue

            no_data_count = 0
            consecutive_errors = 0

            pending_frames.extend(batch)

            while len(pending_frames) >= detect_batch:
                batch_chunk = pending_frames[:detect_batch]
                pending_frames = pending_frames[detect_batch:]

                if use_legacy:
                    _process_analyze_legacy(batch_chunk, ai_core, sock, analyze_id, frame_path)
                else:
                    _process_analyze_batch(batch_chunk, batch_engine, sock, analyze_id, frame_path)
                infer_count += len(batch_chunk)
                for pf in batch_chunk:
                    if pf.get('_has_result'):
                        get_result_count += 1

            # --- 기존 단일 프레임 처리 코드 (주석 처리) ---
            # for result in batch:
            #     image = result['image']
            #     meta = result['metadata']
            #     image_idx = meta['frame_id']
            #     device_id = meta['device_id']
            #     video_number = meta['video_number']
            #     frame_image_idx = image_idx
            #
            #     inference_data, image_idx, device_id = ai_core.run(image, image_idx, device_id)
            #     # inference_data = None
            #     infer_count += 1
            #
            #     if inference_data is None or len(inference_data) == 0:
            #         inference_data = []
            #         frame_save_dir_path = os.path.join(frame_path, str(analyze_id), device_id, str(video_number),
            #                                            str(frame_image_idx) + ".jpg")
            #     else:
            #         frame_save_dir_path = os.path.join(frame_path, str(analyze_id), device_id, str(video_number), str(frame_image_idx) + ".jpg")
            #         save_frame_async(frame_save_dir_path, result['frame_image'])
            #         get_result_count += 1
            #
            #     for _, v in enumerate(inference_data):
            #         v["frameIdx"] = frame_image_idx
            #
            #     infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx, frame_save_dir_path)
            #     # infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx, "")
            #     sock.send(infer_res)
            # --- 기존 단일 프레임 처리 코드 끝 ---

            # 10초마다 통계 로그
            now = time.time()
            if now - last_stats_time >= 10.0:
                log.info(f"[stats] infer={infer_count} results={get_result_count} pending={len(pending_frames)} shm={image_shm_key}")
                last_stats_time = now

        except (BrokenPipeError, ConnectionError, OSError) as e:
            log.error(f"소켓/IPC 에러 — 프로세스 종료: {e}")
            break
        except Exception as e:
            log.error(f"예기치 않은 에러: {e}", exc_info=True)
            consecutive_errors += 1
            if consecutive_errors > 10:
                log.error(f"연속 에러 {consecutive_errors}회 — 프로세스 종료")
                break

    reader.disconnect()
    log.info(f"[exit] analyze loop ended. infer={infer_count} results={get_result_count}")


def _process_analyze_legacy(batch_chunk: list, ai_core_ref, sock: UnixSocketSender, analyze_id: int, frame_path: str):
    """Total_AI 1프레임 단위 추론 처리 (face 등 레거시 모델용)."""
    for result in batch_chunk:
        image = result['image']
        meta = result['metadata']
        device_id = meta['device_id']
        video_number = meta['video_number']
        frame_image_idx = meta['frame_id']
        image_idx = frame_image_idx

        try:
            run_result = ai_core_ref.run(image, image_idx, device_id)
            # run() 반환값 검증
            if isinstance(run_result, tuple) and len(run_result) == 3:
                inference_data, image_idx, device_id = run_result
            else:
                log.error(f"[legacy] run() 반환값 비정상: type={type(run_result)} value={run_result}")
                inference_data = run_result
            # log.error(f"[legacy_diag] device={device_id} frame={frame_image_idx} run_result_type={type(inference_data)} len={len(inference_data) if inference_data else 'None'} data={inference_data}")
        except Exception as e:
            log.error(f"[legacy] Total_AI.run() 에러: device={device_id} frame={frame_image_idx} => {e}", exc_info=True)
            inference_data = None

        if inference_data is None or len(inference_data) == 0:
            inference_data = []
            result['_has_result'] = False
        else:
            result['_has_result'] = True

        frame_save_dir_path = os.path.join(frame_path, str(analyze_id), str(device_id), str(video_number),
                                           str(frame_image_idx) + ".jpg")

        write_done = None
        if result.get('_has_result'):
            write_done = save_frame_async(frame_save_dir_path, result['frame_image'])

        for v in inference_data:
            if isinstance(v, dict):
                v["frameIdx"] = frame_image_idx

        # 파일 저장 완료 대기 후 전송 (race condition 방지)
        if write_done is not None:
            write_done.wait()

        infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx, frame_save_dir_path)
        sock.send(infer_res)


def _process_analyze_batch(batch_chunk: list, batch_engine_ref: BatchInference, sock: UnixSocketSender, analyze_id: int, frame_path: str):
    """BATCH_SIZE 단위 프레임 배치 추론 처리.

    BatchInference.run_batch()로 배치 추론 후, 검출 결과를 고정 스코어(50)로 전송.
    """
    frames = []
    image_ids = []
    device_ids = []
    metas = []
    frame_sizes = []

    for result in batch_chunk:
        frames.append(result['image'])
        meta = result['metadata']
        image_ids.append(meta['frame_id'])
        device_ids.append(meta['device_id'])
        frame_sizes.append((
            meta.get('orig_height', meta['frame_height']),
            meta.get('orig_width', meta['frame_width']),
        ))
        metas.append(meta)

    # # [진단] 배치 구성 로깅
    # log.info(
    #     f"[batch_map] batch_size={len(frames)} "
    #     f"devices={[m['device_id'] for m in metas]} "
    #     f"frame_ids={[m['frame_id'] for m in metas]}"
    # )

    # BatchInference 배치 추론 실행 → 매칭 결과
    batch_results = batch_engine_ref.run_batch(frames, image_ids, device_ids, frame_sizes)

    # 프레임별 결과 전송
    results_to_send = []
    for i, (inference_data, image_idx, device_id) in enumerate(batch_results):
        result = batch_chunk[i]
        meta = metas[i]
        video_number = meta['video_number']
        frame_image_idx = meta['frame_id']

        # [진단] 프레임-결과 device_id 일관성 검증
        if str(device_id) != str(meta['device_id']):
            # log.error(
            #     f"[batch_map] MISMATCH! idx={i} result_device={device_id} != "
            #     f"input_device={meta['device_id']} frame={frame_image_idx} "
            #     f"— 결과 폐기 (크로스 프레임 오염 방지)"
            # )
            inference_data = []

        # # [진단] 프레임-결과 매핑 검증
        # if inference_data:
        #     log.info(
        #         f"[batch_map] idx={i} device={meta['device_id']} "
        #         f"frame={frame_image_idx} detections={len(inference_data)}"
        #     )

        if inference_data is None:
            inference_data = []

        frame_save_dir_path = os.path.join(frame_path, str(analyze_id), str(device_id), str(video_number),
                                           str(frame_image_idx) + ".jpg")

        if len(inference_data) == 0:
            result['_has_result'] = False
        else:
            # # [디버그] Python에서 직접 1080p 원본에 bbox를 그려 저장
            # # Go 측 좌표 변환 문제 격리용
            # try:
            #     debug_img = cv2.imdecode(
            #         np.frombuffer(result['frame_image'], np.uint8),
            #         cv2.IMREAD_COLOR,
            #     )
            #     if debug_img is not None:
            #         for det in inference_data:
            #             pts = det.get("points")
            #             if pts and len(pts) == 4:
            #                 x1, y1, x2, y2 = [int(v) for v in pts]
            #                 cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #                 label = f"{det.get('targetId', '')} {det.get('accuracy', 0):.2f}"
            #                 cv2.putText(debug_img, label, (x1, max(y1 - 6, 0)),
            #                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #         _, jpeg_buf = cv2.imencode(".jpg", debug_img)
            #         save_frame_async(frame_save_dir_path, jpeg_buf.tobytes())
            #     else:
            #         save_frame_async(frame_save_dir_path, result['frame_image'])
            # except Exception:
            #     save_frame_async(frame_save_dir_path, result['frame_image'])
            result['_write_done'] = save_frame_async(frame_save_dir_path, result['frame_image'])
            result['_has_result'] = True

        for _, v in enumerate(inference_data):
            v["frameIdx"] = frame_image_idx

        infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx, frame_save_dir_path)
        results_to_send.append(infer_res)

    # 파일 저장 완료 대기 후 일괄 전송 (race condition 방지)
    for result in batch_chunk:
        write_done = result.get('_write_done')
        if write_done is not None:
            write_done.wait()

    for res in results_to_send:
        sock.send(res)


# --- 기존 단일 프레임 _process_analyze_batch (ai_core.run 사용, 주석 처리) ---
# def _process_analyze_batch(batch_chunk: list, ai_core_ref, sock: UnixSocketSender, analyze_id: int, frame_path: str):
#     """BATCH_SIZE 단위 프레임 배치 추론 처리 (단일 프레임 ai_core.run 버전)."""
#     results_to_send = []
#
#     for result in batch_chunk:
#         image = result['image']
#         meta = result['metadata']
#         image_idx = meta['frame_id']
#         device_id = meta['device_id']
#         video_number = meta['video_number']
#         frame_image_idx = image_idx
#
#         inference_data, image_idx, device_id = ai_core_ref.run(image, image_idx, device_id)
#
#         if inference_data is None or len(inference_data) == 0:
#             inference_data = []
#             frame_save_dir_path = os.path.join(frame_path, str(analyze_id), device_id, str(video_number),
#                                                str(frame_image_idx) + ".jpg")
#             result['_has_result'] = False
#         else:
#             frame_save_dir_path = os.path.join(frame_path, str(analyze_id), device_id, str(video_number),
#                                                str(frame_image_idx) + ".jpg")
#             save_frame_async(frame_save_dir_path, result['frame_image'])
#             result['_has_result'] = True
#
#         for _, v in enumerate(inference_data):
#             v["frameIdx"] = frame_image_idx
#
#         infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx, frame_save_dir_path)
#         results_to_send.append(infer_res)
#
#     # 배치 결과 일괄 전송
#     for res in results_to_send:
#         sock.send(res)
# --- 기존 단일 프레임 _process_analyze_batch 끝 ---

def area_analyze_image(image_shm_key:str, detect_type:str, sock: UnixSocketSender, analyze_id: int, frame_path: str="/opt/oms/omeye/filestorage/frame", frame_rate=6, ai_core=None):
    import time
    max_size = _get_max_size(detect_type)
    reader = ShmSlotReader(image_shm_key, max_size=max_size)
    log.info(f"ShmSlotReader max_size={max_size} for type={detect_type}")
    if not reader.connect():
        log.error(f"SHM 연결 실패: {image_shm_key}")
        sys.exit(3)

    log.info(f"area analyze polling started: {image_shm_key}")
    # [FIX-1] device별 이벤트 상태 분리 (크로스 디바이스 오염 방지)
    device_states: dict = {}  # {device_id: {"is_event": bool, "one_second_video": int, "segments": int}}
    count: int = 0
    consecutive_errors = 0
    no_data_count = 0
    while True:
        try:
            batch = reader.read_batch_ex(max_batch=8)

            if not batch:
                no_data_count += 1
                if no_data_count < 10:
                    time.sleep(0.0001)
                elif no_data_count < 100:
                    time.sleep(0.001)
                else:
                    time.sleep(0.005)
                continue

            no_data_count = 0
            consecutive_errors = 0

            for result in batch:
                count += 1
                image = result['image']
                meta = result['metadata']
                image_idx = meta['frame_id']
                device_id = meta['device_id']
                video_number = meta['video_number']
                frame_image_idx = image_idx

                # [FIX-1] device별 상태 조회/초기화
                dev_key = device_id
                if dev_key not in device_states:
                    device_states[dev_key] = {"is_event": False, "one_second_video": 0, "segments": 0}
                st = device_states[dev_key]

                origin_size = (
                    meta.get('orig_height', meta['frame_height']),
                    meta.get('orig_width', meta['frame_width']),
                )
                try:
                    inference_data, image_idx, device_id = ai_core.run(image, image_idx, device_id, origin_image_size=origin_size)
                except Exception as e:
                    log.error(f"[area] Total_AI.run() 에러: device={device_id} frame={frame_image_idx} => {e}", exc_info=True)
                    inference_data = None

                if inference_data is None or len(inference_data) == 0:
                    inference_data = []
                    if st["is_event"]:
                        if st["one_second_video"] % frame_rate != 0:
                            st["one_second_video"] += 1
                        else:
                            st["one_second_video"] = 0
                            st["is_event"] = False
                            st["segments"] += 1
                else:
                    if not st["is_event"]:
                        st["is_event"] = True
                    st["one_second_video"] += 1

                # [FIX-2] 검출 결과가 있을 때만 프레임 저장
                write_done = None
                frame_save_dir_path = area_save_frame_image(frame_path, analyze_id, device_id, video_number, frame_image_idx,
                                                                st["segments"])
                write_done = save_frame_async(frame_save_dir_path, result['frame_image'])

                for _, v in enumerate(inference_data):
                    v["frameIdx"] = frame_image_idx

                # [FIX-3] 파일 저장 완료 대기 후 결과 전송 (framePath race condition 방지)
                if write_done is not None:
                    write_done.wait()

                log.info(f"[area_analyze] device={device_id} video={video_number} frame={frame_image_idx} event={st['is_event']} segments={st['segments']} detections={len(inference_data)}")
                infer_res = set_result_json(int(device_id), video_number, frame_image_idx, inference_data, image_idx,
                                            frame_save_dir_path, st["is_event"], st["segments"])
                sock.send(infer_res)
                if st["one_second_video"] >= frame_rate:
                    st["one_second_video"] = 0

        except (BrokenPipeError, ConnectionError, OSError) as e:
            log.error(f"소켓/IPC 에러 — 프로세스 종료: {e}")
            break
        except Exception as e:
            log.error(f"예기치 않은 에러: {e}", exc_info=True)
            consecutive_errors += 1
            if consecutive_errors > 10:
                log.error(f"연속 에러 {consecutive_errors}회 — 프로세스 종료")
                break

    reader.disconnect()
    log.info(f"[exit] area analyze loop ended. count={count}")

def area_save_frame_image(frame_path: str, analyze_id: int, device_id: int, video_number: int, frame_image_idx: int, segments: int) -> str:
    frame_save_dir_path = os.path.join(frame_path, str(analyze_id), str(device_id), str(video_number),
                                       str(frame_image_idx) + ".jpg")
    if not os.path.exists(frame_save_dir_path):
        # print(f"frame save dir not exist: {frame_save_dir_path}", file=sys.stderr)
        os.makedirs(os.path.dirname(frame_save_dir_path), exist_ok=True)
    return frame_save_dir_path

def set_result_json(device_id: int, video_number: int, frame_image_idx: int, inference_data: list[dict], shm_idx, frame_save_path, is_event: bool = False, segments: int = -1) -> bytes:
    res = {}
    res["deviceId"] = device_id
    res["videoNumber"] = video_number
    res["imageIdx"] = frame_image_idx
    res["data"] = inference_data
    res["shmIdx"] = shm_idx
    res["framePath"] = frame_save_path
    res["segment"] = segments
    res["isEvent"] = is_event
    return _json_dumps(res)

def save_image(image, image_key):
    temp_dir = os.path.join("/opt", "oms", "omeye", "ai", "ai-core-wrapping", "test-image", "temp")
    image_name = image_key + ".jpg"
    image_path = os.path.join(temp_dir, image_name)
    with open(image_path, "wb") as f:
        f.write(image)

def load_success(sock: UnixSocketSender):
    success_data = b'success'
    data_size = struct.pack('!I', len(success_data))
    data = data_size + success_data
    sock.send(data)

# TODO: 카메라 아이디와 비디오 번호 받아서 반환하기
if __name__ == "__main__":
    type_list = ["person", "attribute", "face", "carplate", "area", "exportvideo"]
    parser = argparse.ArgumentParser(description="AI core_v.1.0.0 process")
    parser.add_argument("idx", type=str, help="using gpu index")
    parser.add_argument("analyze", type=str, choices=["detect", "analyze"], help="detect target or inference")
    parser.add_argument("key", type=str, help="shared memory key")
    parser.add_argument("-s", "--share_idx", type=int, help="shared memory key for target info")
    parser.add_argument("-t", "--type", choices=type_list, type=str, default="person", help="target type")
    parser.add_argument("--target_id", type=int, default=0, help="target id")
    parser.add_argument("--analyze_id", type=int, default=0, help="analyze id")
    parser.add_argument("--target_json_path", type=str, default="", help="target json path")
    parser.add_argument("--frame_path", type=str, default="/opt/oms/omeye/filestorage/frame", help="save path what frame image")
    parser.add_argument("--test", action="store_true", help="ai function test")
    args = parser.parse_args()

    gpu_idx = int(args.idx)
    analyze_type: str = args.analyze
    shm_key: str = args.key
    shm_id: int = args.share_idx
    detect_type: str = args.type
    test_mode: bool = args.test
    analyze_id: int = args.analyze_id
    frame_path: str = args.frame_path
    target_json_path: str = args.target_json_path

    # 로거 초기화 (프로세스별 로그 파일 생성)
    log = _setup_logger(f"{analyze_type}-{detect_type}", gpu_idx)
    log.info(f"started: gpu={args.idx} analyze={args.analyze} shm={args.key} type={args.type} test={test_mode}")
    if test_mode:
        log.info(f"Test mode: type={detect_type}, target_json_path={target_json_path}")
        if analyze_type == "detect":
            # ai_core = Total_AI(camera_id=0, gpu_idx=0, process_number=0, target_type=detect_type,
            #                    target_feature=None, cfg_path=config_path)
            test_engine = _create_batch_engine(detect_type, None, 0)
            if detect_type == "person" or detect_type == "attribute":
                test_frame = cv2.imread("/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/resize-test.jpg")
                batch_tensor, orig_shapes = test_engine._preproc_frames([test_frame])
                dets = test_engine._batch_detect(batch_tensor, orig_shapes)
                res = dets[0]
                if res is not None:
                    for di in range(len(res)):
                        log.info(f"[test_detect] det[{di}] box={res[di, :4].tolist()} conf={res[di, 4]:.3f} cls={int(res[di, 5])}")
            if detect_type == "face":
                test_frame = cv2.imread("/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/test.jpg")
                batch_results = test_engine.run_batch([test_frame])
                res = batch_results[0][0]
            log.info(f"detect result: {res}")
        else:
            with quiet_mode():
                # target_dict_feature = Target(target_type=detect_type, gpu_idx=0, cfg_path=config_path).set_target(
                #     target_json_path)
                target_dict_feature = BatchTarget(target_type=detect_type, gpu_idx=0, cfg_path=config_path, log=log).set_target(
                    target_json_path)
                log.info(f"타겟데이터: {target_dict_feature}")
                ai_core = Total_AI(camera_id=0, gpu_idx=0, process_number=0, target_type=detect_type,
                                   target_feature=target_dict_feature, cfg_path=config_path)

                if detect_type == "person" or detect_type == "attribute":
                    res = ai_core.run(cv2.imread("/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/resize-test3.jpg"))
                if detect_type == "face" or detect_type == "area":
                    res = ai_core.run(cv2.imread("/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/test.jpg"))
            log.info(f"analyze result: {res}")
            i = 0
            img = cv2.imread("/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/test4.jpg")
            if detect_type != "area":
                for i in range(len(res)):
                    t = res[i]
                    temp = t["points"].tolist()
                    log.debug(f"crop [{i}] points={temp} type={type(temp[0])}")
                    t["points"] = temp
                    
                    # scale_x = 1920 / 640
                    # scale_y = 1080 / 480
                    # print(scale_x, scale_y)
                    # x1 = round(points[0] * scale_x)
                    # x2 = round(points[2] * scale_x)
                    # y1 = round((points[1]) * scale_y)
                    # y2 = round((points[3]) * scale_y)

                    x1, y1, x2, y2 = temp

                    ci = img[y1:y2, x1:x2]
                    cv2.imwrite(f"/opt/oms/omeye/omeye-hss/build-cache/omeye-hss-app/opt/oms/omeye/ai/ai-core-wrapping/test-image/crop/{i}.jpg", ci)
            else:
                pass
        sys.exit(0)

    try:
        if analyze_type == "detect":
            if detect_type == "face":
                # face: Total_AI 1배치 detect
                log.info(f"detect mode: loading Total_AI for {detect_type}")
                with quiet_mode():
                    ai_core = Total_AI(camera_id=0, gpu_idx=gpu_idx, process_number=1,
                                       target_type=detect_type, target_feature=None, cfg_path=config_path)
                log.info("model loaded, reading target data")
                read_target_data(shm_key, shm_id, detect_type)
            else:
                # person/attribute: BatchInference detect
                log.info(f"detect mode: loading batch engine for {detect_type}")
                with quiet_mode():
                    detect_engine = _create_batch_engine(detect_type, None, gpu_idx)
                log.info("batch engine loaded, reading target data")
                read_target_data(shm_key, shm_id, detect_type, batch_engine=detect_engine)
        else:
            log.info(f"analyze mode: loading model for {detect_type}")

            # 타겟 feature 로드 (비교 매칭용)
            try:
                if detect_type == "face":
                    # face: Total_AI 1배치 — BatchTarget은 사용하되 Total_AI로 추론
                    from lib import Target
                    target_dict_feature = Target(target_type=detect_type, gpu_idx=gpu_idx, cfg_path=config_path).set_target(target_json_path)
                else:
                    target_dict_feature = BatchTarget(target_type=detect_type, gpu_idx=gpu_idx, cfg_path=config_path, log=log).set_target(target_json_path)
            except Exception as e:
                log.warning(f"target feature 로드 실패 (매칭 없이 진행): {e}")
                target_dict_feature = None

            if detect_type == "face":
                # face: Total_AI 1배치 analyze
                log.info("face analyze: loading Total_AI (1-batch)")
                with quiet_mode():
                    ai_core = Total_AI(camera_id=0, gpu_idx=gpu_idx, process_number=1,
                                       target_type=detect_type, target_feature=target_dict_feature,
                                       cfg_path=config_path)
                log.info("Total_AI loaded, starting inference loop")
                sock = UnixSocketSender(shm_key)
                load_success(sock)
                analyze_images(shm_key, detect_type, sock, analyze_id, frame_path, ai_core=ai_core)
            elif detect_type == "area":
                # area: Total_AI 기반 전용 분석 루프
                log.info("area analyze: loading Total_AI")
                with quiet_mode():
                    ai_core = Total_AI(camera_id=0, gpu_idx=gpu_idx, process_number=1,
                                       target_type=detect_type, target_feature=target_dict_feature,
                                       cfg_path=config_path)
                log.info("Total_AI loaded, starting area_analyze_image")
                sock = UnixSocketSender(shm_key)
                load_success(sock)
                area_analyze_image(shm_key, detect_type, sock, analyze_id, frame_path, ai_core=ai_core)
            else:
                # person/attribute/carplate: BatchInference 배치 추론
                batch_engine = _create_batch_engine(detect_type, target_dict_feature, gpu_idx)
                log.info("batch engine loaded, starting inference loop")
                sock = UnixSocketSender(shm_key)
                load_success(sock)
                analyze_images(shm_key, detect_type, sock, analyze_id, frame_path, batch_engine=batch_engine)
    except SystemExit:
        raise
    except Exception as e:
        if log:
            log.critical(f"FATAL: gpu={gpu_idx} shm={shm_key} type={detect_type} error={e}", exc_info=True)
        else:
            # 로거 초기화 전 에러 — stderr + syslog 직접 출력
            import syslog as _syslog
            msg = f"ai-core FATAL: gpu={gpu_idx} error={e}"
            print(msg, file=sys.stderr)
            try:
                _syslog.syslog(_syslog.LOG_CRIT, msg)
            except Exception:
                pass
        sys.exit(1)
