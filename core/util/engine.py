import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, namedtuple
import tensorrt as trt

from .encrypt_utils import *


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

Binding = namedtuple("Binding", ("name", "dtype_np", "shape", "data", "ptr", "is_input"))

# def _torch_dtype_from_np(np_dtype):
#     if np_dtype == np.float16:
#         return torch.float16
#     if np_dtype == np.float32:
#         return torch.float32
#     if np_dtype == np.int32:
#         return torch.int32
#     if np_dtype == np.int8:
#         return torch.int8
#     raise TypeError(f"Unsupported dtype: {np_dtype}")


def _torch_dtype_from_trt(trt_dtype: trt.DataType) -> torch.dtype:
    if trt_dtype == trt.DataType.HALF:
        return torch.float16
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    if trt_dtype == trt.DataType.INT8:
        return torch.int8
    if trt_dtype == trt.DataType.INT32:
        return torch.int32
    raise ValueError(f"Unsupported TRT dtype: {trt_dtype}")


class AnalysisEngine_ORT(nn.Module):
    """
    ONNX Runtime GPU 추론 엔진

    v5 변경사항 (기존 대비 최소 수정):
        [1] dynamic batch dim 안전 파싱  — input_shape[0]이 str/None일 때 max_batch=16 fallback
        [2] _warmup()                    — EXHAUSTIVE cuDNN 탐색을 init 시점에 소진
        [3] upstream stream sync 명확화  — YOLO TRTBatchEngine.stream 완료 후 ORT 실행 보장
            → torch.cuda.synchronize() 위치/의미 주석 명확화 (로직 변경 없음)
        [4] actual_batch 슬라이싱 유지   — 기존과 동일
    """

    def __init__(
        self,
        weight: str,
        fp16: bool,
        output_count: int,
        device,
        gpu_mem_limit_gb: float = 4.0,
    ):
        super().__init__()

        import onnxruntime as ort

        self.torch_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.gpu_idx = (
            self.torch_device.index if self.torch_device.index is not None else 0
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        gpu_mem_limit_bytes = int(gpu_mem_limit_gb * 1024 ** 3)
        cuda_provider_options = {
            "device_id"            : self.gpu_idx,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit"        : gpu_mem_limit_bytes,
        }

        self.sess = ort.InferenceSession(
            weight,
            sess_options=sess_options,
            providers=[
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ],
        )

        active = self.sess.get_providers()
        if "CUDAExecutionProvider" not in active:
            print(f"  [ORT] ⚠️  CUDAExecutionProvider 비활성 → CPU 폴백")
            print(f"  [ORT] ⚠️  ORT/cuDNN 버전이 현재 GPU 미지원 가능성 있음")
            print(f"  [ORT] 활성 provider: {active}")
            self._use_gpu = False
        else:
            print(
                f"  [ORT] ✅ GPU 추론 활성 "
                f"(device_id={self.gpu_idx}, mem_limit={gpu_mem_limit_gb:.1f}GB): {active}"
            )
            self._use_gpu = True

        inputs  = self.sess.get_inputs()
        outputs = self.sess.get_outputs()

        if len(inputs) != 1:
            raise ValueError(f"Expected exactly 1 input tensor, got {[i.name for i in inputs]}")
        if len(outputs) < 1:
            raise ValueError("Expected at least 1 output tensor.")

        self.input_name       = inputs[0].name
        self.input_shape      = inputs[0].shape   # e.g. [16, 3, 128, 64] or ['N', 3, 128, 64]
        self.output_names_all = [o.name for o in outputs]

        # ── OUTPUT 반환 정책 ────────────────────────────────────────────────
        if output_count == 1 and len(self.output_names_all) >= 2:
            self.output_names_ret = [self.output_names_all[-1]]
        else:
            self.output_names_ret = self.output_names_all[:output_count]

        if len(self.output_names_ret) < output_count:
            raise ValueError(
                f"Model outputs({len(self.output_names_all)}) < output_count({output_count})"
            )

        # ── max_batch / input_buffer 결정 ──────────────────────────────────
        # dynamic batch 모델: input_shape[0]이 str('N') 또는 None
        # → max_batch=16 fallback, input_buffer shape을 (16, C, H, W)로 고정
        _b = self.input_shape[0]
        if isinstance(_b, int) and _b > 0:
            self.max_batch = _b
        else:
            self.max_batch = 32
            print(f"  [ORT] ⚠️  dynamic batch dim='{_b}' → max_batch={self.max_batch}")

        # input_shape의 batch dim을 max_batch로 교체하여 buffer shape 확정
        _buf_shape = (self.max_batch,) + tuple(self.input_shape[1:])
        self.input_buffer = torch.zeros(
            _buf_shape,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.fixed_batch_size = self.max_batch

        # ── IO Binding 초기화 ──────────────────────────────────────────────
        if self._use_gpu:
            self._io_binding = self.sess.io_binding()
        else:
            self._io_binding = None

        # ── EXHAUSTIVE cuDNN 탐색을 init 시점에 소진 ──────────────────────
        # production forward()가 처음 호출될 때 수십 초 블로킹되는 것을 방지
        if self._use_gpu:
            self._warmup()

    # ────────────────────────────────────────────────────────────────────────
    def _warmup(self, n_runs: int = 2):
        """EXHAUSTIVE cuDNN 알고리즘 탐색을 init 시점에 소진.

        1회: EXHAUSTIVE 벤치마크 실행 (수초~수십초, GPU/모델 크기에 따라 다름)
        2회: cuDNN 캐시 hit 확인
        이후 production forward()는 캐시된 알고리즘으로 즉시 실행.
        """
        import time as _t
        print(f"  [ORT] ⏳ EXHAUSTIVE cuDNN warmup 시작 ({n_runs}회)...")
        t0 = _t.perf_counter()
        dummy = torch.zeros_like(self.input_buffer)  # (max_batch, C, H, W)
        for _ in range(n_runs):
            self._forward_iobinding(dummy)
        elapsed = (_t.perf_counter() - t0) * 1000
        print(f"  [ORT] ✅ warmup 완료: {elapsed:.0f}ms (이후는 cuDNN 캐시 hit)")

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> list:
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_cuda or x.device.index != self.gpu_idx:
            x = x.to(self.torch_device)

        if self._use_gpu:
            return self._forward_iobinding(x)
        return self._forward_numpy(x)

    # ────────────────────────────────────────────────────────────────────────
    def _forward_iobinding(self, x: torch.Tensor) -> list:
        actual_batch = x.shape[0]

        # ── [upstream race condition 방지] ─────────────────────────────────
        # YOLO TRTBatchEngine은 self.stream(별도 CUDA 스트림)에서 추론 후
        # stream.synchronize()로 완료를 보장한다.
        # 그 결과로 만들어진 crop tensor(x)가 default stream 또는
        # 다른 스트림에서 추가 가공(preproc_crops)될 수 있으므로,
        # ORT 실행 전 device 전체 동기화로 모든 upstream 작업 완료를 보장.
        # → non_blocking=False copy_ + 아래 synchronize()가 이중 보호.
        self.input_buffer[:actual_batch].copy_(x, non_blocking=False)
        torch.cuda.synchronize(self.torch_device)  # upstream 전체 완료 보장

        iob = self._io_binding
        iob.clear_binding_inputs()
        iob.clear_binding_outputs()

        iob.bind_input(
            name         = self.input_name,
            device_type  = "cuda",
            device_id    = self.gpu_idx,
            element_type = np.float32,
            shape        = tuple(self.input_buffer.shape),  # 항상 max_batch
            buffer_ptr   = self.input_buffer.data_ptr(),
        )
        for name in self.output_names_ret:
            iob.bind_output(name, device_type="cuda", device_id=self.gpu_idx)

        self.sess.run_with_iobinding(iob)

        # ORT 완료 후 PyTorch가 출력 읽기 전 sync
        torch.cuda.synchronize(self.torch_device)

        results = []
        for ort_val in iob.get_outputs():
            try:
                tensor = torch.from_dlpack(ort_val.to_dlpack()).clone()
            except Exception:
                tensor = torch.from_numpy(ort_val.numpy()).to(self.torch_device)
            results.append(tensor[:actual_batch])

        return results

    # ────────────────────────────────────────────────────────────────────────
    def _forward_numpy(self, x: torch.Tensor) -> list:
        """CPU 폴백 경로 — CUDAExecutionProvider 비활성 시."""
        x_np = x.cpu().numpy()
        outputs = self.sess.run(
            self.output_names_ret,
            {self.input_name: x_np},
        )
        return [torch.from_numpy(out).to(self.torch_device) for out in outputs]


# ═══════════════════════════════════════════════════════════════════════════════
# TRT 엔진 래퍼 (trt_batch_test.py 기반)
# ═══════════════════════════════════════════════════════════════════════════════

class TRTBatchEngine:
    """TensorRT 엔진 래퍼 (동적 배치, PyTorch CUDA 기반).

    trt_batch_test.py의 TRTEngine 패턴 기반.
    """

    def __init__(self, engine_path: str, device: torch.device = torch.device("cuda:0")):
        import os
        self.device = device

        # CUDA 컨텍스트를 명시적으로 활성화 (멀티GPU 환경에서 필수)
        gpu_idx = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_idx)
        torch.cuda.init()
        # 더미 텐서로 CUDA 컨텍스트 확보
        _dummy = torch.zeros(1, device=device)
        del _dummy

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine 파일 없음: {engine_path}")

        file_size = os.path.getsize(engine_path)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # with open(engine_path, "rb") as f:
        #     engine_data = f.read()
        
        if engine_path.endswith(".bin"):
            try:
                with open(engine_path, "rb") as f:
                    enc_data = f.read()
                engine_data = decrypt_bytes(enc_data)
            except ValueError as e:
                raise ValueError(f"[EngineLoader] Decrypt failed: {engine_path} | {e}") from e
            except Exception as e:
                raise RuntimeError(f"[EngineLoader] Unexpected error during decrypt: {engine_path} | {e}") from e
        else:
            with open(engine_path, "rb") as f:
                engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        del engine_data # 메모리 해제
        if self.engine is None:
            raise RuntimeError(
                f"TRT engine deserialize 실패: {engine_path} "
                f"(size={file_size} bytes, gpu={gpu_idx}, "
                f"trt_version={trt.__version__})"
            )
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            info = {"name": name, "shape": shape, "dtype": dtype}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)

        self.input_name = self.inputs[0]["name"]
        self.fp16 = self.inputs[0]["dtype"] == np.float16

        # dynamic shape 여부 확인
        self.dynamic = any(-1 in inp["shape"] for inp in self.inputs)

        # max batch size 추출 + max_shape로 초기화 (DetectEngine_v2 패턴)
        # → 버퍼를 max_shape 크기로 고정 할당하여 shape 변경에 의한 재할당 방지
        self.max_batch = 16  # 기본값
        if self.dynamic:
            try:
                # TRT 10.x: get_tensor_profile_shape(tensor_name, profile_index)
                # TRT 8.x: get_profile_shape(tensor_name, profile_index) — ICudaEngine
                if hasattr(self.engine, 'get_tensor_profile_shape'):
                    profile_shapes = self.engine.get_tensor_profile_shape(self.input_name, 0)
                elif hasattr(self.engine, 'get_profile_shape'):
                    profile_shapes = self.engine.get_profile_shape(self.input_name, 0)
                else:
                    raise AttributeError("no profile_shape API found on engine")
                max_shape = tuple(profile_shapes[2])
                self.max_batch = max_shape[0]
                self.context.set_input_shape(self.input_name, max_shape)
            except Exception:
                fallback = list(self.inputs[0]["shape"])
                fallback[0] = 1
                self.context.set_input_shape(self.input_name, tuple(fallback))
        else:
            self.max_batch = self.inputs[0]["shape"][0]

        # 엔진 입력 shape 저장 (H, W 추출용)
        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))

        # 내부 버퍼 사전 할당 (max_shape 크기 — 주소 불변 보장)
        self._alloc_buffers()

    def _alloc_buffers(self):
        """입출력 버퍼 사전 할당."""
        # 입력 버퍼
        in_shape = tuple(self.context.get_tensor_shape(self.input_name))
        in_dtype = torch.float16 if self.fp16 else torch.float32
        self.input_buffer = torch.empty(in_shape, dtype=in_dtype, device=self.device)
        self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
        self._cur_input_shape = in_shape

        # 출력 버퍼
        self.output_buffers = {}
        for out_info in self.outputs:
            name = out_info["name"]
            out_shape = tuple(self.context.get_tensor_shape(name))
            out_dtype = torch.float16 if out_info["dtype"] == np.float16 else torch.float32
            buf = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self.output_buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

    def infer(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """PyTorch GPU 텐서 입력 → TRT 추론 → PyTorch GPU 텐서 출력 리스트.

        DetectEngine_v2 패턴: max_shape 고정 버퍼 + set_input_shape만 변경.
        버퍼 재할당/주소 변경 없이 안정적 추론.
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device, non_blocking=True)
        if self.fp16 and input_tensor.dtype != torch.float16:
            input_tensor = input_tensor.half()
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        cur_shape = tuple(input_tensor.shape)
        actual_batch = cur_shape[0]

        # dynamic shape 변경 시 context에 shape만 설정 (버퍼/주소는 불변)
        if cur_shape != self._cur_input_shape:
            self.context.set_input_shape(self.input_name, cur_shape)
            self._cur_input_shape = cur_shape

        # 크로스-스트림 경합 방지:
        # copy_와 execute_async_v3를 동일 스트림(self.stream)에서 실행하여
        # copy_ 완료 후 TRT 추론이 시작되도록 보장.
        # default stream의 input_tensor 생성 완료를 event로 동기화.
        default_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(default_stream)

        with torch.cuda.stream(self.stream):
            self.input_buffer[:actual_batch].copy_(input_tensor)

        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("[TRT] execute_async_v3 failed")

        self.stream.synchronize()

        # clone도 self.stream 완료 후 default stream에서 안전하게 실행
        # (stream.synchronize()가 모든 self.stream 작업 완료를 보장)
        results = []
        for out in self.outputs:
            buf = self.output_buffers[out["name"]]
            results.append(buf[:actual_batch].clone())
        return results

class AnalysisEngine_v4(nn.Module):
    """
    AnalysisEngine_v3 대비 개선 사항:
    
    1. Stream 고정 (init 1회) + 명시적 synchronize
       - v3: clone()이 PyTorch 기본 stream에서 실행되어 암묵적 동기화 발생
             → TRT가 PT보다 느려지는 주원인
       - v4: 전용 stream으로 copy_ + enqueue 통일,
             stream.synchronize() 1번만 명시적으로 호출
    
    2. set_tensor_address init 1회만 바인딩
       - v3: 매 forward마다 input/output 전체 주소 재바인딩 (불필요한 C++ 호출)
       - v4: 버퍼 주소는 불변이므로 init에서 1번만 바인딩
    
    3. clone().detach() 완전 제거
       - v3: 출력 수만큼 GPU 메모리 할당 + 복사 발생
       - v4: 출력 버퍼 직접 참조 반환
             (순차 호출 구조이므로 다음 forward 전에 소비 완료 → 버퍼 오염 없음)
    
    주의:
       - 반환된 텐서는 다음 forward() 호출 전에 소비해야 합니다.
       - 다음 forward() 이후에도 이전 결과가 필요하면 .clone() 직접 호출하세요.
    """

    def __init__(self, weight: str, fp16: bool, output_count: int, device: int):
        super().__init__()

        # --- device 고정 ---
        #torch.cuda.set_device(device)
        #self.torch_device = torch.device(f"cuda:{device}")
        self.torch_device = device

        # --- 엔진 로드 ---
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(weight, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # ---- I/O 텐서 이름 수집 ----
        input_names, output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1:
            raise ValueError(f"Expected exactly 1 input tensor, got {input_names}")
        if len(output_names) < 1:
            raise ValueError("Expected at least 1 output tensor.")

        self.input_name = input_names[0]
        self.output_names_all = list(output_names)

        # ---- OUTPUT 반환 정책 ----
        # output이 2개 이상이고 output_count==1 이면 마지막 출력만 반환
        if output_count == 1 and len(self.output_names_all) >= 2:
            self.output_names_ret = [self.output_names_all[-1]]
        else:
            self.output_names_ret = self.output_names_all[:output_count]

        if len(self.output_names_ret) < output_count:
            raise ValueError(
                f"Engine outputs({len(self.output_names_all)}) < output_count({output_count})"
            )

        # ---- 버퍼 할당 ----
        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        in_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(self.input_name))
        self.input_buffer = torch.empty(in_shape, dtype=in_dtype, device=self.torch_device)

        self.output_buffers_all = []
        for name in self.output_names_all:
            out_shape = tuple(self.engine.get_tensor_shape(name))
            out_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            self.output_buffers_all.append(
                torch.empty(out_shape, dtype=out_dtype, device=self.torch_device)
            )

        self._out_index = {n: i for i, n in enumerate(self.output_names_all)}

        # ✅ [개선 1] stream 고정 - init에서 1번만 생성
        self.stream = torch.cuda.Stream(device=self.torch_device)
        self.stream_handle = self.stream.cuda_stream

        # ✅ [개선 2] set_tensor_address init에서 1번만 바인딩
        self._bind_addresses_once()
        
        if hasattr(self.context, "enqueue_v3"):
            self._enqueue = self.context.enqueue_v3        # TRT 10.x
        elif hasattr(self.context, "execute_async_v3"):
            self._enqueue = self.context.execute_async_v3  # TRT 8~9.x


    def _bind_addresses_once(self):
        """버퍼 주소는 불변이므로 init에서 1번만 바인딩"""
        self.context.set_tensor_address(
            self.input_name, int(self.input_buffer.data_ptr())
        )
        for name, buf in zip(self.output_names_all, self.output_buffers_all):
            self.context.set_tensor_address(name, int(buf.data_ptr()))

    def forward(self, x: torch.Tensor) -> list:
        # # dtype 변환 (필요 시만)
        # if x.dtype != self.input_buffer.dtype:
        #     x = x.to(dtype=self.input_buffer.dtype)
        # if not x.is_contiguous():
        #     x = x.contiguous()

        # # ✅ [개선 1] 고정 stream으로 copy_ + enqueue 통일
        # self.input_buffer.copy_(x, non_blocking=True)

        # # set_tensor_address 제거 - 이미 init에서 바인딩됨
        # ok = self._enqueue(self.stream_handle)
        # if ok is False:
        #     raise RuntimeError("TensorRT enqueue_v3 returned False")

        # # ✅ [개선 1] 명시적 동기화 1번 (암묵적 동기화 제거)
        # self.stream.synchronize()
        
        with torch.cuda.stream(self.stream):
            if x.dtype != self.input_buffer.dtype:
                x = x.to(dtype=self.input_buffer.dtype)
            if not x.is_contiguous():
                x = x.contiguous()

            self.input_buffer.copy_(x, non_blocking=False)
        
        self.stream.synchronize()    
        ok = self._enqueue(self.stream_handle)
        if ok is False:
            raise RuntimeError("TensorRT enqueue_v3 returned False")

        torch.cuda.synchronize()

        
        return [
            self.output_buffers_all[self._out_index[n]].clone()
            for n in self.output_names_ret
        ]

def _torch_dtype_from_np(dtype_np: np.dtype) -> torch.dtype:
    mapping = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32:   torch.int32,
        np.int64:   torch.int64,
        np.uint8:   torch.uint8,
        np.bool_:   torch.bool,
    }
    for np_type, torch_type in mapping.items():
        if dtype_np == np_type:
            return torch_type
    raise TypeError(f"Unsupported numpy dtype: {dtype_np}")

class SegDetectEngineTRT_v2(torch.nn.Module):
    """
    YOLOv5-seg TensorRT loader v2.

    DetectEngine_v2와 동일한 구조로 통일:
      1. stream 고정 (init 1회) + 명시적 synchronize
      2. set_tensor_address static 모드에서 init 1회만
      3. input을 내부 버퍼로 copy_ 후 추론 (외부 텐서 주소 의존 제거)

    forward() → (pred, proto)  ← seg 전용 (detect 와 반환 형태 다름)

    주의:
      - 반환된 pred/proto 텐서는 다음 forward() 호출 전에 소비해야 합니다
      - dynamic shape 모드에서 shape 변경 시 재바인딩이 발생합니다
    """

    def __init__(
        self,
        engine_path: str,
        device: torch.device = torch.device("cuda:0"),
        input_name: str = "images",
        nc: int | None = None,
        nm: int | None = 32,
    ):
        super().__init__()
        if device.type == "cpu":
            device = torch.device("cuda:0")
        self.device = device
        self.nc = nc
        self.nm = nm

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        bindings: dict[str, Binding] = OrderedDict()
        output_names: list[str] = []
        dynamic = False
        fp16 = False

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype_np = trt.nptype(engine.get_tensor_dtype(name))

            if dtype_np == np.float16:
                fp16 = True

            shape_engine = tuple(engine.get_tensor_shape(name))

            if is_input and (-1 in shape_engine):
                dynamic = True
                max_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2])
                context.set_input_shape(name, max_shape)

            shape_ctx = tuple(context.get_tensor_shape(name))
            torch_dtype = _torch_dtype_from_np(dtype_np)
            buf = torch.empty(shape_ctx, dtype=torch_dtype, device=device)

            bindings[name] = Binding(
                name=name,
                dtype_np=dtype_np,
                shape=shape_ctx,
                data=buf,
                ptr=int(buf.data_ptr()),
                is_input=is_input,
            )
            if not is_input:
                output_names.append(name)

        self.engine = engine
        self.context = context
        self.bindings = bindings
        self.output_names = output_names
        self.dynamic = dynamic
        self.fp16 = fp16

        # input name 결정
        if input_name in bindings and bindings[input_name].is_input:
            self.input_name = input_name
        else:
            self.input_name = next(k for k, v in bindings.items() if v.is_input)

        # pred / proto 출력 이름 결정 (seg 전용)
        self.pred_name, self.proto_name = self._resolve_pred_proto_names()

        # ✅ [개선 1] stream 고정 - init에서 1번만 생성
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream_handle = self.stream.cuda_stream

        # ✅ [개선 3] 내부 input_buffer 생성 (외부 텐서 의존 제거)
        in_binding = self.bindings[self.input_name]
        self.input_buffer = torch.empty(
            in_binding.shape,
            dtype=_torch_dtype_from_np(in_binding.dtype_np),
            device=self.device,
        )
        self.bindings[self.input_name] = in_binding._replace(
            data=self.input_buffer,
            ptr=int(self.input_buffer.data_ptr()),
        )

        # ✅ [개선 2] static shape이면 init에서 1번만 주소 바인딩
        if not self.dynamic:
            self._bind_addresses_once()

    # ──────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────
    def _bind_addresses_once(self):
        """static shape 전용 - 주소 불변이므로 init에서 1번만 바인딩."""
        for name, binding in self.bindings.items():
            self.context.set_tensor_address(name, binding.ptr)

    def _resolve_pred_proto_names(self) -> tuple[str, str]:
        """
        출력 텐서 중 pred(3D), proto(4D)를 구분.
        우선순위: rank → nc/nm hint → 크기 기반 heuristic
        """
        outs = self.output_names
        if len(outs) < 2:
            raise ValueError(f"[SegDetectEngineTRT_v2] Need ≥ 2 outputs, got {outs}")

        shapes = {n: tuple(self.bindings[n].shape) for n in outs}

        rank4 = [n for n in outs if len(shapes[n]) == 4]
        rank3 = [n for n in outs if len(shapes[n]) == 3]

        if len(rank4) == 1 and len(rank3) == 1:
            return rank3[0], rank4[0]

        pred_name = proto_name = None

        if self.nm is not None:
            for n in outs:
                sh = shapes[n]
                if len(sh) == 4 and sh[1] == self.nm:
                    proto_name = n
                    break

        if self.nc is not None and self.nm is not None:
            expect_d = 5 + self.nc + self.nm
            for n in outs:
                sh = shapes[n]
                if len(sh) == 3 and sh[-1] == expect_d:
                    pred_name = n
                    break

        if pred_name is None:
            cand3 = [n for n in outs if len(shapes[n]) == 3]
            if cand3:
                pred_name = max(cand3, key=lambda n: int(np.prod(shapes[n])))

        if proto_name is None:
            cand4 = [n for n in outs if len(shapes[n]) == 4]
            if cand4:
                proto_name = max(cand4, key=lambda n: int(np.prod(shapes[n])))

        if pred_name is None or proto_name is None:
            raise ValueError(f"[SegDetectEngineTRT_v2] Cannot resolve pred/proto. shapes={shapes}")

        return pred_name, proto_name

    # ──────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────
    def forward(self, im: torch.Tensor, src_event: torch.cuda.Stream | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        im : (B, C, H, W) torch.Tensor
        returns : (pred, proto) - 버퍼 직접 참조, 다음 forward 전 소비 필요
        """
        if im.device != self.device:
            im = im.to(self.device, non_blocking=True)
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if not im.is_contiguous():
            im = im.contiguous()

        in_binding = self.bindings[self.input_name]

        # ── static shape ──────────────────────────────────────────────
        if not self.dynamic:
            if tuple(im.shape) != in_binding.shape:
                raise ValueError(
                    f"[TRT] Input shape mismatch: got {tuple(im.shape)} "
                    f"expected {in_binding.shape}"
                )
            # ✅ [개선 3] 내부 버퍼로 copy_ (고정 주소 유지)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self.input_buffer.copy_(im, non_blocking=True)

        # ── dynamic shape ─────────────────────────────────────────────
        else:
            if tuple(im.shape) != in_binding.shape:
                self.context.set_input_shape(self.input_name, tuple(im.shape))

                # input buffer 재할당
                new_in_buf = torch.empty(
                    tuple(im.shape),
                    dtype=_torch_dtype_from_np(in_binding.dtype_np),
                    device=self.device,
                )
                self.input_buffer = new_in_buf
                self.bindings[self.input_name] = in_binding._replace(
                    shape=tuple(im.shape),
                    data=new_in_buf,
                    ptr=int(new_in_buf.data_ptr()),
                )

                # output buffer 재할당
                for name in self.output_names:
                    new_shape = tuple(self.context.get_tensor_shape(name))
                    b_out = self.bindings[name]
                    if new_shape != b_out.shape:
                        new_out_buf = torch.empty(
                            new_shape,
                            dtype=_torch_dtype_from_np(b_out.dtype_np),
                            device=self.device,
                        )
                        self.bindings[name] = b_out._replace(
                            shape=new_shape,
                            data=new_out_buf,
                            ptr=int(new_out_buf.data_ptr()),
                        )

                # ✅ shape 변경 시에만 재바인딩
                self._bind_addresses_once()
            
            # #self.stream.wait_stream(torch.cuda.current_stream())
            # if src_event is not None:
            #     self.stream.wait_event(src_event)
            # else:
            #     self.stream.wait_stream(torch.cuda.current_stream())
            
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self.input_buffer.copy_(im, non_blocking=True)

        # ✅ [개선 1] 고정 stream으로 실행
        self.context.execute_async_v3(self.stream_handle)

        # ✅ [개선 1] 명시적 동기화 1번
        self.stream.synchronize()

        # seg 전용: pred / proto clone 후 반환 (버퍼 참조 해제 → 다음 forward 안전)
        return self.bindings[self.pred_name].data.clone(), self.bindings[self.proto_name].data.clone()

    # ──────────────────────────────────────────────
    # 편의 메서드
    # ──────────────────────────────────────────────
    def warmup(self, imgsz: tuple = (1, 3, 640, 640), n: int = 3):
        """엔진 워밍업 – 첫 추론 JIT 지연 제거."""
        dtype = torch.float16 if self.fp16 else torch.float32
        dummy = torch.zeros(imgsz, dtype=dtype, device=self.device)
        for _ in range(n):
            self.forward(dummy)