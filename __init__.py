# # lib/__init__.py
# import os, sys

# # lib 디렉터리 자체 경로
# _LIB_DIR = os.path.dirname(__file__)

# # lib 디렉터리가 sys.path에 없으면 추가
# if _LIB_DIR not in sys.path:
#     sys.path.insert(0, _LIB_DIR)

# # 이제 lib 디렉터리 안에 있는 core_v.1.0.0.so를 top-level "core_v.1.0.0"로 import 가능
# from .core import analysis, target
# # import core_v.1.0.0

# BatchInference = analysis.BatchInference
# FaceInference = analysis.FaceInference
# AreaInference = analysis.AreaInference
# #AutoDetect = analysis.AutoDetect
# ExportVideo = analysis.ExportVideo
# BatchTarget = target.BatchTarget

# __all__ = ["BatchInference", "FaceInference", "BatchTarget", "ExportVideo"]

from .core import analysis, target

BatchInference = analysis.BatchInference
FaceInference = analysis.FaceInference
AreaInference = analysis.AreaInference
ExportVideo = analysis.ExportVideo
BatchTarget = target.BatchTarget

__all__ = [
    "BatchInference",
    "FaceInference",
    "AreaInference",
    "BatchTarget",
    "ExportVideo",
]