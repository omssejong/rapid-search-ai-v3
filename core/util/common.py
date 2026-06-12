import yaml

class DotDict(dict):
    """dict를 . 으로 접근 가능하게 만드는 클래스"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_dotdict(obj):
    if isinstance(obj, dict):
        return DotDict({k: to_dotdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_dotdict(v) for v in obj]
    else:
        return obj

def section_to_dataclass(dotdict_section, dataclass_cls):
    """DotDict 의 특정 섹션을 dataclass 로 변환"""
    return dataclass_cls(**dict(dotdict_section))