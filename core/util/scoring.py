import numpy as np
import math
from scipy import spatial

def average_rank_score_logits(
    logits: np.ndarray,
    gt_indices: list[int],
) -> tuple[float, float, bool]:

    N = logits.size
    k = len(gt_indices)

    sorted_idx = np.argsort(-logits, kind='stable')

    ranks = []
    for gt in gt_indices:
        pos0 = np.where(sorted_idx == gt)[0][0]  # 0-based
        ranks.append(pos0 + 1)                    # 1-based

    avg_rank = np.mean(ranks)

    # ✅ 3위까지 완만한 선형 감소, 이후 지수 급감
    if avg_rank <= 3:
        score = 1.0 - 0.2 * (avg_rank - 1)           # rank1=1.0, rank2=0.8, rank3=0.6
    else:
        score = 0.6 * np.exp(-0.5 * (avg_rank - 3))  # rank4=0.364, rank8=0.049

    # ✅ top3 기준 필터
    top3_count = sum(1 for r in ranks if r <= 3)
    required = math.ceil(k / 2)                       # k=1→1, k=2→1, k=3→2
    delete = top3_count < required

    return float(score), float(avg_rank), delete

def best_color_score_logits(
    logits: np.ndarray,
    gt_indices: list[int],
    logit_threshold: float = 0.3
) -> tuple[int, float, bool]:

    # (gt_index, logit 값)
    candidates = [(int(gt), float(logits[gt])) for gt in gt_indices]

    # ✅ logit이 가장 높은 색 선택
    best_gt, best_score = max(candidates, key=lambda x: x[1])

    # ✅ 전체가 threshold 이하이면 fail
    logit_fail = all(logits[gt] <= logit_threshold for gt in gt_indices)

    # ✅ delete 조건 (단순화)
    delete = logit_fail

    return best_gt, best_score, delete


def compute_color_score(output, color_list, color_map, target_set, match_color_filter_thresh):
    filtered = [v for v in color_list if v in target_set]
    if not filtered:
        return None, 0, False
    result = output[color_list]
    gt_local = [color_map[v] for v in filtered]
    best_gt, best_score, delete = average_rank_score_logits(result, gt_local, match_color_filter_thresh)
    return best_gt, best_score, delete

# 2차 개선 버전
# def compute_color_score(output, color_list, color_map, target_set):
#     filtered = [v for v in color_list if v in target_set and output[v] > 0.5]
#     if not filtered:
#         return 0, False
#     result = output[color_list]
#     gt_local = [color_map[v] for v in filtered]
#     score, avg_rank, delete = average_rank_score_logits(result, gt_local)
#     return score, delete

def plate_match(target_features_dict, result_ocr, box):
    scores_dict = {}
    for _, (target_id, target_ocr) in enumerate(target_features_dict.items()):
        target_ocr_idx, target_ocr_val = target_ocr
        
        # 맞는 글자 수 카운팅
        cnt = 0
        for i in target_ocr_idx:
            #cnt += 1
            if result_ocr[i] == target_ocr_val[i]:
                cnt += 1
                continue
            else:
                break
            
        
        if cnt == len(target_ocr_idx):
            result = "".join(result_ocr)
            scores_dict[target_id] = {"score" : 0, "infer" : result, "bbox" : box}
    return scores_dict

def face_match_score(target_features, feature, score_threshold:int = 0.5):
    scores_dict = {}
    for target_dict in target_features:
        for id, target in target_dict.items():      
            score = 1 - spatial.distance.cosine(target, feature)
            if score >= score_threshold:
                scores_dict[id] = score
    return scores_dict