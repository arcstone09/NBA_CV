"""
alignment.py
input : (video_path, target_clock, output_path) 
output : video_path의 pbp영상을 target_clock 기준 -1~1초 동안 crop 한 영상을 output_path에 저장
"""
import cv2
import os
import re
import math
import pytesseract
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


# =========================
# 설정
# =========================
from pathlib import Path


BASE_DIR = Path("/Users/arcstone/Desktop/snupi/nba_cv")
video_path = BASE_DIR / "data" / "raw_data" / "0022401100_39.mp4"


@dataclass
class Config:
    # coarse search 시 몇 프레임마다 볼지
    coarse_stride: int = 4

    # coarse 단계에서 OCR할 후보 ROI들
    # scoreboard가 보통 상단/하단 모서리에 있다고 가정
    candidate_rois: Tuple[Tuple[float, float, float, float], ...] = (
        # (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
        (0.00, 0.00, 0.45, 0.18),  # top-left wide ## 영상의 가로 0% ~ 45%, 세로 0% ~ 18% 즉 왼쪽 위 넓은 영역.
        (0.55, 0.00, 1.00, 0.18),  # top-right wide
        (0.00, 0.00, 0.35, 0.14),  # top-left tighter
        (0.65, 0.00, 1.00, 0.14),  # top-right tighter
        (0.00, 0.82, 0.45, 1.00),  # bottom-left
        (0.55, 0.82, 1.00, 1.00),  # bottom-right
    )

    # OCR 전처리용 upscale 배수
    ocr_scale: int = 3

    # OCR 설정
    tesseract_psm: int = 6

    # refine 단계에서 best 주변 몇 프레임을 추가 확인할지
    refine_radius: int = 20

    # 저장할 temporal window
    # 영상 fps에 따라(pba 기본 60인듯) 앞 뒤 1초씩 총 2초로 해야할듯.
    save_left: int = 60
    save_right: int = 60

    # 디버그 폴더 생성 여부
    save_debug_images: bool = False


# =========================
# 유틸
# =========================

def parse_clock_to_seconds(clock_str: str) -> int:
    """
    "03:21" -> 201
    """
    m = re.match(r"^\s*(\d{1,2})[:\.](\d{2})\s*$", clock_str)
    if not m:
        raise ValueError(f"Invalid clock string: {clock_str}")
    mm = int(m.group(1))
    ss = int(m.group(2))
    return mm * 60 + ss


def normalize_ocr_text(text: str) -> str:
    """
    OCR 결과를 시간 파싱하기 쉽게 정리
    "O8.38" → "08:38"
    "8;38" → "8:38" 등 교정.
    """
    text = text.upper()
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("L", "1")
    text = text.replace("|", "1")
    text = text.replace(" ", "")
    text = text.replace(";", ":")
    text = text.replace(",", ":")
    text = text.replace(".", ":")
    return text


def extract_clock_candidates(text: str) -> List[str]:
    """
    OCR 문자열에서 mm:ss 후보 추출
    예: "2ND3:21GSW" -> ["3:21"]
    """
    text = normalize_ocr_text(text)
    # 0:00 ~ 12:59 정도를 넉넉히 커버, OCR 문자열에서 m:ss 혹은 mm:ss 형태만 뽑기
    matches = re.findall(r"(\d{1,2}:\d{2})", text)
    return matches


def extract_period_candidates(text: str) -> List[int]:
    """
    OCR 문자열에서 quarter/period 후보 추출
    예: "2ND", "Q2", "2Q" 등
    """
    text = text.upper().replace(" ", "")
    candidates = []

    patterns = [
        r"Q([1-4])",
        r"([1-4])Q",
        r"([1-4])ST",
        r"([1-4])ND",
        r"([1-4])RD",
        r"([1-4])TH",
        r"PERIOD([1-4])",
    ]

    for p in patterns:
        for m in re.findall(p, text):
            try:
                candidates.append(int(m))
            except ValueError:
                pass

    return list(set(candidates))


def clock_distance_seconds(a: str, b: str) -> int:
    """
    두 경기시간 mm:ss 사이 차이(초)
    두 시간 문자열 차이를 초 단위 절댓값으로 계산. "08:38" vs "08:40" → 2초
    이 값이 작을수록 target에 가깝다.
    """
    return abs(parse_clock_to_seconds(a) - parse_clock_to_seconds(b))


def preprocess_for_ocr(img_bgr):
    """
    OCR 잘 되도록 전처리.
    return은 결과는 OCR에 넣기 좋은 binary 비슷한 이미지.
    """
    #색은 버리고 밝기만 남김.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 확대. 작은 scoreboard 숫자를 OCR이 잘 읽게 3배 확대.
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # bilateral로 edge 보존. 경계는 살리고 노이즈 줄임.
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # adaptive threshold. 조명이나 배경이 균일하지 않아도 글자/배경을 흑백으로 좀 더 잘 분리.
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    # 너무 작은 점 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def ocr_text(img_bgr, psm: int = 6) -> str:
    """
    pytesseract OCR
    실제 OCR 수행 함수.
    먼저 preprocess_for_ocr
    그다음 pytesseract.image_to_string
    """
    proc = preprocess_for_ocr(img_bgr)
    # --oem 3: 기본 LSTM OCR 엔진, --psm 6: 텍스트 블록 모드
    config = f"--oem 3 --psm {psm}" 
    text = pytesseract.image_to_string(proc, config=config)
    return text


def get_roi_from_ratio(frame, roi_ratio: Tuple[float, float, float, float]):
    """
    ratio 기반 ROI(ROI = Region Of Interest, 관심영역)를 실제 픽셀 좌표로 바꿔서 잘라냄.
    frame width가 1920일 때, x1_ratio = 0.55면, x1 = 1056x1_ratio = 0.55면, x1 = 1056
    """
    h, w = frame.shape[:2]
    x1 = int(w * roi_ratio[0])
    y1 = int(h * roi_ratio[1])
    x2 = int(w * roi_ratio[2])
    y2 = int(h * roi_ratio[3])
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def score_ocr_result(
    text: str,
    target_clock: str,
    target_period: Optional[int] = None
) -> Tuple[float, Optional[str], Optional[int]]:
    """
    OCR 결과가 target과 얼마나 맞는지 점수화
    낮을수록 좋음.
    """
    clock_candidates = extract_clock_candidates(text)
    period_candidates = extract_period_candidates(text)

    # 시간 후보가 없으면 완전 나쁜 점수.
    if not clock_candidates:
        return float("inf"), None, None

    best_score = float("inf") 
    best_clock = None
    best_period = None

    for c in clock_candidates:
        try:
            d_clock = clock_distance_seconds(c, target_clock)
        except Exception:
            continue

        # 기본은 clock 차이로 점수
        score = float(d_clock)

        # period가 주어졌으면 맞는 후보가 있을 때 보너스/패널티
        # 이 부분 개선 필요할듯. 
        if target_period is not None:
            if target_period in period_candidates:
                score -= 0.5
                cand_p = target_period
            elif len(period_candidates) > 0:
                score += 3.0
                cand_p = period_candidates[0]
            else:
                score += 1.0
                cand_p = None
        else:
            cand_p = period_candidates[0] if period_candidates else None

        if score < best_score:
            best_score = score
            best_clock = c
            best_period = cand_p
    # 가장 좋은 점수 반환 (best_score, best_clock, best_period) : (0.5, "08:38", 2)
    return best_score, best_clock, best_period


# =========================
# 탐색
# =========================

def coarse_search_best_anchor(
    video_path: str,
    target_clock: str,
    target_period: Optional[int],
    cfg: Config,
    debug_dir: str
) -> Tuple[int, Tuple[int, int, int, int], pd.DataFrame]:
    """
    1차 탐색:
    - 여러 candidate ROI를 coarse stride로 훑으면서
    - target 시간에 가장 가까운 frame, ROI를 찾음
    - reuturn : coarse best frame index, 그때 사용한 ROI 좌표. 모든 결과 DataFrame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []

    best_score = float("inf")
    best_frame_idx = -1
    best_roi = None

    for frame_idx in range(0, frame_count, cfg.coarse_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 프레임 하나당 scoreboard 후보 영역 여러 개를 검사.
        for roi_id, roi_ratio in enumerate(cfg.candidate_rois):
            roi_img, roi_xyxy = get_roi_from_ratio(frame, roi_ratio)

            try:
                text = ocr_text(roi_img, psm=cfg.tesseract_psm)
            except Exception:
                text = ""

            score, matched_clock, matched_period = score_ocr_result(
                text=text,
                target_clock=target_clock,
                target_period=target_period
            )
            # 모든 탐색 결과를 테이블로 저장해 나중에 csv로 볼 수 있게 함.
            rows.append({
                "frame_idx": frame_idx,
                "roi_id": roi_id,
                "roi_xyxy": roi_xyxy,
                "ocr_text": text,
                "matched_clock": matched_clock,
                "matched_period": matched_period,
                "score": score,
            })

            if score < best_score:
                best_score = score
                best_frame_idx = frame_idx
                best_roi = roi_xyxy

                if cfg.save_debug_images:
                    dbg = frame.copy()
                    x1, y1, x2, y2 = roi_xyxy
                    # 현재 best 프레임에 초록 박스를 그리고 best_coarse_frame.jpg, best_coarse_roi.jpg 로 저장.
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        dbg,
                        f"BEST so far: frame={frame_idx}, score={score:.2f}, clock={matched_clock}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    cv2.imwrite(os.path.join(debug_dir, "best_coarse_frame.jpg"), dbg)
                    cv2.imwrite(os.path.join(debug_dir, "best_coarse_roi.jpg"), roi_img)

    cap.release()

    if best_frame_idx < 0 or best_roi is None:
        raise RuntimeError("Failed to find any plausible scoreboard/time match.")

    df = pd.DataFrame(rows).sort_values(["score", "frame_idx"]).reset_index(drop=True)

    return best_frame_idx, best_roi, df


def refine_anchor_near_best(
    video_path: str,
    target_clock: str,
    target_period: Optional[int],
    coarse_best_frame: int,
    best_roi_xyxy: Tuple[int, int, int, int],
    cfg: Config,
    debug_dir: str
) -> Tuple[int, pd.DataFrame]:
    """
    2차 탐색:
    - coarse best frame 주변만
    - best ROI 고정으로 더 촘촘히 OCR해서 최종 frame 선택
    - return : 최종 anchor frame과 refine 결과표.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = best_roi_xyxy

    # coarse best frame ±20 프레임 범위.
    left = max(0, coarse_best_frame - cfg.refine_radius)
    right = min(frame_count - 1, coarse_best_frame + cfg.refine_radius)

    rows = []
    best_score = float("inf")
    best_frame_idx = coarse_best_frame

    for frame_idx in range(left, right + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        roi_img = frame[y1:y2, x1:x2].copy()

        try:
            text = ocr_text(roi_img, psm=cfg.tesseract_psm)
        except Exception:
            text = ""

        score, matched_clock, matched_period = score_ocr_result(
            text=text,
            target_clock=target_clock,
            target_period=target_period
        )

        rows.append({
            "frame_idx": frame_idx,
            "ocr_text": text,
            "matched_clock": matched_clock,
            "matched_period": matched_period,
            "score": score,
        })

        if score < best_score:
            best_score = score
            best_frame_idx = frame_idx

            if cfg.save_debug_images:
                dbg = frame.copy()
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    dbg,
                    f"REFINE BEST: frame={frame_idx}, score={score:.2f}, clock={matched_clock}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imwrite(os.path.join(debug_dir, "best_refine_frame.jpg"), dbg)
                cv2.imwrite(os.path.join(debug_dir, "best_refine_roi.jpg"), roi_img)

    cap.release()

    df = pd.DataFrame(rows).sort_values(["score", "frame_idx"]).reset_index(drop=True)
    return best_frame_idx, df


# =========================
# 저장
# =========================
# 최종 anchor frame 주변 영상만 새 mp4로 저장한다.
def save_temporal_crop(
    video_path: str,
    anchor_frame_idx: int,
    output_path: str,
    left_frames: int,
    right_frames: int
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_idx = max(0, anchor_frame_idx - left_frames)
    end_idx = min(frame_count - 1, anchor_frame_idx + right_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    for _ in range(start_idx, end_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()

    return {
        "start_frame": start_idx,
        "end_frame": end_idx,
        "saved_frames": end_idx - start_idx + 1,
        "fps": fps,
    }


# =========================
# 메인 파이프라인
# =========================
# 전체 orchestrator 함수.

def crop_clip_around_target_clock(
    video_path: str,
    target_clock: str,
    output_path: str,
    target_period: Optional[int] = None,
    debug_dir: Optional[str] = None,
    cfg: Optional[Config] = None
) -> Dict:
    """
    video_path: 입력 mp4
    target_clock: "03:21" 형태
    target_period: 1,2,3,4 중 하나 또는 None
    output_path: 잘라낸 새 mp4 저장 경로
    debug_dir: 디버그 파일 저장 폴더
    cfg: 설정 객체
    """
    if cfg is None:
        cfg = Config()

    if debug_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        debug_dir = f"./debug_{base}_{target_clock.replace(':', '-')}"
    os.makedirs(debug_dir, exist_ok=True)

    # 1차 coarse 탐색
    coarse_best_frame, best_roi, coarse_df = coarse_search_best_anchor(
        video_path=video_path,
        target_clock=target_clock,
        target_period=target_period,
        cfg=cfg,
        debug_dir=debug_dir
    )
    coarse_df.to_csv(os.path.join(debug_dir, "coarse_search_results.csv"), index=False)

    # 2차 refine 탐색
    final_anchor_frame, refine_df = refine_anchor_near_best(
        video_path=video_path,
        target_clock=target_clock,
        target_period=target_period,
        coarse_best_frame=coarse_best_frame,
        best_roi_xyxy=best_roi,
        cfg=cfg,
        debug_dir=debug_dir
    )
    refine_df.to_csv(os.path.join(debug_dir, "refine_search_results.csv"), index=False)

    # clip 저장
    save_info = save_temporal_crop(
        video_path=video_path,
        anchor_frame_idx=final_anchor_frame,
        output_path=output_path,
        left_frames=cfg.save_left,
        right_frames=cfg.save_right
    )

    # 메타데이터 저장
    meta = {
        "video_path": video_path,
        "target_clock": target_clock,
        "target_period": target_period,
        "best_roi_xyxy": best_roi,
        "coarse_best_frame": coarse_best_frame,
        "final_anchor_frame": final_anchor_frame,
        "output_path": output_path,
        **save_info
    }
    pd.DataFrame([meta]).to_csv(os.path.join(debug_dir, "summary.csv"), index=False)

    return meta


# =========================
# 실행 예시
# =========================

if __name__ == "__main__":
    # video_path = "0022401100_39.mp4"
    target_clock = "08:38"
    target_period = 2   # 모르면 None 가능
    output_path = "cropped_around_0022401100_39.mp4"

    result = crop_clip_around_target_clock(
        video_path=video_path,
        target_clock=target_clock,
        target_period=target_period,
        output_path=output_path
    )

    print("=== DONE ===")
    for k, v in result.items():
        print(f"{k}: {v}")