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
from pathlib import Path


# =========================
# 설정
# =========================

BASE_DIR = Path("/Users/arcstone/Desktop/snupi/NBA_CV")
video_path = BASE_DIR / "data" / "raw_data" / "0022401100_39.mp4"


@dataclass
class Config:
    # 몇 프레임마다 OCR할지.
    # 기존 4는 너무 느림. 60fps 기준 12면 약 0.2초마다 확인.
    coarse_stride: int = 12

    # scoreboard 후보 영역.
    # 너무 많이 잡으면 느려짐.
    candidate_rois: Tuple[Tuple[float, float, float, float], ...] = (
        (0.00, 0.00, 0.45, 0.18),
        (0.55, 0.00, 1.00, 0.18),
        (0.00, 0.82, 0.45, 1.00),
        (0.55, 0.82, 1.00, 1.00),
    )

    # OCR 전처리 확대 배수
    ocr_scale: int = 3

    # scoreboard 시간만 읽을 것이므로 psm 7이 더 안정적인 경우가 많음.
    tesseract_psm: int = 7

    # coarse best 주변 refine 범위
    refine_radius: int = 30

    # 저장할 temporal window
    save_left: int = 60
    save_right: int = 60

    # 디버그 이미지 저장 여부
    save_debug_images: bool = False

    # target_clock과 OCR clock의 허용 오차.
    # 1.5면 대체로 같은 초 또는 ±1초 정도만 허용.
    max_acceptable_score: float = 1.5


# =========================
# 유틸
# =========================

def parse_clock_to_seconds(clock_str: str) -> int:
    """
    "03:21" -> 201
    """
    m = re.match(r"^\s*(\d{1,2})[:\.](\d{2})\s*$", str(clock_str))
    if not m:
        raise ValueError(f"Invalid clock string: {clock_str}")

    mm = int(m.group(1))
    ss = int(m.group(2))

    if ss < 0 or ss >= 60:
        raise ValueError(f"Invalid seconds in clock: {clock_str}")

    return mm * 60 + ss


def normalize_ocr_text(text: str) -> str:
    """
    OCR 결과를 시간 파싱하기 쉽게 정리.
    """
    text = str(text).upper()
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
    OCR 문자열에서 mm:ss 후보 추출.
    """
    text = normalize_ocr_text(text)

    matches = re.findall(r"(\d{1,2}:\d{2})", text)

    valid = []
    for m in matches:
        try:
            sec = parse_clock_to_seconds(m)
            mm, ss = m.split(":")
            mm = int(mm)
            ss = int(ss)

            # NBA quarter clock은 보통 0:00 ~ 12:00.
            # 넉넉하게 0:00 ~ 15:00까지 허용.
            if 0 <= mm <= 15 and 0 <= ss < 60:
                valid.append(f"{mm:02d}:{ss:02d}")
        except Exception:
            pass

    return valid


def extract_period_candidates(text: str) -> List[int]:
    """
    OCR 문자열에서 quarter/period 후보 추출.
    """
    text = str(text).upper().replace(" ", "")
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
    return abs(parse_clock_to_seconds(a) - parse_clock_to_seconds(b))


def preprocess_for_ocr(img_bgr, scale: int = 3):
    """
    pytesseract에 넣기 위한 전처리.
    GPU를 여기서 쓰는 것은 대부분 이득이 작음.
    이유: pytesseract 자체가 CPU라서 결국 CPU 이미지가 필요함.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    gray = cv2.resize(
        gray,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def ocr_text(img_bgr, psm: int = 7, scale: int = 3) -> str:
    """
    pytesseract OCR.
    OCR 실패 시 예외를 밖으로 던지지 않고 빈 문자열 반환.
    """
    try:
        proc = preprocess_for_ocr(img_bgr, scale=scale)
        if proc is None:
            return ""

        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=0123456789:."
        )

        text = pytesseract.image_to_string(proc, config=config)
        return text

    except Exception:
        return ""


def get_roi_from_ratio(frame, roi_ratio: Tuple[float, float, float, float]):
    h, w = frame.shape[:2]

    x1 = int(w * roi_ratio[0])
    y1 = int(h * roi_ratio[1])
    x2 = int(w * roi_ratio[2])
    y2 = int(h * roi_ratio[3])

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def score_ocr_result(
    text: str,
    target_clock: str,
    target_period: Optional[int] = None
) -> Tuple[float, Optional[str], Optional[int]]:
    """
    OCR 결과가 target과 얼마나 맞는지 점수화.
    낮을수록 좋음.
    """
    clock_candidates = extract_clock_candidates(text)
    period_candidates = extract_period_candidates(text)

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

        score = float(d_clock)

        cand_p = None
        if target_period is not None:
            if target_period in period_candidates:
                score -= 0.5
                cand_p = target_period
            elif len(period_candidates) > 0:
                score += 3.0
                cand_p = period_candidates[0]
            else:
                score += 1.0
        else:
            cand_p = period_candidates[0] if period_candidates else None

        if score < best_score:
            best_score = score
            best_clock = c
            best_period = cand_p

    return best_score, best_clock, best_period


def safe_write_csv(df: Optional[pd.DataFrame], path: str):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


# =========================
# 탐색
# =========================

def coarse_search_best_anchor(
    video_path: str,
    target_clock: str,
    target_period: Optional[int],
    cfg: Config,
    debug_dir: str
) -> Optional[Tuple[int, Tuple[int, int, int, int], pd.DataFrame, float]]:
    """
    1차 탐색.
    실패 시 None 반환.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    best_score = float("inf")
    best_frame_idx = -1
    best_roi = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % cfg.coarse_stride != 0:
            frame_idx += 1
            continue

        for roi_id, roi_ratio in enumerate(cfg.candidate_rois):
            roi_img, roi_xyxy = get_roi_from_ratio(frame, roi_ratio)

            text = ocr_text(
                roi_img,
                psm=cfg.tesseract_psm,
                scale=cfg.ocr_scale
            )

            score, matched_clock, matched_period = score_ocr_result(
                text=text,
                target_clock=target_clock,
                target_period=target_period
            )

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
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        dbg,
                        f"BEST coarse: frame={frame_idx}, score={score:.2f}, clock={matched_clock}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    cv2.imwrite(os.path.join(debug_dir, "best_coarse_frame.jpg"), dbg)
                    cv2.imwrite(os.path.join(debug_dir, "best_coarse_roi.jpg"), roi_img)

        # 완전 일치하면 더 볼 필요 없이 멈춤.
        if best_score <= 0.0:
            break

        frame_idx += 1

    cap.release()

    if best_frame_idx < 0 or best_roi is None:
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "frame_idx"]).reset_index(drop=True)

    if best_score > cfg.max_acceptable_score:
        return None

    return best_frame_idx, best_roi, df, best_score


def refine_anchor_near_best(
    video_path: str,
    target_clock: str,
    target_period: Optional[int],
    coarse_best_frame: int,
    best_roi_xyxy: Tuple[int, int, int, int],
    cfg: Config,
    debug_dir: str
) -> Optional[Tuple[int, pd.DataFrame, float]]:
    """
    2차 탐색.
    실패 시 None 반환.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = best_roi_xyxy

    left = max(0, coarse_best_frame - cfg.refine_radius)
    right = min(frame_count - 1, coarse_best_frame + cfg.refine_radius)

    rows = []
    best_score = float("inf")
    best_frame_idx = coarse_best_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, left)
    current = left

    while current <= right:
        ret, frame = cap.read()
        if not ret:
            break

        roi_img = frame[y1:y2, x1:x2].copy()

        text = ocr_text(
            roi_img,
            psm=cfg.tesseract_psm,
            scale=cfg.ocr_scale
        )

        score, matched_clock, matched_period = score_ocr_result(
            text=text,
            target_clock=target_clock,
            target_period=target_period
        )

        rows.append({
            "frame_idx": current,
            "ocr_text": text,
            "matched_clock": matched_clock,
            "matched_period": matched_period,
            "score": score,
        })

        if score < best_score:
            best_score = score
            best_frame_idx = current

            if cfg.save_debug_images:
                dbg = frame.copy()
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    dbg,
                    f"BEST refine: frame={current}, score={score:.2f}, clock={matched_clock}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imwrite(os.path.join(debug_dir, "best_refine_frame.jpg"), dbg)
                cv2.imwrite(os.path.join(debug_dir, "best_refine_roi.jpg"), roi_img)

        current += 1

    cap.release()

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "frame_idx"]).reset_index(drop=True)

    if best_score > cfg.max_acceptable_score:
        return None

    return best_frame_idx, df, best_score


# =========================
# 저장
# =========================

def save_temporal_crop(
    video_path: str,
    anchor_frame_idx: int,
    output_path: str,
    left_frames: int,
    right_frames: int
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_idx = max(0, anchor_frame_idx - left_frames)
    end_idx = min(frame_count - 1, anchor_frame_idx + right_frames)

    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    saved = 0
    for _ in range(start_idx, end_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        saved += 1

    writer.release()
    cap.release()

    return {
        "start_frame": start_idx,
        "end_frame": start_idx + saved - 1,
        "saved_frames": saved,
        "fps": fps,
    }


# =========================
# 메인 파이프라인
# =========================

def crop_clip_around_target_clock(
    video_path: str,
    target_clock: str,
    output_path: str,
    target_period: Optional[int] = None,
    debug_dir: Optional[str] = None,
    cfg: Optional[Config] = None
) -> Optional[Dict]:
    """
    성공 시 meta dict 반환.
    실패 시 None 반환.
    """
    if cfg is None:
        cfg = Config()

    if debug_dir is None:
        base = os.path.splitext(os.path.basename(str(video_path)))[0]
        debug_dir = f"./debug_{base}_{target_clock.replace(':', '-')}"

    os.makedirs(debug_dir, exist_ok=True)

    coarse_result = coarse_search_best_anchor(
        video_path=video_path,
        target_clock=target_clock,
        target_period=target_period,
        cfg=cfg,
        debug_dir=debug_dir
    )

    if coarse_result is None:
        return None

    coarse_best_frame, best_roi, coarse_df, coarse_score = coarse_result
    safe_write_csv(coarse_df, os.path.join(debug_dir, "coarse_search_results.csv"))

    refine_result = refine_anchor_near_best(
        video_path=video_path,
        target_clock=target_clock,
        target_period=target_period,
        coarse_best_frame=coarse_best_frame,
        best_roi_xyxy=best_roi,
        cfg=cfg,
        debug_dir=debug_dir
    )

    if refine_result is None:
        return None

    final_anchor_frame, refine_df, refine_score = refine_result
    safe_write_csv(refine_df, os.path.join(debug_dir, "refine_search_results.csv"))

    save_info = save_temporal_crop(
        video_path=video_path,
        anchor_frame_idx=final_anchor_frame,
        output_path=output_path,
        left_frames=cfg.save_left,
        right_frames=cfg.save_right
    )

    meta = {
        "video_path": video_path,
        "target_clock": target_clock,
        "target_period": target_period,
        "best_roi_xyxy": best_roi,
        "coarse_best_frame": coarse_best_frame,
        "final_anchor_frame": final_anchor_frame,
        "coarse_score": coarse_score,
        "refine_score": refine_score,
        "best_score": refine_score,
        "output_path": output_path,
        **save_info,
    }

    pd.DataFrame([meta]).to_csv(os.path.join(debug_dir, "summary.csv"), index=False)

    return meta


# =========================
# 실행 예시
# =========================

if __name__ == "__main__":
    target_clock = "08:38"
    target_period = 2
    output_path = "cropped_around_0022401100_39.mp4"

    result = crop_clip_around_target_clock(
        video_path=str(video_path),
        target_clock=target_clock,
        target_period=target_period,
        output_path=output_path
    )

    if result is None:
        print("=== FAILED ===")
    else:
        print("=== DONE ===")
        for k, v in result.items():
            print(f"{k}: {v}")