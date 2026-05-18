"""
data_pipe_crop.py
../../data/curry_24_align에 curry의 24/25 season 모든 슛 시도 pbp align 영상 다운
"""

import data_pipe_download as dpd
import alignment as al
import time
import pandas as pd
import re
import os
from pathlib import Path

BASE_DIR = Path("/workspace/NBA_CV")
input_path = BASE_DIR / "data" / "curry_24_raw_data"
output_path = BASE_DIR / "data" / "curry_24_crop_data"
csv_path = BASE_DIR / "data" / "curry_2024_25_all_shots.csv"
failed_log_path = BASE_DIR / "data" / "curry_24_crop_failed_log.csv"

# BASE_DIR = Path("/Users/arcstone/Desktop/snupi/nba_cv")
# input_path = BASE_DIR / "data" / "curry_24_raw_data"
# output_path = BASE_DIR / "data" / "curry_24_crop_data"


def iso_duration_to_mmss(s):
    match = re.match(r'PT(\d+)M([\d.]+)S', str(s))
    if not match:
        raise ValueError(f"Invalid format: {s}")

    minutes = int(match.group(1))
    seconds = int(float(match.group(2)))

    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    """
    # 1. 전체 경기 목록 수집
    player_name = "Stephen Curry"
    curry_id = dpd.get_player_id(player_name)
    print("Curry player_id =", curry_id)

    seasons = ["2024-25"]
    season_type = "Regular Season"

    all_logs = []
    for s in seasons:
        df = dpd.fetch_player_gamelog(curry_id, season=s, season_type=season_type)
        df["SEASON"] = s
        all_logs.append(df)
        time.sleep(1.0)

    logs = pd.concat(all_logs, ignore_index=True)
    logs = logs.sort_values("GAME_DATE")

    game_ids_2025 = logs["Game_ID"].unique().tolist()

    game_ids = game_ids_2025

    curry_shots_df = dpd.extract_player_shots_all_games(game_ids, curry_id)
    """

    output_path.mkdir(parents=True, exist_ok=True)

    curry_shots_df = pd.read_csv(csv_path)
    curry_shots_df["GAME_ID"] = curry_shots_df["GAME_ID"].astype(str).str.zfill(10)
    curry_shots_df["EVENTNUM"] = curry_shots_df["EVENTNUM"].astype(int).astype(str)

    cfg = al.Config(
        coarse_stride=12,
        refine_radius=30,
        save_left=60,
        save_right=60,
        max_acceptable_score=1.5,
        save_debug_images=False,
    )

    failed_rows = []

    for idx, row in curry_shots_df.iterrows():
        game_id = row["GAME_ID"]
        event_id = row["EVENTNUM"]

        video_path = input_path / f"{game_id}_{event_id}.mp4"
        crop_output_path = BASE_DIR / "data" / "curry_24_crop_data" / f"{game_id}_{event_id}_crop.mp4"

        if not video_path.exists():
            print(f"[SKIP] 파일 없음: {video_path.name}")
            failed_rows.append({
                "GAME_ID": game_id,
                "EVENTNUM": event_id,
                "reason": "file_not_found",
            })
            continue

        if crop_output_path.exists():
            print(f"[SKIP] 이미 crop 존재: {crop_output_path.name}")
            continue

        try:
            target_clock = iso_duration_to_mmss(row["PCTIMESTRING"])

            result = al.crop_clip_around_target_clock(
                video_path=str(video_path),
                target_clock=target_clock,
                output_path=str(crop_output_path),
                cfg=cfg,
            )

            if result is None:
                print(f"[FAIL] OCR 감지 실패: {video_path.name}, target={target_clock}")
                failed_rows.append({
                    "GAME_ID": game_id,
                    "EVENTNUM": event_id,
                    "target_clock": target_clock,
                    "reason": "ocr_not_found_or_low_confidence",
                })
                continue

            print(
                f"[DONE] {idx + 1}/{len(curry_shots_df)} "
                f"{video_path.name} -> {crop_output_path.name}, "
                f"anchor={result.get('final_anchor_frame')}, "
                f"score={result.get('best_score')}"
            )

        except Exception as e:
            print(f"[ERROR] {video_path.name}: {e}")
            failed_rows.append({
                "GAME_ID": game_id,
                "EVENTNUM": event_id,
                "reason": str(e),
            })
            continue

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_log_path, index=False)
        print(f"[LOG] failed log saved to {failed_log_path}")