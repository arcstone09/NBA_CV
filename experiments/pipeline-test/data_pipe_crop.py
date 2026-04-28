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

#BASE_DIR = Path("/Users/arcstone/Desktop/snupi/nba_cv")
#input_path = BASE_DIR / "data" / "curry_24_raw_data"
#output_path = BASE_DIR / "data" / "curry_24_crop_data"


def iso_duration_to_mmss(s):
    match = re.match(r'PT(\d+)M([\d.]+)S', s)
    if not match:
        raise ValueError("Invalid format")

    minutes = int(match.group(1))
    seconds = int(float(match.group(2)))  # 36.00 → 36

    return f"{minutes:02d}:{seconds:02d}"

# 테스트
#print(iso_duration_to_mmss("PT08M36.00S"))  # 08:36

if __name__ == "__main__":
    # 1. 전체 경기 목록 수집
    player_name = "Stephen Curry"
    curry_id = dpd.get_player_id(player_name)
    print("Curry player_id =", curry_id)  # 보통 201939

    seasons = ["2024-25"]
    season_type = "Regular Season"  # 필요하면 "Playoffs"도 별도 호출

    all_logs = []
    for s in seasons:
        df = dpd.fetch_player_gamelog(curry_id, season=s, season_type=season_type)
        df["SEASON"] = s
        all_logs.append(df)
        time.sleep(1.0)  # 서버 부담 줄이기

    logs = pd.concat(all_logs, ignore_index=True)
    logs = logs.sort_values("GAME_DATE")


    # Game_ID 리스트
    game_ids_2025 = logs["Game_ID"].unique().tolist()

    # 2. 경기별 PBP에서 슛 시도 이벤트 전부 뽑기
    game_ids = game_ids_2025[:3]

    curry_shots_df = dpd.extract_player_shots_all_games(game_ids, curry_id)
    
    # 3. 실제 raw_data에 pbp 존재하는 영상에 한 해 crop하고 저장.
    
    for _, row in curry_shots_df.iterrows():
        game_id = row["GAME_ID"]
        event_id = row["EVENTNUM"]

        video_path = input_path / f"{game_id}_{event_id}.mp4"
        crop_output_path = BASE_DIR / "data" / "curry_24_crop_data" / f"{game_id}_{event_id}_crop.mp4"

        # 파일이 없으면 시도하지 않음
        if not video_path.exists():
            print(f"[SKIP] 파일 없음: {video_path.name}")
            continue

        target_clock = iso_duration_to_mmss(row["PCTIMESTRING"])
        
        result = al.crop_clip_around_target_clock(
            video_path=video_path,
            target_clock=target_clock,
            output_path=crop_output_path
        )

        print(f"[DONE] {video_path.name} -> {crop_output_path.name}, result={result}")
        


    