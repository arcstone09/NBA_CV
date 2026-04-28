"""
data_pipe_split_csv.py
../../data/curry_24_crop_data에 존재하는 curry의 24/25 season 모든 슛 시도에 대한 data csv로 저장
"""
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, playbyplayv3, videoevents
import data_pipe_download as dpd
from pathlib import Path
import time
import pandas as pd
import numpy as np

BASE_DIR = Path("/workspace/NBA_CV")
#BASE_DIR = Path("/Users/arcstone/Desktop/snupi/nba_cv")
input_path = BASE_DIR / "data" / "curry_24_crop_data"

# split 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

def video_exists(row):
    video_file = input_path / row['VIDEO_PATH']
    return video_file.exists()

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
    game_ids = game_ids_2025[0:10]

    curry_shots_df = dpd.extract_player_shots_all_games(game_ids, curry_id)

    # 3. 
    ## GAME_ID, EVENTNUM, EVENTMSGTYPE, 
    ## video_path, label, game_id, event_id, split
    curry_shots_df = curry_shots_df[['GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE']]
    curry_shots_df = curry_shots_df.rename(columns={'EVENTMSGTYPE':'LABEL'})
    curry_shots_df["GAME_ID"] = curry_shots_df["GAME_ID"].astype(str).str.zfill(10)
    curry_shots_df["EVENTNUM"] = curry_shots_df["EVENTNUM"].astype(int).astype(str)
    curry_shots_df['LABEL'] = curry_shots_df['LABEL'] % 2
    curry_shots_df['VIDEO_PATH'] = curry_shots_df.apply(
        lambda x: f"{x['GAME_ID']}_{x['EVENTNUM']}_crop.mp4", axis=1
    )
    curry_shots_df = curry_shots_df[curry_shots_df.apply(video_exists, axis=1)]
    curry_shots_df = curry_shots_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    n = len(curry_shots_df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    curry_shots_df["SPLIT"] = "test"
    curry_shots_df.loc[:n_train - 1, "SPLIT"] = "train"
    curry_shots_df.loc[n_train:n_train + n_val - 1, "SPLIT"] = "val"

    output_path = BASE_DIR / "data" / "curry_24_split.csv"
    curry_shots_df.to_csv(output_path, index=False)


