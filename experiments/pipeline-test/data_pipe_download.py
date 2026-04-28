"""
data_pipe_download.py
../../data/curry_24_raw_data에 curry의 24/25 season 모든 슛 시도 pbp 영상 다운
"""

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, playbyplayv3, videoevents
import pandas as pd
import time
from datetime import datetime
import random
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path("/workspace/NBA_CV")
#BASE_DIR = Path("/Users/arcstone/Desktop/snupi/nba_cv")
RAW_DATA_DIR = BASE_DIR / "data" / "curry_24_raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

#BASE_DIR = Path(__file__).resolve().parents[2]
#RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
#RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

API_HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Priority": "u=3, i",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
}

VIDEO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
    "Referer": "https://www.nba.com/",
    "Range": "bytes=0-",
    "Connection": "keep-alive",
}

FAILED_DOWNLOADS_CSV = RAW_DATA_DIR / "failed_downloads.csv"


def build_session(default_headers: dict) -> requests.Session:
    session = requests.Session()
    session.headers.update(default_headers)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


API_SESSION = build_session(API_HEADERS)
VIDEO_SESSION = build_session(VIDEO_HEADERS)

# 1. 전체 경기 목록 수집
## 1.1 Stephen Curry player_id 찾기
def get_player_id(full_name: str) -> int:
    candidates = players.find_players_by_full_name(full_name)
    if not candidates:
        raise ValueError(f"No player found for name={full_name}")
    return candidates[0]["id"]

## 1.2 PlayerGameLog 호출
def fetch_player_gamelog(player_id: int, season: str, season_type: str = "Regular Season",
                         max_retry: int = 5, sleep_sec: float = 1.2) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            df = gl.get_data_frames()[0]
            return df
        except Exception as e:
            last_err = e
            # 레이트리밋/일시 오류 대비
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Failed to fetch PlayerGameLog after {max_retry} retries: {last_err}")

# 2. 경기별 PBP에서 슛 시도 이벤트 전부 뽑기
## 2.1 PlayByPlayV2fh 이벤트 테이블 가져오기

def fetch_pbp(game_id: str, max_retry=5, timeout=60):
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(
                game_id=game_id,
                start_period=1,
                end_period=10,  # OT 대비
                timeout=timeout
            )
            # V3도 get_data_frames() 지원: 첫 DF가 PlayByPlay인 경우가 보통
            dfs = pbp.get_data_frames()
            if not dfs or dfs[0].empty:
                raise RuntimeError("empty response dataframe")
            return dfs[0]
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt + random.uniform(0.0, 0.4))
    raise RuntimeError(f"PBP(V3) fetch failed for {game_id}: {last_err}")

## 2.2 Curry 슛 이벤트 필터링
def extract_player_shots_from_game(game_id, player_id):
    df = fetch_pbp(game_id)

    # 슛 이벤트: 1 = Made Shot, 2 = Missed Shot
    shot_df = df[
        (df["personId"] == player_id) &
        (df["isFieldGoal"] == 1) &
        (df["shotResult"].notna())
    ].copy()

    if shot_df.empty:
        return pd.DataFrame()

    # 필요한 컬럼만 정리
    ## 슛 시도 시각도 넣어야 할듯!!
    result = pd.DataFrame({
        "GAME_ID": shot_df["gameId"],
        "EVENTNUM": shot_df["actionNumber"],      # V2 스타일로 이름만 맞춤
        "PERIOD": shot_df["period"],
        "PCTIMESTRING": shot_df["clock"],         # 예: "PT11M32.00S" 형태일 수 있음
        "EVENTMSGTYPE": shot_df["shotResult"].map({"Made": 1, "Missed": 2}).fillna(-1).astype(int),
        "VIDEO_AVAILABLE_FLAG": shot_df["videoAvailable"],
        "DESCRIPTION": shot_df["description"],
    })

    return result

## 2.3 여러 경기 처리
def extract_player_shots_all_games(game_ids, player_id):
    all_shots = []

    for i, game_id in enumerate(game_ids):
        print(f"[{i+1}/{len(game_ids)}] Processing {game_id}")

        try:
            game_shots = extract_player_shots_from_game(game_id, player_id)
        except Exception as e:
            print(f"[FAIL] extract shots from {game_id}: {e}")
            game_shots = pd.DataFrame()

        if not game_shots.empty:
            all_shots.append(game_shots)

        # 서버 차단 방지
        time.sleep(random.uniform(1.5, 3.0))

    if not all_shots:
        return pd.DataFrame()

    return pd.concat(all_shots, ignore_index=True)

# 3. PBP 영상 다운
# 3.1 GameID, GameEventID가 주어졌을 때 해당 PBP 이벤트 영상 다운
def download_pbp(game_id: str, event_id: int, max_retry: int = 4):
    url = "https://stats.nba.com/stats/videoeventsasset"

    params = {
        "GameID": game_id,
        "GameEventID": event_id
    }

    headers = API_HEADERS

    filename = RAW_DATA_DIR / f"{game_id}_{event_id}.mp4"

    if filename.exists() and filename.stat().st_size > 0:
        print(f"skip existing: {filename.name}")
        return str(filename)

    last_err = None

    for attempt in range(1, max_retry + 1):
        try:
            r = API_SESSION.get(url, params=params, headers=headers, timeout=(10, 60))
            r.raise_for_status()

            data = r.json()

            meta = data.get("resultSets", {}).get("Meta", {})
            video_urls = meta.get("videoUrls", [])

            if not video_urls:
                raise RuntimeError(f"No video URL found in response for {game_id}_{event_id}")

            preferred = None
            for item in video_urls:
                if item.get("lurl"):
                    preferred = item["lurl"]
                    break
            if preferred is None:
                for item in video_urls:
                    if item.get("murl"):
                        preferred = item["murl"]
                        break
            if preferred is None:
                for item in video_urls:
                    if item.get("surl"):
                        preferred = item["surl"]
                        break
            if preferred is None:
                raise RuntimeError(f"No downloadable mp4 URL found for {game_id}_{event_id}")

            video_url = preferred

            temp_filename = filename.with_suffix(".part")

            with VIDEO_SESSION.get(video_url, stream=True, headers=VIDEO_HEADERS, timeout=(10, 180)) as r:
                print("status:", r.status_code)
                r.raise_for_status()

                total = 0
                with open(temp_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):  # 256KB
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                            #print(f"downloaded: {total / 1024 / 1024:.2f} MB")

            if temp_filename.stat().st_size == 0:
                temp_filename.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded file is empty for {game_id}_{event_id}")

            temp_filename.replace(filename)
            return str(filename)

        except Exception as e:
            last_err = e
            if filename.with_suffix(".part").exists():
                filename.with_suffix(".part").unlink(missing_ok=True)
            sleep_sec = min(3.0 * attempt + random.uniform(0.5, 1.5), 15.0)
            print(f"[retry {attempt}/{max_retry}] {game_id}_{event_id}: {e}")
            time.sleep(sleep_sec)

    raise RuntimeError(f"download_pbp failed for {game_id}_{event_id}: {last_err}")

# 3.2 GameID, GameEventID가 주어졌을 때 해당 PBP 이벤트 영상 다운
def download_pbps_by_shots_df(player_shots_df: pd.DataFrame):
    if player_shots_df is None or player_shots_df.empty:
        print("No shot events to download.")
        return

    required_cols = {"GAME_ID", "EVENTNUM"}
    missing = required_cols - set(player_shots_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = player_shots_df.copy()

    if "VIDEO_AVAILABLE_FLAG" in work_df.columns:
        work_df = work_df[work_df["VIDEO_AVAILABLE_FLAG"] == 1].copy()

    work_df = work_df.drop_duplicates(subset=["GAME_ID", "EVENTNUM"]).reset_index(drop=True)

    total = len(work_df)
    failed_rows = []

    for i, row in enumerate(work_df.itertuples(index=False), start=1):
        game_id = str(row.GAME_ID)
        event_id = int(row.EVENTNUM)

        print(f"[{i}/{total}] downloading {game_id}_{event_id}.mp4")

        try:
            download_pbp(game_id, event_id)
        except Exception as e:
            print(f"[FAIL] {game_id}_{event_id}: {e}")
            failed_rows.append({
                "GAME_ID": game_id,
                "EVENTNUM": event_id,
                "ERROR": str(e),
            })

        # 서버 부담/차단 방지
        time.sleep(random.uniform(2.0, 4.0))

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(FAILED_DOWNLOADS_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved failed download log to: {FAILED_DOWNLOADS_CSV}")

# 4. 중계화면 시각 vs PBP 기록 시각 비교로, 슛 릴리즈 프레임 찾기
    
## 전체 실행 
if __name__ == "__main__":
    # 1. 전체 경기 목록 수집
    player_name = "Stephen Curry"
    curry_id = get_player_id(player_name)
    print("Curry player_id =", curry_id)  # 보통 201939

    seasons = ["2024-25"]
    season_type = "Regular Season"  # 필요하면 "Playoffs"도 별도 호출

    all_logs = []
    for s in seasons:
        df = fetch_player_gamelog(curry_id, season=s, season_type=season_type)
        df["SEASON"] = s
        all_logs.append(df)
        time.sleep(1.0)  # 서버 부담 줄이기

    logs = pd.concat(all_logs, ignore_index=True)
    logs = logs.sort_values("GAME_DATE")


    # Game_ID 리스트
    game_ids_2025 = logs["Game_ID"].unique().tolist()

    # 2. 경기별 PBP에서 슛 시도 이벤트 전부 뽑기
    game_ids = game_ids_2025

    curry_shots_df = extract_player_shots_all_games(game_ids, curry_id)

    curry_shots_df.to_csv("curry_2024_25_all_shots.csv", index=False, encoding="utf-8-sig")

    # 3. PBP 영상 다운
    download_pbps_by_shots_df(curry_shots_df)