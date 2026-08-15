import nflreadpy as nfl
import pandas as pd
from src.config import START_SEASON, END_SEASON, DATA_RAW_DIR
from src.data.utils import season_range, save_raw


def fetch_schedules(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Pull NFL schedule/game data for the given season range and return
    it as a pandas DataFrame.
    """
    team_stats = nfl.load_team_stats(seasons=season_range(start_season,end_season))
    team_stats_pd = team_stats.to_pandas()
    return team_stats_pd

if __name__ == "__main__":
    df = fetch_schedules(START_SEASON, END_SEASON)
    save_raw(df, "team_stats.csv", DATA_RAW_DIR)