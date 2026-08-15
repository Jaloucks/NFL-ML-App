import nflreadpy as nfl
import pandas as pd
from src.config import START_SEASON, END_SEASON, DATA_RAW_DIR
from src.data.utils import season_range, save_raw


def fetch_schedules(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Pull NFL schedule/game data for the given season range and return
    it as a pandas DataFrame.
    """
    rosters = nfl.load_rosters_weekly(seasons=season_range(start_season,end_season))
    rosters_pd = rosters.to_pandas()
    return rosters_pd

if __name__ == "__main__":
    df = fetch_schedules(START_SEASON, END_SEASON)
    save_raw(df, "rosters.csv", DATA_RAW_DIR)