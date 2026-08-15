import nflreadpy as nfl
import pandas as pd
from src.config import START_SEASON, END_SEASON, DATA_RAW_DIR
from src.data.utils import season_range, save_raw

def fetch_player_stats(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Pull weekly NFL player statistics for the given season range and
    return as a pandas DataFrame.
    """
    player_stats = nfl.load_player_stats(seasons=season_range(start_season, end_season))
    player_stats_pd = player_stats.to_pandas()
    return player_stats_pd

if __name__ == "__main__":
    df = fetch_player_stats(START_SEASON, END_SEASON)
    save_raw(df, "player_stats.csv", DATA_RAW_DIR)