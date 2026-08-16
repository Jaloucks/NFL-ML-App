import pandas as pd
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.data.utils import save_raw

TEAM_ABBR_MAP = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LA",
}

schedules = pd.read_csv(DATA_RAW_DIR / "schedules.csv")
team_stats = pd.read_csv(DATA_RAW_DIR / "team_stats.csv")

schedules["home_team"] = schedules["home_team"].replace(TEAM_ABBR_MAP)
schedules["away_team"] = schedules["away_team"].replace(TEAM_ABBR_MAP)

schedule_teams = set(schedules["home_team"].unique()) | set(schedules["away_team"].unique())
team_stats_teams = set(team_stats["team"].unique())

# print("In schedules but not team_stats:", schedule_teams - team_stats_teams)
# print("In team_stats but not schedules:", team_stats_teams - schedule_teams)

def build_game_team_dataset(schedules: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team-week stats into schedules twice (home perspective and away
    perspective) to produce one row per game with both teams' stats attached.
    """
    team_stats_clean = team_stats.drop(columns=["season_type", "opponent_team", "game_id"])

    key_cols = ["season", "week", "team"]

    rename_map = {
        col: f"home_{col}"
        for col in team_stats_clean.columns
        if col not in key_cols
    }

    home_stats = team_stats_clean.rename(columns=rename_map)
    home_stats = home_stats.rename(columns={"team":"home_team"})

    rename_map = {
        col: f"away_{col}"
        for col in team_stats_clean.columns
        if col not in key_cols
    }

    away_stats = team_stats_clean.rename(columns=rename_map)
    away_stats = away_stats.rename(columns={"team":"away_team"})

    merged = schedules.merge(home_stats, on=["season", "week", "home_team"], how="left")
    merged = merged.merge(away_stats, on=["season", "week", "away_team"], how="left")

    return merged

if __name__ == "__main__":
    game_dataset = build_game_team_dataset(schedules, team_stats)
    print("Row count matches schedules:", len(game_dataset) == len(schedules))
    print("Nulls in home_passing_yards:", game_dataset["home_passing_yards"].isna().sum())
    print("Nulls in away_passing_yards:", game_dataset["away_passing_yards"].isna().sum())
    save_raw(game_dataset, "game_team_stats.csv", DATA_PROCESSED_DIR)