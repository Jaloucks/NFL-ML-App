import pandas as pd
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.data.utils import save_raw

def compute_season_to_date_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Replace each stat column with that team's average of the same stat
    across their prior games in the same season only (excluding the
    current game), to produce leakage-safe pre-game features.
    """
    result = team_stats.copy()
    result = result.sort_values(["team", "season", "week"])

    identifier_cols = ["season", "week", "team", "season_type", "opponent_team", "game_id"]
    stat_cols = [
        c for c in result.select_dtypes(include="number").columns
        if c not in identifier_cols
    ]

    for col in stat_cols:
        result[col] = result.groupby(["team", "season"])[col].transform(
            lambda s: s.expanding().mean().shift(1)
        )

    return result

def compute_rest_days(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-team game log from schedules, compute each team's days of
    rest since their previous game within the same season, and merge the
    result back onto schedules as home_rest_days / away_rest_days.
    """
    schedules = schedules.copy()
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    home_log = schedules[["game_id", "season", "gameday", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away_log = schedules[["game_id", "season", "gameday", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    team_game_log = pd.concat([home_log, away_log], ignore_index=True)

    team_game_log = team_game_log.sort_values(["team", "season", "gameday"])
    team_game_log["rest_days"] = (
        team_game_log.groupby(["team", "season"])["gameday"].diff().dt.days
    )

    home_rest = team_game_log[["game_id", "team", "rest_days"]].rename(
        columns={"team": "home_team", "rest_days": "home_rest_days"}
    )
    away_rest = team_game_log[["game_id", "team", "rest_days"]].rename(
        columns={"team": "away_team", "rest_days": "away_rest_days"}
    )

    schedules = schedules.merge(home_rest, on=["game_id", "home_team"], how="left")
    schedules = schedules.merge(away_rest, on=["game_id", "away_team"], how="left")

    return schedules

if __name__ == "__main__":
    team_stats = pd.read_csv(DATA_RAW_DIR / "team_stats.csv")
    team_history = compute_season_to_date_stats(team_stats)
    save_raw(team_history, "team_history.csv", DATA_PROCESSED_DIR)

    schedules = pd.read_csv(DATA_RAW_DIR / "schedules.csv")
    schedules_with_rest = compute_rest_days(schedules)
    save_raw(schedules_with_rest, "schedules_with_rest.csv", DATA_PROCESSED_DIR)