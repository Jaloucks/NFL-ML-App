import pandas as pd

EXCLUDE_COLS = [
    "season", "week",
    "result", "home_score", "away_score",
    "home_moneyline", "away_moneyline", "spread_line", "total_line",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
    "ftn", "pff", "gsis", "espn", "pfr", "old_game_id", "nfl_detail_id",
    "home_passing_cpoe", "away_passing_cpoe",
    "wind", "temp",
    "home_fg_missed_0_19", "away_fg_missed_0_19",
]

def split_train_test(df: pd.DataFrame, test_seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a game-level dataset into train/test sets by season, so that
    test data always represents games chronologically after training data.
    """

    test_df = df[df["season"].isin(test_seasons)]
    train_df = df[~df["season"].isin(test_seasons)]
    return train_df, test_df

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns only, minus target/leakage/betting-market columns."""
    return [c for c in df.select_dtypes(include="number").columns if c not in EXCLUDE_COLS]

def build_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result"):
    """Drop rows with any missing feature (cold-start weeks) and split into X, y."""
    clean = df.dropna(subset=feature_cols)
    return clean[feature_cols], clean[target_col]

def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize features to mean 0, std 1, using statistics computed from
    X_train only (never X_test), then apply those same stats to both.
    """
    mean_X_train = X_train.mean()
    std_X_train = X_train.std()
    stand_X_train = (X_train - mean_X_train) / std_X_train
    stand_X_test = (X_test - mean_X_train) / std_X_train
    return stand_X_train, stand_X_test