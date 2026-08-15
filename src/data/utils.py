from pathlib import Path
import pandas as pd


def season_range(start_season: int, end_season: int) -> list[int]:
    """Build an inclusive list of seasons from start_season to end_season."""
    return list(range(start_season, end_season + 1))


def save_raw(df: pd.DataFrame, filename: str, data_raw_dir: Path) -> None:
    """Save a DataFrame to the raw data directory and print a confirmation."""
    path = data_raw_dir / filename
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")