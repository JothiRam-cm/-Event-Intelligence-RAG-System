import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/V_EVENT_DETAILS_202512311554.csv")


def load_event_data():
    """
    Load the event CSV and perform basic inspection.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("✅ CSV Loaded Successfully")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample Rows:")
    print(df.head(3))

    return df


if __name__ == "__main__":
    load_event_data()
