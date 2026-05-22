import argparse
from pathlib import Path

import joblib
import pandas as pd


def read_csv(path: str, encoding: str) -> pd.DataFrame:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return pd.read_csv(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["description"].fillna("")
    df = df.copy()
    df["text"] = text.str.strip()
    return df


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(data_dir / "model.joblib"),
        help="Path to model.joblib",
    )
    parser.add_argument(
        "--data",
        default=str(data_dir / "test.csv"),
        help="Path to test.csv",
    )
    parser.add_argument(
        "--out",
        default=str(data_dir / "predictions.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--encoding", default="cp1251", help="CSV encoding")
    args = parser.parse_args()

    payload = joblib.load(args.model)
    pipeline = payload["pipeline"]
    category_map = payload.get("category_map", {})

    df = read_csv(args.data, args.encoding)
    df = build_features(df)

    X = df[["text"]].copy()
    pred = pipeline.predict(X)

    result = pd.DataFrame({
        "item_id": df["item_id"],
        "category_id": pred,
    })

    if category_map:
        result["category_name"] = result["category_id"].map(category_map)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding=args.encoding)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
