import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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
        "--train",
        default=str(data_dir / "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--category",
        default=str(data_dir / "category.csv"),
        help="Path to category.csv",
    )
    parser.add_argument(
        "--model_out",
        default=str(data_dir / "model.joblib"),
        help="Where to save model.joblib",
    )
    parser.add_argument(
        "--metrics_out",
        default=str(data_dir / "metrics.json"),
        help="Optional metrics JSON path",
    )
    parser.add_argument("--encoding", default="cp1251", help="CSV encoding")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = read_csv(args.train, args.encoding)
    df = build_features(df)

    if args.sample:
        df = df.sample(args.sample, random_state=args.random_state)

    y = df["category_id"].astype(int)
    X = df[["text"]].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    text_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "text"),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=None,
        class_weight="balanced",
        solver="saga",
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_val, val_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "f1_macro": float(f1_score(y_val, val_pred, average="macro")),
        "f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
        "top_k_accuracy": {
            "k": 3,
            "value": float(top_k_accuracy_score(y_val, val_proba, k=3, labels=labels)),
        },
        "confusion_matrix": {
            "labels": [int(label) for label in labels],
            "matrix": cm.tolist(),
        },
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }

    category_df = read_csv(args.category, args.encoding)
    category_map = dict(zip(category_df["category_id"], category_df["name"]))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "category_map": category_map}, args.model_out)

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
