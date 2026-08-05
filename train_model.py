import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "data" / "investigative_workflow.db"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "lead_classifier.joblib"


def load_labelled_leads():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT title, description, training_label
            FROM leads
            WHERE training_label IS NOT NULL
              AND training_label != ''
            """,
            con,
        )

    df["combined_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df = df[df["combined_text"].str.strip() != ""]
    return df


def main():
    df = load_labelled_leads()

    if len(df) < 12:
        raise ValueError("Label at least 12 leads before training the model.")

    class_counts = df["training_label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two different training labels are required.")
    if class_counts.min() < 2:
        raise ValueError("Each label needs at least two examples for a basic split.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["training_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["training_label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Labelled records: {len(df)}")
    print("Class distribution:")
    print(class_counts)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
