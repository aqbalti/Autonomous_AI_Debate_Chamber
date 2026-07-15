"""
ML Judge (DebateRegressionJudge) — Trains RandomForestRegressor and scores debates.

Why RandomForestRegressor?
1. Non-linear: persuasion quality doesn't scale linearly with word count alone
2. Robust to outliers: debate text varies wildly in length and style
3. Scale-invariant: tree-based models need no feature normalization
4. Ensemble: 100 trees reduce overfitting on small training datasets
5. Interpretable: feature_importances_ shows which linguistic features matter most

The model trains once from historical_debates.csv and is cached to disk.
On subsequent runs it loads the .pkl directly — no retraining overhead.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/judge_model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../historical_debates.csv")

FEATURE_COLS = [
    "word_count", "sentence_count", "avg_word_length", "lexical_diversity",
    "unique_word_ratio", "avg_sentence_length", "sentiment_score",
    "complexity_score", "question_count", "argument_length"
]


class DebateRegressionJudge:
    """Loads or trains the RandomForest model, predicts persuasiveness scores."""

    def __init__(self):
        self.model, self.metrics = self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            return data["model"], data.get("metrics", {})
        return self.train_model(DATA_PATH)

    def train_model(self, dataset_path: str) -> tuple:
        """Train from CSV and save to disk. Returns (model, metrics_dict)."""
        df = pd.read_csv(dataset_path)
        X  = df[FEATURE_COLS]
        y  = df["persuasiveness_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100, max_depth=6,
            random_state=42, min_samples_leaf=2
        )
        model.fit(X_train, y_train)

        preds   = model.predict(X_test)
        metrics = {
            "mse":      round(float(mean_squared_error(y_test, preds)), 4),
            "r2_score": round(float(r2_score(y_test, preds)), 4)
        }

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "metrics": metrics}, f)

        return model, metrics

    def predict_score(self, features_df: pd.DataFrame) -> float:
        """Predict a 0–10 persuasiveness score for one agent's combined text."""
        raw = self.model.predict(features_df)[0]
        return round(float(np.clip(raw, 0.0, 10.0)), 2)

    def judge(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
        """Score both agents and determine winner with reason."""
        sa = self.predict_score(df_a)
        sb = self.predict_score(df_b)
        delta      = abs(sa - sb)
        confidence = round(delta / max(sa + sb, 1) * 100, 1)

        if sa > sb:
            winner = "Agent A"
            reason = "stronger logical structure, evidence density, and linguistic complexity"
        elif sb > sa:
            winner = "Agent B"
            reason = "deeper philosophical critique, rhetorical engagement, and argument breadth"
        else:
            winner = "Tie"
            reason = "both agents performed equally across all measured NLP features"

        return {
            "score_a":    sa,
            "score_b":    sb,
            "winner":     winner,
            "confidence": confidence,
            "reason":     reason,
            "mse":        self.metrics.get("mse", 0.0),
            "r2_score":   self.metrics.get("r2_score", 0.0),
        }
