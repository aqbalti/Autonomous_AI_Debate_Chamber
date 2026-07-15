"""
Feature Extractor — Converts debate text into numeric ML features.

Why these features?
- word_count / argument_length: longer, denser arguments score higher
- sentence_count: structured thinking shows more debate preparation
- avg_word_length: sophisticated vocabulary = deeper analytical thought
- lexical_diversity / unique_word_ratio: richer language = stronger framing
- avg_sentence_length: well-formed long sentences show structured reasoning
- sentiment_score: positively-framed arguments are more persuasive
- complexity_score: long words + long sentences = analytical depth
- question_count: rhetorical questions engage and destabilize opponents

Why NOT use NLTK/spaCy: avoids heavy installation dependency for submission.
The lexicon-based approach is sufficient for scoring relative debate quality.
"""
import re
import pandas as pd


FEATURE_COLS = [
    "word_count", "sentence_count", "avg_word_length", "lexical_diversity",
    "unique_word_ratio", "avg_sentence_length", "sentiment_score",
    "complexity_score", "question_count", "argument_length"
]

_POSITIVE = {"evidence", "proves", "demonstrates", "clearly", "supports",
             "benefits", "success", "effective", "shows", "truth", "fact",
             "logical", "rational", "valid", "strong", "research", "data",
             "study", "consistent", "proven", "empirical"}
_NEGATIVE  = {"fails", "wrong", "flawed", "ignores", "contradicts",
              "misleading", "incorrect", "false", "weak", "error",
              "assumption", "unproven", "simplistic", "naive"}


class FeatureExtractor:
    """Extracts 10 linguistic features from raw debate text."""

    def extract(self, text: str) -> dict:
        words     = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        unique    = set(w.lower().strip(".,!?;:\"'") for w in words)

        wc  = len(words)
        sc  = max(len(sentences), 1)
        awl = sum(len(w) for w in words) / max(wc, 1)
        ld  = len(unique) / max(wc, 1)
        uwr = ld
        asl = wc / sc
        ss  = self._sentiment(unique)
        cs  = min((awl / 10) * 0.4 + (asl / 30) * 0.4 +
                  sum(1 for w in words if len(w) > 7) / max(wc, 1) * 0.2, 1.0)
        qc  = text.count("?")

        return {
            "word_count":          wc,
            "sentence_count":      sc,
            "avg_word_length":     round(awl, 3),
            "lexical_diversity":   round(ld,  3),
            "unique_word_ratio":   round(uwr, 3),
            "avg_sentence_length": round(asl, 3),
            "sentiment_score":     round(ss,  3),
            "complexity_score":    round(cs,  3),
            "question_count":      qc,
            "argument_length":     wc,
        }

    def _sentiment(self, word_set: set) -> float:
        pos = len(word_set & _POSITIVE)
        neg = len(word_set & _NEGATIVE)
        return (pos / (pos + neg)) if (pos + neg) > 0 else 0.5

    def to_dataframe(self, features: dict) -> pd.DataFrame:
        return pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
