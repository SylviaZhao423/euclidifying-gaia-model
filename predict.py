#!/usr/bin/env python3
"""
Example: classify Gaia DR3 sources as star or galaxy.

Loads a 10-seed XGBoost ensemble and predicts P(gal) for input sources.
P(gal) > 0.5: likely galaxy candidate
P(gal) > 0.9: high-confidence galaxy candidate

Multiple models available:
  q1_only      — trained on Euclid Q1 labels only (default, higher recall)
  q1_ero_mix_3..8 — augmented with ERO stars at 3:1 to 8:1 ratios
  q1_ero_mix_8.6 — all ERO stars (highest precision)

Usage:
    python predict.py                              # demo with random features
    python predict.py input.csv                    # predict with Q1-only model
    python predict.py input.csv --model q1_ero_mix_8.6  # predict with all-ERO model
"""

import sys
from pathlib import Path
import numpy as np
import xgboost as xgb

from features import FEATURES

MODEL_DIR = Path(__file__).parent / "models"
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

AVAILABLE_MODELS = {
    "q1_only": "Q1-only (default, higher recall)",
    "q1_ero_mix_3": "ERO-augmented 3:1",
    "q1_ero_mix_4": "ERO-augmented 4:1",
    "q1_ero_mix_5": "ERO-augmented 5:1",
    "q1_ero_mix_6": "ERO-augmented 6:1",
    "q1_ero_mix_7": "ERO-augmented 7:1",
    "q1_ero_mix_8": "ERO-augmented 8:1",
    "q1_ero_mix_8.6": "ERO-augmented 8.63:1 (all ERO, highest precision)",
}


def load_ensemble(model_name="q1_only"):
    """Load all 10 seed models for the specified model variant.

    Parameters
    ----------
    model_name : str
        'q1_only' (default) or 'q1_ero_mix_8.6'
    """
    model_dir = MODEL_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Available models: {list(AVAILABLE_MODELS.keys())}")
    models = []
    for seed in SEEDS:
        mp = model_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            print(f"  WARNING: {mp.name} not found, skipping")
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        models.append(m)
    print(f"Loaded {len(models)} seed models ({model_name})")
    return models


def predict_pgal(models, X):
    """Predict P(gal) as ensemble mean of all seed models.

    Parameters
    ----------
    models : list of XGBClassifier
    X : array of shape (n_sources, 14)
        The 14 Gaia features in the order defined in features.py.
        NaN values are handled internally by XGBoost.

    Returns
    -------
    p_gal : array of shape (n_sources,)
        Probability of being a galaxy (0 = star, 1 = galaxy).
    """
    probas = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probas, axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classify Gaia DR3 sources as star or galaxy")
    parser.add_argument("input", nargs="?", default=None,
                        help="CSV file with 14 feature columns (demo mode if omitted)")
    parser.add_argument("--model", default="q1_only", choices=list(AVAILABLE_MODELS.keys()),
                        help="Model variant (default: q1_only)")
    args = parser.parse_args()

    models = load_ensemble(args.model)

    if args.input:
        import pandas as pd
        df = pd.read_csv(args.input)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            print(f"Required: {FEATURES}")
            return 1
        X = df[FEATURES].values.astype(float)
        print(f"Loaded {len(X):,} sources from {args.input}")
    else:
        print("Demo mode: generating 100 random sources")
        print("(predictions will be meaningless — use real Gaia data)")
        X = np.random.randn(100, len(FEATURES))

    p_gal = predict_pgal(models, X)

    print(f"\nResults:")
    print(f"  P(gal) range: [{p_gal.min():.4f}, {p_gal.max():.4f}]")
    print(f"  P(gal) > 0.5: {(p_gal >= 0.5).sum():,} sources ({100*(p_gal >= 0.5).mean():.1f}%)")
    print(f"  P(gal) > 0.9: {(p_gal >= 0.9).sum():,} sources ({100*(p_gal >= 0.9).mean():.1f}%)")

    if args.input:
        out = args.input.replace('.csv', '_pgal.csv')
        import pandas as pd
        df['p_gal'] = p_gal
        df.to_csv(out, index=False)
        print(f"  -> Saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
