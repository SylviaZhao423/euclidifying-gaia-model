#!/usr/bin/env python3
"""
Deploy both XGBoost ensembles onto the full Gaia DR3 source catalogue.

Streams gaia_dr3.gaia_source in random_index range batches, computes the 3
derived features inline in SQL, loads both ensembles, predicts P(galaxy)
per ensemble, and COPYs results into a WSDB output table.

Usage:
    PGUSER=<wsdb_user> python deploy_gaia.py [options]

Resume support: re-run with --start-ri set to one past the highest
random_index already in the output table.

Options:
    --test-mode            Process only the first batch (≈5M rows).
    --batch-size N         random_index range per batch (default 5,000,000).
    --start-ri N           Start random_index (default 0).
    --end-ri N             End random_index (default 2,000,000,000).
    --drop-existing        DROP the output table before writing.
    --output-table NAME    Output table (default {PGUSER}.gaia_q1_pgal_predictions).
    --mag-cut F            phot_g_mean_mag >= cut (default 17).
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, compute_derived_features_sql

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MODEL_DIR = Path(__file__).parent / "models"
SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]


def load_ensemble(model_name: str) -> list[xgb.Booster]:
    model_dir = MODEL_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    boosters = []
    for seed in SEEDS:
        p = model_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not p.exists():
            print(f"  WARNING: {p.name} missing, skipping")
            continue
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    print(f"  [{model_name}] loaded {len(boosters):>2} seed models")
    return boosters


def predict_ensemble(boosters, X):
    dmat = xgb.DMatrix(X)
    probs = np.stack([b.predict(dmat) for b in boosters], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


def _build_select_expr():
    """Build the SELECT column expression list against gaia_dr3.gaia_source."""
    derived = compute_derived_features_sql()
    cols = []
    for f in FEATURES:
        if f in derived:
            cols.append(f"{derived[f]} AS {f}")
        else:
            cols.append(f)
    return ",\n               ".join(cols)


def fetch_batch(cur, ri_lo, ri_hi, mag_cut):
    feature_select = _build_select_expr()
    cur.execute(f"""
        SELECT source_id, random_index, ra, dec, phot_g_mean_mag,
               {feature_select}
        FROM gaia_dr3.gaia_source
        WHERE random_index >= {ri_lo}
          AND random_index <  {ri_hi}
          AND phot_g_mean_mag IS NOT NULL
          AND phot_g_mean_mag >= {mag_cut}
    """)
    rows = cur.fetchall()
    cols = ["source_id", "random_index", "ra", "dec", "phot_g_mean_mag"] + list(FEATURES)
    df = pd.DataFrame(rows, columns=cols)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def copy_predictions(cur, table, df, q1_mean, q1_std, mix_mean, mix_std):
    buf = io.StringIO()
    sids = df["source_id"].astype(np.int64).values
    ris  = df["random_index"].astype(np.int64).values
    ras  = df["ra"].astype(float).values
    decs = df["dec"].astype(float).values
    gmag = df["phot_g_mean_mag"].astype(float).values
    for i in range(len(df)):
        buf.write(
            f"{int(sids[i])}\t{int(ris[i])}\t"
            f"{ras[i]:.10f}\t{decs[i]:.10f}\t{gmag[i]:.6f}\t"
            f"{q1_mean[i]:.6f}\t{q1_std[i]:.6f}\t"
            f"{mix_mean[i]:.6f}\t{mix_std[i]:.6f}\n"
        )
    buf.seek(0)
    cur.copy_from(buf, table.split(".", 1)[1], sep="\t",
                  columns=("source_id", "random_index", "ra", "dec",
                           "phot_g_mean_mag",
                           "q1_pgal_mean", "q1_pgal_std",
                           "mix_pgal_mean", "mix_pgal_std"))


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-mode", action="store_true",
                        help="Process only the first batch.")
    parser.add_argument("--batch-size", type=int, default=5_000_000)
    parser.add_argument("--start-ri", type=int, default=0)
    parser.add_argument("--end-ri", type=int, default=2_000_000_000)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--output-table", default=None,
                        help="Fully-qualified output table (default {PGUSER}.gaia_q1_pgal_predictions)")
    parser.add_argument("--mag-cut", type=float, default=17.0)
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    output_table = args.output_table or f"{username}.gaia_q1_pgal_predictions"
    if args.test_mode and args.output_table is None:
        output_table += "_test"

    print("=" * 70)
    print("Deploy Q1 + Q1+ERO ensembles onto Gaia DR3")
    print(f"  Source     : gaia_dr3.gaia_source")
    print(f"  Filter     : phot_g_mean_mag >= {args.mag_cut}")
    print(f"  Output     : {output_table}")
    print(f"  Batch size : {args.batch_size:,}")
    print(f"  RI range   : [{args.start_ri:,}, {args.end_ri:,})")
    print(f"  Test mode  : {args.test_mode}")
    print("=" * 70)

    print("\n[1] Loading model ensembles...")
    q1_models  = load_ensemble("q1_only")
    mix_models = load_ensemble("q1_ero_mix_8.6")

    print("\n[2] Connecting to WSDB...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()

    if args.drop_existing:
        cur.execute(f"DROP TABLE IF EXISTS {output_table};")
        print(f"  DROPped existing {output_table}")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            source_id        BIGINT PRIMARY KEY,
            random_index     BIGINT,
            ra               DOUBLE PRECISION,
            dec              DOUBLE PRECISION,
            phot_g_mean_mag  REAL,
            q1_pgal_mean     REAL,
            q1_pgal_std      REAL,
            mix_pgal_mean    REAL,
            mix_pgal_std     REAL
        );
    """)
    conn.commit()
    print(f"  Output table ready: {output_table}")

    print("\n[3] Streaming gaia_dr3.gaia_source, predicting, COPYing...")
    n_batches = 0
    n_total   = 0
    t_start   = time.time()
    ri_lo = args.start_ri
    while ri_lo < args.end_ri:
        ri_hi = min(ri_lo + args.batch_size, args.end_ri)
        t0 = time.time()
        df = fetch_batch(cur, ri_lo, ri_hi, args.mag_cut)
        n_rows = len(df)
        t_query = time.time() - t0
        if n_rows == 0:
            print(f"  ri[{ri_lo:>11,}, {ri_hi:>11,})  empty (q={t_query:.1f}s)")
            ri_lo = ri_hi
            continue

        X = df[list(FEATURES)].values.astype(np.float32)
        t1 = time.time()
        q1_mean, q1_std   = predict_ensemble(q1_models, X)
        mix_mean, mix_std = predict_ensemble(mix_models, X)
        t_pred = time.time() - t1

        t2 = time.time()
        copy_predictions(cur, output_table, df, q1_mean, q1_std, mix_mean, mix_std)
        conn.commit()
        t_copy = time.time() - t2

        elapsed = time.time() - t0
        n_batches += 1
        n_total   += n_rows
        print(f"  ri[{ri_lo:>11,}, {ri_hi:>11,})  rows={n_rows:>9,}  "
              f"(q={t_query:6.1f}s  pred={t_pred:5.1f}s  copy={t_copy:5.1f}s  "
              f"total={elapsed:6.1f}s)  cum={n_total:>12,}  "
              f"wall={(time.time()-t_start)/60:.1f}min")

        if args.test_mode:
            print("  TEST MODE — stopping after one batch.")
            break

        ri_lo = ri_hi

    if not args.test_mode and n_total > 0:
        print("\n[4] Adding q3c spatial index + ANALYZE...")
        idx_name = output_table.split(".")[-1] + "_q3c_idx"
        try:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {output_table} (q3c_ang2ipix(ra, dec));")
            cur.execute(f"CLUSTER {output_table} USING {idx_name};")
            conn.commit()
            print("  q3c index built + clustered")
        except Exception as e:
            conn.rollback()
            print(f"  q3c index step failed (non-fatal): {e}")
        cur.execute(f"ANALYZE {output_table};")
        conn.commit()

    print(f"\nDone.  {n_batches} batch(es), {n_total:,} rows  "
          f"in {(time.time()-t_start)/60:.1f} min")
    cur.execute(f"SELECT count(*) FROM {output_table};")
    print(f"Output table {output_table}: {cur.fetchone()[0]:,} total rows")
    cur.close(); conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
