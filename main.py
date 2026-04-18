"""
main.py — CLI Entry Point
==========================
Run the full ML pipeline from the command line, or just download data.

Usage:
    python main.py pipeline     # run full ML pipeline (prints metrics)
    python main.py download     # download BV-BRC CSVs automatically
    python main.py streamlit    # launch Streamlit dashboard
"""

import sys
import argparse


def run_pipeline():
    """Execute the complete ML pipeline end-to-end."""
    print("=" * 60)
    print("  AMR Prediction — A. baumannii / Meropenem")
    print("  COMSATS University Islamabad | ALC 354")
    print("=" * 60)

    # ── 1. DATA ──────────────────────────────────────────────
    print("\n[PHASE 2] Dataset & Preprocessing")
    from controllers.data_controller import DataController
    dc = DataController(use_synthetic_fallback=True)
    dc.load().preprocess()
    X_tr, X_te, y_tr, y_te, features, X_tr_bal, y_tr_bal = dc.get_splits()
    print(f"  Features : {len(features)}")
    print(f"  Samples  : {dc.n_samples:,}")
    print(f"  Source   : {dc.data_source}")
    bal = dc.class_balance
    print(f"  Resistant: {bal['resistant']:,} ({bal['resistant_pct']:.1f}%)")

    # ── 2. CHI2 ──────────────────────────────────────────────
    chi2_df = dc.get_chi2_ranking()
    print(f"\n  Top 5 discriminative genes:")
    for _, row in chi2_df.head(5).iterrows():
        print(f"    {row['Gene']:20s}  χ²={row['Chi2_Score']:.2f}")

    # ── 3. TRAIN ─────────────────────────────────────────────
    print("\n[PHASE 3] Model Development")
    from controllers.train_controller import TrainController
    tc = TrainController()
    tc.train(X_tr_bal, y_tr_bal)
    tc.cross_validate(X_tr, y_tr)

    # ── 4. EVAL ──────────────────────────────────────────────
    print("\n[PHASE 4] Evaluation & Analysis")
    from controllers.eval_controller import EvalController
    ec = EvalController()
    ec.evaluate_all(tc.models, X_te, y_te)

    print("\n=== FINAL METRICS TABLE ===")
    print(ec.metrics_df.to_string())

    best = ec.best_model_name()
    print(f"\n  Best model: {best}")
    ec.print_report(best)

    # ── 5. SHAP ──────────────────────────────────────────────
    print("\n[PHASE 4 cont.] SHAP Explainability")
    from views.shap_views import SHAPAnalyser
    sa = SHAPAnalyser(tc.models["XGBoost"], features)
    sa.compute(X_te)
    print("\n  Top 10 globally important genes (SHAP):")
    for gene, val in sa.mean_abs_shap.head(10).items():
        print(f"    {gene:20s}  mean|SHAP|={val:.4f}")

    print("\n✅ Pipeline complete. Run 'python main.py streamlit' for the dashboard.")
    return tc.models, ec, sa, dc


def download_data():
    """Download real BV-BRC data automatically via their public API."""
    import requests
    import time
    from config import BVBRC, DataPaths

    DataPaths.RAW_DIR.mkdir(parents=True, exist_ok=True)

    downloads = [
        ("AMR Phenotype (Meropenem)", BVBRC.AMR_CSV_URL, DataPaths.AMR_PHENOTYPE),
        ("Specialty Genes (AMR)",     BVBRC.SP_GENES_URL,  DataPaths.SP_GENES),
    ]

    headers = {
        "Accept": "text/csv",
        "User-Agent": "AMR-ML-Project/1.0 (student research; COMSATS University)",
    }

    for name, url, path in downloads:
        if path.exists():
            print(f"  ✅ Already exists: {path.name}")
            continue

        print(f"\n  Downloading: {name}")
        print(f"  URL: {url[:80]}...")
        try:
            resp = requests.get(url, headers=headers, timeout=120, stream=True)
            resp.raise_for_status()

            total = 0
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
                    if total % (1024 * 1024) == 0:
                        print(f"    {total // 1024 // 1024} MB downloaded...")

            print(f"  ✅ Saved: {path}  ({total / 1024:.0f} KB)")
            time.sleep(2)  # be polite to BV-BRC servers
        except requests.RequestException as e:
            print(f"  ❌ Download failed: {e}")
            print(f"  → Manual download instructions in DATA_GUIDE.md")


def launch_streamlit():
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMR ML Project — ALC 354")
    parser.add_argument("command", choices=["pipeline", "download", "streamlit"],
                         nargs="?", default="pipeline")
    args = parser.parse_args()

    if args.command == "download":
        print("=== BV-BRC Data Downloader ===")
        download_data()
    elif args.command == "streamlit":
        launch_streamlit()
    else:
        run_pipeline()