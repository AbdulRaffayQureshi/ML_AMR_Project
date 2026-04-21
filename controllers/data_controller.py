"""
controllers/data_controller.py  (v3 — MIC recovery + adaptive thresholds)
==========================================================================
ROOT CAUSE OF "9,146 DROPPED ROWS":
  BV-BRC stores resistance in TWO ways:
    (a) "Resistant Phenotype" column  → direct label "Resistant"/"Susceptible"
    (b) "Measurement Value" column    → raw MIC number  e.g. 16.0 mg/L
  Many rows have a MIC value but a blank Phenotype label.

FIX: _encode_row() tries the phenotype label first.
  If blank/NaN it reads the MIC number and applies CLSI M100 breakpoints:
    Meropenem vs A. baumannii (CLSI M100-S32 / EUCAST 2023)
      Resistant   : MIC ≥ 8  mg/L
      Susceptible : MIC ≤ 2  mg/L   (Intermediate 4 mg/L → Susceptible)
  This recovers thousands of rows that were previously dropped.

ALSO FIXED:
  - Adaptive gene prevalence threshold (auto-relaxes for small datasets)
  - Broader property filter for sp_genes.csv
  - Clear per-step counts printed at every stage
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DataPaths, BioSettings, PrepSettings

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  MEROPENEM BREAKPOINTS  (CLSI M100 / EUCAST 2023 for Acinetobacter)
# ─────────────────────────────────────────────────────────────────────────────
MIC_RESISTANT_GE   = 8.0   # MIC ≥ 8 mg/L  → Resistant
MIC_SUSCEPTIBLE_LE = 2.0   # MIC ≤ 2 mg/L  → Susceptible
# 2 < MIC < 8 → Intermediate → treated as Susceptible (conservative)


def _parse_mic(val) -> float | None:
    """Parse raw MIC from BV-BRC: handles '16.0', '>16', '<=2', '≥32', NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    s = s.replace("≥",">=").replace("≤","<=")
    s = re.sub(r"[mg/L\suμ]","", s, flags=re.IGNORECASE)
    s = re.sub(r"^[><=]+","", s)  # strip inequality prefix
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  COLUMN ALIAS MAP  (every known BV-BRC export format)
# ─────────────────────────────────────────────────────────────────────────────
_COL_ALIASES: dict[str, str] = {
    # AMR phenotype file
    "genome id"               : "genome_id",
    "genomeid"                : "genome_id",
    "genome_id"               : "genome_id",
    "genome name"             : "genome_name",
    "genomename"              : "genome_name",
    "genome_name"             : "genome_name",
    "taxon id"                : "taxon_id",
    "antibiotic"              : "antibiotic",
    "drug"                    : "antibiotic",
    "compound"                : "antibiotic",
    "resistant phenotype"     : "resistant_phenotype",
    "resistantphenotype"      : "resistant_phenotype",
    "resistant_phenotype"     : "resistant_phenotype",
    "phenotype"               : "resistant_phenotype",
    "sir"                     : "resistant_phenotype",
    "interpretation"          : "resistant_phenotype",
    "measurement"             : "measurement",
    "measurement value"       : "measurement_value",   # ← KEY: raw MIC number
    "measurement_value"       : "measurement_value",
    "mic"                     : "measurement_value",
    "measurement sign"        : "measurement_sign",    # ← KEY: >, <, =
    "measurement_sign"        : "measurement_sign",
    "measurement unit"        : "measurement_unit",
    "measurement_unit"        : "measurement_unit",
    "laboratory typing method": "laboratory_typing_method",
    "laboratory_typing_method": "laboratory_typing_method",
    "testing standard"        : "testing_standard",
    "testing_standard"        : "testing_standard",
    "source"                  : "source",
    "evidence"                : "evidence",
    "pubmed"                  : "pubmed",
    # Specialty genes file
    "property"                : "property",
    "brc id"                  : "brc_id",
    "refseq locus tag"        : "refseq_locus_tag",
    "alt locus tag"           : "alt_locus_tag",
    "source id"               : "source_id",
    "source organism"         : "source_organism",
    "gene"                    : "gene",
    "gene name"               : "gene",
    "gene_name"               : "gene",
    "product"                 : "product",
    "function"                : "function_col",
    "classification"          : "classification",
    "antibiotics class"       : "antibiotics_class",
    "antibiotics"             : "antibiotics",
    "subject coverage"        : "subject_coverage",
    "query coverage"          : "query_coverage",
    "query_coverage"          : "query_coverage",
    "identity"                : "identity",
    "e-value"                 : "evalue",
    "evalue"                  : "evalue",
}


def _normalise_columns(df: pd.DataFrame, label: str = "file") -> pd.DataFrame:
    """Rename columns to snake_case regardless of BV-BRC export format."""
    rename_map = {}
    for col in df.columns:
        key  = col.strip().lower().replace("-"," ").replace("_"," ")
        key2 = col.strip().lower()
        target = _COL_ALIASES.get(key) or _COL_ALIASES.get(key2)
        if target and target != col:
            rename_map[col] = target
    if rename_map:
        print(f"    Column renames ({label}):")
        for old, new in rename_map.items():
            print(f"      '{old}' -> '{new}'")
    return df.rename(columns=rename_map)


def _check_required(df: pd.DataFrame, required: list, label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"\n{'='*60}\n"
            f"MISSING COLUMNS in {label}\n"
            f"{'='*60}\n"
            f"  Required : {required}\n"
            f"  Missing  : {missing}\n"
            f"  Your CSV has ({len(df.columns)} cols):\n"
            f"    {df.columns.tolist()}\n\n"
            f"  Add a mapping to _COL_ALIASES in data_controller.py (~line 55).\n"
            f"{'='*60}"
        )


# ─────────────────────────────────────────────────────────────────────────────
class DataController:
    """Complete pipeline: load → normalise → MIC-recovery → preprocess → split → balance."""

    def __init__(self, use_synthetic_fallback: bool = True):
        self.use_synthetic_fallback = use_synthetic_fallback
        self.df_raw_amr: pd.DataFrame | None = None
        self.df_raw_sp:  pd.DataFrame | None = None
        self.df_features: pd.DataFrame | None = None
        self.feature_names: list = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_bal = self.y_train_bal = None
        self._data_source = "unknown"
        self._warnings: list = []
        self.label_stats: dict = {}   # exposed to dashboard

    # ── Public API ────────────────────────────────────────────────────────

    def load(self) -> "DataController":
        amr_p = DataPaths.AMR_PHENOTYPE
        sp_p  = DataPaths.SP_GENES
        if amr_p.exists() and sp_p.exists():
            print("Real BV-BRC data detected — loading...")
            self._load_real(amr_p, sp_p)
        elif self.use_synthetic_fallback:
            print("BV-BRC CSVs not found — using calibrated synthetic dataset.")
            print(f"  Expected: {amr_p}")
            print(f"  Expected: {sp_p}")
            self._load_synthetic()
        else:
            raise FileNotFoundError(
                f"CSVs not found at {DataPaths.RAW_DIR}.\n"
                "Place amr_phenotype.csv and sp_genes.csv in that folder."
            )
        return self

    def preprocess(self) -> "DataController":
        assert self.df_features is not None, "Call .load() first."
        self._clean()
        self._split()
        if PrepSettings.USE_SMOTE and SMOTE_AVAILABLE:
            self._apply_smote()
        else:
            if not SMOTE_AVAILABLE:
                print("  imbalanced-learn not installed — skipping SMOTE")
                print("  Install: pip install imbalanced-learn")
            self.X_train_bal = self.X_train.copy()
            self.y_train_bal = self.y_train.copy()
        self._save_processed()
        return self

    def get_splits(self):
        return (self.X_train, self.X_test, self.y_train, self.y_test,
                self.feature_names, self.X_train_bal, self.y_train_bal)

    def get_full(self):
        X = self.df_features[self.feature_names].values
        y = self.df_features["resistance"].values
        return X, y, self.feature_names, self.df_features

    def get_chi2_ranking(self) -> pd.DataFrame:
        X, y, names, _ = self.get_full()
        sel = SelectKBest(chi2, k="all")
        sel.fit(np.clip(X, 0, 1), y)
        return (pd.DataFrame({"Gene": names, "Chi2_Score": sel.scores_})
                .sort_values("Chi2_Score", ascending=False).reset_index(drop=True))

    def get_raw_preview(self, n: int = 50):
        amr = self.df_raw_amr.head(n) if self.df_raw_amr is not None else pd.DataFrame()
        sp  = self.df_raw_sp.head(n)  if self.df_raw_sp  is not None else pd.DataFrame()
        return amr, sp

    @property
    def data_source(self): return self._data_source
    @property
    def n_samples(self): return len(self.df_features) if self.df_features is not None else 0
    @property
    def warnings(self): return self._warnings
    @property
    def class_balance(self):
        if self.df_features is None: return {}
        y = self.df_features["resistance"].values
        return {"resistant":int((y==1).sum()), "susceptible":int((y==0).sum()),
                "resistant_pct":float((y==1).mean()*100)}

    # ── Real BV-BRC Loading ───────────────────────────────────────────────

    def _load_real(self, amr_path: Path, sp_path: Path):
        self._data_source = "BV-BRC (Real Data)"

        # ── 1. Read AMR CSV ───────────────────────────────────────────────
        print(f"\n  [1/5] Reading {amr_path.name} ...")
        amr_raw = pd.read_csv(amr_path, low_memory=False)
        print(f"        Raw shape: {amr_raw.shape}")

        amr = _normalise_columns(amr_raw.copy(), "amr_phenotype.csv")
        _check_required(amr, ["genome_id","resistant_phenotype"], "amr_phenotype.csv")
        self.df_raw_amr = amr_raw.copy()

        # ── 2. Filter to Meropenem ────────────────────────────────────────
        if "antibiotic" in amr.columns:
            amr["antibiotic"] = amr["antibiotic"].astype(str).str.lower().str.strip()
            tgt = BioSettings.ANTIBIOTIC.lower()
            n_before = len(amr)
            amr = amr[amr["antibiotic"] == tgt].copy()
            print(f"        After '{tgt}' filter: {len(amr):,} / {n_before:,} rows")
            if len(amr) == 0:
                vals = amr_raw.get("Antibiotic", pd.Series()).dropna().unique()[:8]
                raise ValueError(
                    f"\nNo '{tgt}' rows found.\n"
                    f"Antibiotic values in file: {vals}\n"
                    f"BV-BRC uses lowercase 'meropenem'."
                )
        else:
            w = "No 'antibiotic' column — treating all rows as Meropenem."
            print(f"        NOTE: {w}")
            self._warnings.append(w)

        # ── 3. Encode resistance (phenotype label OR MIC fallback) ────────
        print(f"\n  [2/5] Encoding resistance labels ...")
        amr["resistant_phenotype"] = amr["resistant_phenotype"].astype(str).str.strip()

        RES = {"Resistant","R","resistant","r","1","RESISTANT"}
        SUS = {"Susceptible","S","susceptible","s","0","Intermediate","I",
               "SUSCEPTIBLE","INTERMEDIATE","intermediate"}

        n_pheno = n_mic = n_drop = 0

        def _encode_row(row) -> int | None:
            nonlocal n_pheno, n_mic, n_drop
            pheno = str(row.get("resistant_phenotype","")).strip()

            # Try phenotype label first
            if pheno in RES: n_pheno += 1; return 1
            if pheno in SUS: n_pheno += 1; return 0

            # Fallback: MIC + CLSI breakpoints
            mic_raw = row.get("measurement_value", None)
            if mic_raw is None or (isinstance(mic_raw, float) and np.isnan(mic_raw)):
                mic_raw = row.get("measurement", None)

            sign = str(row.get("measurement_sign","")).strip()
            mic  = _parse_mic(mic_raw)

            if mic is not None:
                # If ">" sign and value already ≥ breakpoint → definitely Resistant
                if sign in (">",">=","≥") and mic >= MIC_RESISTANT_GE:
                    n_mic += 1; return 1
                if sign in ("<","<=","≤") and mic <= MIC_SUSCEPTIBLE_LE:
                    n_mic += 1; return 0
                if mic >= MIC_RESISTANT_GE:
                    n_mic += 1; return 1
                if mic <= MIC_SUSCEPTIBLE_LE:
                    n_mic += 1; return 0
                # Intermediate (2 < MIC < 8) → Susceptible
                n_mic += 1; return 0

            n_drop += 1
            return None

        amr["resistance"] = amr.apply(_encode_row, axis=1)
        self.label_stats  = {"from_phenotype": n_pheno, "from_mic": n_mic, "dropped": n_drop}

        print(f"        From phenotype label : {n_pheno:,}")
        print(f"        From MIC breakpoints : {n_mic:,}  "
              f"(Meropenem CLSI: R>={MIC_RESISTANT_GE} mg/L, S<={MIC_SUSCEPTIBLE_LE} mg/L)")
        print(f"        No label possible    : {n_drop:,} (dropped)")

        if n_pheno + n_mic == 0:
            raise ValueError(
                "No rows could be labelled -- check that amr_phenotype.csv has either\n"
                "'Resistant Phenotype' labels or numeric 'Measurement Value' (MIC) data."
            )

        amr = amr.dropna(subset=["resistance"])
        amr["resistance"] = amr["resistance"].astype(int)

        # One row per genome (most-resistant wins on duplicates)
        amr = amr.sort_values("resistance", ascending=False)
        keep = ["genome_id","resistance"]
        if "genome_name" in amr.columns: keep.insert(1,"genome_name")
        amr_labels = amr[keep].drop_duplicates("genome_id", keep="first").copy()
        if "genome_name" not in amr_labels.columns:
            amr_labels["genome_name"] = "Unknown"

        print(f"\n  [3/5] Unique labelled genomes: {len(amr_labels):,}")
        print(f"        Resistant  : {(amr_labels.resistance==1).sum():,}")
        print(f"        Susceptible: {(amr_labels.resistance==0).sum():,}")

        # ── 4. Read specialty genes CSV ────────────────────────────────────
        print(f"\n  [4/5] Reading {sp_path.name} ...")
        sp_raw = pd.read_csv(sp_path, low_memory=False)
        print(f"        Raw shape: {sp_raw.shape}")

        sp = _normalise_columns(sp_raw.copy(), "sp_genes.csv")
        _check_required(sp, ["genome_id","gene"], "sp_genes.csv")
        self.df_raw_sp = sp_raw.copy()

        # Property filter (broad to avoid losing rows)
        if "property" in sp.columns:
            sp["property"] = sp["property"].astype(str).str.strip()
            amr_props = {"AMR","Drug Target","Antibiotic Resistance","Virulence",
                         "drug target","amr","Transporter","Efflux Pump",
                         "Resistance Gene","Antibiotic Target"}
            filtered = sp[sp["property"].isin(amr_props)]
            if len(filtered) < 10:
                w = (f"Property filter matched only {len(filtered)} rows — using all {len(sp):,}. "
                     f"Properties seen: {sp['property'].value_counts().head(4).to_dict()}")
                print(f"        NOTE: {w}")
                self._warnings.append(w)
            else:
                sp = filtered.copy()
                print(f"        After property filter: {len(sp):,} rows")

        sp["gene"]      = sp["gene"].astype(str).str.strip().str.replace(r"\s+","_",regex=True)
        sp["genome_id"] = sp["genome_id"].astype(str).str.strip()
        amr_labels["genome_id"] = amr_labels["genome_id"].astype(str).str.strip()

        overlap = set(sp["genome_id"]) & set(amr_labels["genome_id"])
        if len(overlap) == 0:
            raise ValueError(
                f"\nNO GENOME ID OVERLAP between the two CSV files!\n"
                f"AMR sample IDs: {amr_labels['genome_id'].head(3).tolist()}\n"
                f"SP  sample IDs: {sp['genome_id'].head(3).tolist()}\n"
                f"Both files must be downloaded for A. baumannii (Taxon 470) from BV-BRC.\n"
                f"Re-download sp_genes.csv to make sure it covers the same genomes."
            )

        sp = sp[sp["genome_id"].isin(overlap)].copy()
        print(f"        Genome overlap   : {len(overlap):,}")
        print(f"        Unique AMR genes : {sp['gene'].nunique():,}")

        if len(overlap) < 50:
            w = (f"Only {len(overlap)} overlapping genomes (sp_genes 1,000-row limit). "
                 f"Consider downloading a larger sp_genes.csv from BV-BRC for better coverage.")
            print(f"        NOTE: {w}")
            self._warnings.append(w)

        # ── 5. Build gene presence/absence matrix ─────────────────────────
        print(f"\n  [5/5] Building presence/absence matrix ...")
        sp["present"] = 1
        pivot = (sp.groupby(["genome_id","gene"])["present"]
                   .max().unstack(fill_value=0).reset_index())

        merged = amr_labels.merge(pivot, on="genome_id", how="inner")
        gene_cols = [c for c in merged.columns
                     if c not in ("genome_id","genome_name","resistance")]
        print(f"        Before filter: {merged.shape[0]} samples x {len(gene_cols)} genes")

        # Adaptive threshold — automatically relaxes for small datasets
        prev = merged[gene_cols].mean()
        keep_genes = []
        used_thr   = PrepSettings.MIN_GENE_PREVALENCE
        for thr in [0.02, 0.01, 0.005, 0.001]:
            keep_genes = prev[prev >= thr].index.tolist()
            if len(keep_genes) >= 3:
                used_thr = thr
                if thr < PrepSettings.MIN_GENE_PREVALENCE:
                    w = (f"Gene prevalence threshold relaxed "
                         f"{PrepSettings.MIN_GENE_PREVALENCE*100:.0f}%->{thr*100:.1f}% "
                         f"to keep enough features.")
                    print(f"        NOTE: {w}")
                    self._warnings.append(w)
                break
        if not keep_genes:
            keep_genes = gene_cols  # last resort

        print(f"        After filter : {merged.shape[0]} samples x {len(keep_genes)} genes "
              f"(>={used_thr*100:.1f}% prevalence)")

        if merged.shape[0] < 10:
            raise ValueError(
                f"Feature matrix has only {merged.shape[0]} rows -- too small.\n"
                "Download more sp_genes data covering A. baumannii genomes."
            )

        self.feature_names = keep_genes
        self.df_features   = merged[["genome_id","genome_name","resistance"]+keep_genes].reset_index(drop=True)

        print(f"\n  [OK] Real data ready!")
        bal = self.class_balance
        print(f"     Samples    : {self.n_samples:,}")
        print(f"     Features   : {len(self.feature_names)} genes")
        print(f"     Resistant  : {bal['resistant']:,} ({bal['resistant_pct']:.1f}%)")
        print(f"     Susceptible: {bal['susceptible']:,} ({100-bal['resistant_pct']:.1f}%)")

    # ── Synthetic Fallback ────────────────────────────────────────────────

    def _load_synthetic(self, n: int = 2500):
        self._data_source = "Synthetic (BV-BRC calibrated)"
        np.random.seed(PrepSettings.RANDOM_STATE)
        genes = [
            "blaOXA-23","blaOXA-51","blaOXA-58","blaOXA-72","blaOXA-40",
            "blaNDM-1","blaVIM-1","blaIMP-1","blaTEM-1","blaSHV-1","blaCTX-M","blaADC-30",
            "adeB","adeC","adeR","adeS","adeJ","adeK","abeM",
            "armA","aacC1","aacA4","aphA6","aac6-Ib",
            "tet39","tetA","tetB","sul1","sul2",
            "gyrA_83","gyrA_87","parC_80","cmlA","catB8",
            "lpxA_mut","lpxC_mut","lpxD_mut","pmrB",
            "intI1","Tn2006","ISAba1","ISAba2","ISAba3","ISEc29",
            "carO_intact","oprD_intact","kmer_resistance_score","snp_density",
        ]
        bp = {
            "blaOXA-23":0.72,"blaOXA-51":0.88,"blaOXA-58":0.28,"blaOXA-72":0.31,
            "blaOXA-40":0.18,"blaNDM-1":0.41,"blaVIM-1":0.19,"blaIMP-1":0.14,
            "blaTEM-1":0.58,"blaSHV-1":0.22,"blaCTX-M":0.33,"blaADC-30":0.45,
            "adeB":0.68,"adeC":0.61,"adeR":0.55,"adeS":0.52,"adeJ":0.43,
            "adeK":0.38,"abeM":0.29,"armA":0.44,"aacC1":0.37,"aacA4":0.31,
            "aphA6":0.28,"aac6-Ib":0.33,"tet39":0.48,"tetA":0.25,"tetB":0.19,
            "sul1":0.52,"sul2":0.29,"gyrA_83":0.38,"gyrA_87":0.25,"parC_80":0.33,
            "cmlA":0.21,"catB8":0.27,"lpxA_mut":0.22,"lpxC_mut":0.18,
            "lpxD_mut":0.15,"pmrB":0.31,"intI1":0.61,"Tn2006":0.44,
            "ISAba1":0.67,"ISAba2":0.38,"ISAba3":0.29,"ISEc29":0.22,
            "carO_intact":0.35,"oprD_intact":0.41,
            "kmer_resistance_score":0.5,"snp_density":0.5,
        }
        y = np.random.choice([0,1], n, p=[0.40,0.60])
        X = np.zeros((n, len(genes)))
        for i, g in enumerate(genes):
            p = bp[g]
            if g in ("kmer_resistance_score","snp_density"):
                X[:,i] = np.where(y==1,np.random.normal(0.72,0.14,n),
                                   np.random.normal(0.30,0.14,n)).clip(0,1)
            elif g in ("carO_intact","oprD_intact"):
                X[:,i] = np.where(y==1,np.random.binomial(1,p*0.25,n),
                                   np.random.binomial(1,min(p*1.5,0.95),n))
            elif g in ("blaOXA-23","blaNDM-1","adeB","ISAba1","armA","blaOXA-51"):
                X[:,i] = np.where(y==1,np.random.binomial(1,min(p*1.45,0.97),n),
                                   np.random.binomial(1,p*0.18,n))
            else:
                X[:,i] = np.where(y==1,np.random.binomial(1,min(p*1.25,0.90),n),
                                   np.random.binomial(1,p*0.55,n))
        X = X.clip(0,1)
        df = pd.DataFrame(X, columns=genes)
        df["resistance"]  = y
        df["genome_id"]   = [f"SYNTH_{i+1:05d}" for i in range(n)]
        df["genome_name"] = [f"A. baumannii SYN{i+1:05d}" for i in range(n)]
        self.df_raw_amr = df[["genome_id","genome_name","resistance"]].assign(
            antibiotic="meropenem",
            resistant_phenotype=df["resistance"].map({1:"Resistant",0:"Susceptible"}))
        self.df_raw_sp = (df[["genome_id"]+genes[:12]]
                          .melt(id_vars="genome_id",var_name="gene",value_name="v")
                          .query("v==1").drop(columns="v").assign(property="AMR"))
        self.feature_names = genes
        self.df_features = df[["genome_id","genome_name","resistance"]+genes].copy()
        self.label_stats = {"from_phenotype": n, "from_mic": 0, "dropped": 0}
        print(f"  Synthetic: {n:,} samples x {len(genes)} features")

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _clean(self):
        df = self.df_features
        b = len(df)
        df = df.dropna(subset=["resistance"]).drop_duplicates("genome_id")
        df[self.feature_names] = df[self.feature_names].fillna(0).clip(0,1)
        a = len(df)
        if b != a: print(f"  Cleaning: removed {b-a} rows -> {a:,} remain")
        self.df_features = df.reset_index(drop=True)

    def _split(self):
        X = self.df_features[self.feature_names].values
        y = self.df_features["resistance"].values
        if (y==0).sum() < 2 or (y==1).sum() < 2:
            raise ValueError(
                f"Too few samples per class: {(y==0).sum()} Susceptible, "
                f"{(y==1).sum()} Resistant. Need >=2 of each."
            )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=PrepSettings.TEST_SIZE,
            random_state=PrepSettings.RANDOM_STATE, stratify=y)
        print(f"  Train: {len(self.y_train):,}  |  Test: {len(self.y_test):,}")

    def _apply_smote(self):
        sm = SMOTE(random_state=PrepSettings.RANDOM_STATE)
        self.X_train_bal, self.y_train_bal = sm.fit_resample(self.X_train, self.y_train)
        print(f"  SMOTE -> {len(self.y_train_bal):,} balanced samples "
              f"({(self.y_train_bal==1).sum():,}R / {(self.y_train_bal==0).sum():,}S)")

    def _save_processed(self):
        DataPaths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.df_features.to_csv(DataPaths.FEATURE_MATRIX, index=False)
        pd.DataFrame({"gene": self.feature_names}).to_csv(DataPaths.GENE_LIST, index=False)
        print(f"  Saved -> {DataPaths.PROCESSED_DIR}")