import argparse
import json
import time
import warnings
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
 
# =====================================================================
# 1. KONFIGURASI
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "training"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "classifier"

 
CONFIG_ORDER = ("E0", "E1", "E2", "E3")
SEVERITY_ORDER = ["Low", "Medium", "High"]
PRIMARY_METRIC = "macro_f1"
 
# Kolom yang tidak boleh ada di fitur (sama dengan FORBIDDEN_FEATURE_COLUMNS
# di preprocessing.py; diulang di sini sebagai pagar kedua di sisi pemakai).
FORBIDDEN_FEATURE_COLUMNS = (
    "severity", "severity_score", "fcw", "sct", "fault_category", "service_tier",
    "service", "fault_type", "scenario", "run_number", "inject_time",
)
 
# Hanya dipakai untuk membentuk TARGET probe dari metadata (bukan fitur).
FAULT_CATEGORY = {
    "cpu": "resource", "mem": "resource", "disk": "resource",
    "delay": "network", "loss": "network", "socket": "resource",
}
 
# RandomForest sklearn >= 1.4 menangani NaN secara native.
SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
NATIVE_NAN_SUPPORT = SKLEARN_VERSION >= (1, 4)

# Tujuh metrik yang tersedia seragam di kelima service  yang diijeksikan fault
# Urutan ini menentukan urutan kolom pada matriks relatif, jadi jangan diubah
# setelah hasil disimpan (urutan kolom mempengaruhi pemecahan simpul pada randomforest dengan random_state tetap)
RELATIVE_METRICS = (
    "cpu", "mem", "diskio", "socket", "latency-50", "latency-90", "workload"
) 

# Lima statistik yang sama dengan FEATURE_STATISTICS di preprocessing.py
# Kelima nya dibutuhkan: memakai delta_mean saja menurunkan Macro-f1 sebesar 0,263
RELATIVE_STATISTICS = ("baseline_mean", "incident_max", "delta_mean", "zscore_max", "zscore_mean")

# Kategori kalimat log, sama dengan LOG_TEMPLATE_CATEGORIES di preprocessing.py.
# currencyservice tidak punya kategori "error"; kolomnya NaN dan itu ditangani
# RandomForest secara native.
RELATIVE_LOG_CATEGORIES = ("error", "lifecycle", "routine")

# Dua awalan kolom trace. Nama operasi berbeda antar service (checkoutservice
# hanya PlaceOrder, productcatalogservice punya GetProduct dan ListProducts),
# jadi tidak ada kosakata operasi bersama untuk diselaraskan. Diringkas dengan
# MAKSIMUM atas seluruh operasi milik service terdampak: fault mengenai
# service-nya, sehingga operasi terparah yang menjadi sinyal.
RELATIVE_TRACE_PREFIXES = ("trace_latency", "trace_error")

# Kolom yang memang global (bukan milik satu service) dipakai apa adanya,
# tanpa penyelarasan.
RELATIVE_GLOBAL_LOG_COLUMNS = ("log__all__new_template_count",)
RELATIVE_GLOBAL_SVC_PREFIX = "svc__"

# Modalitas yang diselaraskan untuk tiap konfigurasi, mengikuti pertingkatan
# E0 subset E1 subset E2 subset E3 pada file fitur absolut.
RELATIVE_CONFIG_MODALITIES = {
    "E0": ("metric",),
    "E1": ("metric", "log"),
    "E2": ("metric", "log", "trace"),
    "E3": ("metric", "log", "trace", "svc"),
}
 
# =====================================================================
# 2. MEMUAT DATA
# =====================================================================
def load_common_tables(data_dir):
    """Membaca label dan metadata, memastikan keduanya sejajar."""
    labels = pd.read_csv(data_dir / "derived_severity_labels.csv", usecols=["case_id", "severity"])
    meta = pd.read_csv(
        data_dir / "case_metadata.csv",
        dtype={"run_number": "string"},
    )
 
    if labels["case_id"].duplicated().any() or meta["case_id"].duplicated().any():
        raise RuntimeError("case_id duplikat pada label atau metadata")
    if not labels["case_id"].equals(meta["case_id"]):
        raise RuntimeError("Urutan case_id label dan metadata tidak sama")
    if not set(labels["severity"]).issubset(SEVERITY_ORDER):
        raise RuntimeError(f"Kelas severity tidak dikenal: {set(labels['severity'])}")
 
    return labels, meta
 
 
def load_features(data_dir, config, case_ids):
    """Membaca features_<config>.csv, memvalidasi, mengembalikan (matriks, nama kolom)."""
    frame = pd.read_csv(data_dir / f"features_{config}.csv")
 
    if frame.columns[0] != "case_id":
        raise RuntimeError(f"{config}: kolom pertama harus case_id")
    if not frame["case_id"].equals(case_ids):
        raise RuntimeError(f"{config}: urutan case_id berbeda dari label")
 
    feature_columns = [column for column in frame.columns if column != "case_id"]
 
    forbidden = [column for column in FORBIDDEN_FEATURE_COLUMNS if column in feature_columns]
    if forbidden:
        raise RuntimeError(f"{config}: kolom terlarang di fitur: {forbidden}")
 
    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise RuntimeError(f"{config}: kolom non-numerik: {non_numeric[:5]}")
 
    values = frame[feature_columns].to_numpy(dtype="float64")
    if np.isinf(values).any():
        raise RuntimeError(f"{config}: ada nilai tak hingga")
 
    return values, feature_columns


def build_relative_specs(frame, services, modalities):
    """
    Menyusun resep tiap kolom relatif: (nama, {service: [kolom sumber]}, cara).

    "cara" bernilai "one" bila satu kolom sumber per service, atau "max" bila
    beberapa kolom diringkas dengan maksimum (kasus trace, karena nama operasi
    berbeda antar service).

    Kolom global (log__all__*, svc__*) tidak masuk ke sini; ia ditangani
    terpisah sebagai kolom apa adanya.
    """
    specs = []

    if "metric" in modalities:
        for metric in RELATIVE_METRICS:
            for statistic in RELATIVE_STATISTICS:
                specs.append((
                    f"self__{metric}__{statistic}",
                    {s: [f"metric__{s}_{metric}__{statistic}"] for s in services},
                    "one",
                ))

    if "log" in modalities:
        for category in RELATIVE_LOG_CATEGORIES:
            for statistic in RELATIVE_STATISTICS:
                specs.append((
                    f"self_log__{category}__{statistic}",
                    {s: [f"log__{s}__{category}__{statistic}"] for s in services},
                    "one",
                ))
        specs.append((
            "self_log__new_template_count",
            {s: [f"log__{s}__new_template_count"] for s in services},
            "one",
        ))

    if "trace" in modalities:
        for prefix in RELATIVE_TRACE_PREFIXES:
            for statistic in RELATIVE_STATISTICS:
                mapping = {
                    s: [c for c in frame.columns
                        if c.startswith(f"{prefix}__{s}_") and c.endswith(f"__{statistic}")]
                    for s in services
                }
                specs.append((f"self_{prefix}__max__{statistic}", mapping, "max"))

    return specs


def collect_relative_global_columns(frame, modalities):
    """Kolom yang memang bukan milik satu service, dipakai apa adanya."""
    columns = []
    if "log" in modalities:
        columns += [c for c in RELATIVE_GLOBAL_LOG_COLUMNS if c in frame.columns]
    if "svc" in modalities:
        columns += [c for c in frame.columns if c.startswith(RELATIVE_GLOBAL_SVC_PREFIX)]
    return columns


def build_relative_lookup(frame, services, modalities=("metric",)):
    """
    Menyusun satu matriks fitur relatif untuk SETIAP kandidat service, dihitung
    sekali di awal.

    Keluaran:
        lookup : array (n_service, n_run, n_fitur)
                 lookup[i, j, k] = nilai fitur k milik service ke-i, pada run ke-j
        names  : daftar nama kolom

    Kolom global diletakkan di belakang: nilainya sama untuk semua kandidat
    service, sehingga penyelarasan tidak mengubahnya.

    Kolom yang tidak tersedia pada satu pun kandidat dibuang, dan namanya dicetak.

    Alasan dihitung sekali di awal: penyusunan ulang matriks ini di dalam loop
    fold memakai akses baris pandas memakan waktu sekitar sepuluh kali lebih
    lama daripada pengambilan lewat indeks numpy.
    """
    specs = build_relative_specs(frame, services, modalities)
    global_columns = collect_relative_global_columns(frame, modalities)
    names = [name for name, _, _ in specs] + global_columns

    lookup = np.full((len(services), len(frame), len(names)), np.nan)
    for service_index, service in enumerate(services):
        for spec_index, (_, mapping, how) in enumerate(specs):
            columns = [c for c in mapping.get(service, []) if c in frame.columns]
            if not columns:
                continue
            block = frame[columns].apply(pd.to_numeric, errors="coerce")
            lookup[service_index, :, spec_index] = (
                block.max(axis=1).to_numpy() if how == "max" else block.iloc[:, 0].to_numpy()
            )
        for offset, column in enumerate(global_columns):
            lookup[service_index, :, len(specs) + offset] = (
                pd.to_numeric(frame[column], errors="coerce").to_numpy()
            )

    kept = ~np.all(np.isnan(lookup), axis=(0, 1))
    dropped = [name for name, keep in zip(names, kept) if not keep]
    if dropped:
        print(f"[REL] {len(dropped)} kolom relatif tidak tersedia, dibuang: {dropped[:5]}")

    return lookup[:, :, kept], [name for name, keep in zip(names, kept) if keep]


def take_relative(lookup, service_index):
    """
    Mengambil, untuk setiap run, baris dari matriks milik service yang ditunjuk
    untuk run itu. Hasilnya array (n_run, n_fitur).

    service_index[j] = indeks service yang dipakai untuk menyelaraskan run ke-j.
    """
    return lookup[service_index, np.arange(lookup.shape[1])]

# =====================================================================
# 3. PEMBAGIAN FOLD (dibuat sekali, disimpan, dipakai ulang)
# =====================================================================
def build_or_load_splits(y, groups, case_ids, n_splits, n_repeats, seed, results_dir):
    """
    Mengembalikan daftar (repeat, fold, indeks_train, indeks_test).
    Berkas fold dinamai dengan parameternya; jika sudah ada, dipakai ulang
    sehingga eksekusi berikutnya memakai pembagian yang persis sama.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    folds_path = results_dir / f"folds_k{n_splits}_r{n_repeats}_seed{seed}.csv"
 
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        if set(folds["case_id"]) != set(case_ids):
            raise RuntimeError(f"{folds_path.name} tidak cocok dengan daftar case_id saat ini")
        index_of = {case_id: i for i, case_id in enumerate(case_ids)}
        splits = []
        for (repeat, fold), part in folds.groupby(["repeat", "fold"], sort=True):
            test_idx = np.array(sorted(index_of[c] for c in part["case_id"]))
            train_idx = np.setdiff1d(np.arange(len(case_ids)), test_idx)
            splits.append((int(repeat), int(fold), train_idx, test_idx))
        print(f"[OK] Fold dibaca dari {folds_path.name}")
        return splits
 
    splits, records = [], []
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            if set(y[train_idx]) != set(SEVERITY_ORDER):
                raise RuntimeError(f"Repeat {repeat} fold {fold}: train tidak memuat semua kelas")
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError(f"Repeat {repeat} fold {fold}: kelompok bocor antar train/test")
            splits.append((repeat, fold, train_idx, test_idx))
            records.extend(
                {"repeat": repeat, "fold": fold, "case_id": case_ids[i]} for i in test_idx
            )
 
    pd.DataFrame(records).to_csv(folds_path, index=False)
    print(f"[OK] Fold dibuat dan disimpan: {folds_path.name}")
    return splits
 
 
# =====================================================================
# 4. MODEL
# =====================================================================
def make_random_forest(n_estimators, seed, n_jobs):
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=n_jobs,
    )
    if NATIVE_NAN_SUPPORT:
        return forest
    # Cadangan untuk sklearn lama: median dihitung dari data train saja.
    return make_pipeline(SimpleImputer(strategy="median"), forest)
 
 
def make_dummy(strategy, seed):
    return DummyClassifier(strategy=strategy, random_state=seed)


def wants_fold_models(args, config):
    """
    True bila model tiap fold perlu disimpan untuk konfigurasi ini.
    --save-fold-models tanpa nilai  -> semua konfigurasi yang dijalankan
    --save-fold-models E0rel        -> hanya yang disebut
    tidak dipakai sama sekali       -> tidak ada yang disimpan
    """
    if args.save_fold_models is None:
        return False
    return not args.save_fold_models or config in args.save_fold_models


def make_service_locator(n_estimator, seed, n_jobs):
    """
    Model tahap pertama: menebak service mana yang terdampak, dari fitur absolut.

    Tanpa class_weight, karena jumlah run per service sudah hampir seimbang
    (18 run per service, kecuali checkoutservice 17 karena satu run dikecualikan
    pada Fase 7).
    """
    forest = RandomForestClassifier(
        n_estimators=n_estimator,
        max_features="sqrt",
        random_state=seed,
        n_jobs=n_jobs,
    )
    if NATIVE_NAN_SUPPORT:
        return forest
    return make_pipeline(SimpleImputer(strategy="median"), forest)
 
 
# =====================================================================
# 5. EVALUASI
# =====================================================================
def _align_proba(model, proba, classes):
    """Menyusun probabilitas mengikuti urutan `classes`, mengisi 0 untuk kelas yang tak terlihat."""
    fitted = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    aligned = np.zeros((proba.shape[0], len(classes)))
    for column, label in enumerate(fitted):
        aligned[:, classes.index(label)] = proba[:, column]
    return aligned
 
 
def cross_validate(model_factory, X, y, splits, n_repeats, classes, collect = None):
    """
    Menjalankan seluruh fold. Setiap run mendapat tepat satu prediksi per repeat
    (karena tiap repeat adalah partisi lengkap). Mengembalikan:
        pred  : array (n_repeats, n_run) berisi label
        proba : array (n_repeats, n_run, n_kelas)
    """
    n = len(y)
    pred = np.empty((n_repeats, n), dtype=object)
    proba = np.zeros((n_repeats, n, len(classes)))
 
    for repeat, fold, train_idx, test_idx in splits:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        if collect is not None:
            collect.append({"repeat": repeat, "fold": fold, "model": model, "test_idx": test_idx})
        pred[repeat, test_idx] = model.predict(X[test_idx])
        proba[repeat, test_idx] = _align_proba(model, model.predict_proba(X[test_idx]), classes)
 
    if (pred == None).any():  # noqa: E711 - array object, perbandingan elemen
        raise RuntimeError("Ada run tanpa prediksi out-of-fold")
    return pred, proba

def cross_validate_relative(X_absolute, lookup, services, service_true, y, splits, n_repeats, classes, locator_factory, model_factory, collect=None):
    """
    Validasi silang dua tahap, dijalankan di dalam setiap fold:

      Tahap 1  Latih penebak service dari fitur ABSOLUT, hanya pada data latih
               fold ini, lalu prediksi service untuk data uji.
      Tahap 2  Selaraskan fitur ke service tersebut, lalu latih model severity.

    Penyelarasan data latih memakai service sebenarnya; penyelarasan data uji
    memakai service hasil prediksi. Jadi sisi uji tidak pernah menerima
    identitas service yang sebenarnya, dan tidak ada informasi label yang bocor
    ke sana.

    Mengembalikan (pred, proba, akurasi_penebak_service).
    """
    n = len(y)
    index_of = {service: i for i, service in enumerate(services)}
    true_index = np.array([index_of[service] for service in service_true])

    pred = np.empty((n_repeats, n), dtype=object)
    proba = np.zeros((n_repeats, n, len(classes)))
    locator_hits = []

    for repeat, fold, train_idx, test_idx in splits:
        # ---- Tahap 1: penebak service ----
        locator = locator_factory()
        locator.fit(X_absolute[train_idx], service_true[train_idx])
        predicted_service = locator.predict(X_absolute[test_idx])
        locator_hits.append(float((predicted_service == service_true[test_idx]).mean()))

        # ---- Tahap 2: model severity ----
        assignment = true_index.copy()
        assignment[test_idx] = [index_of[service] for service in predicted_service]
        X_relative = take_relative(lookup, assignment)

        model = model_factory()
        model.fit(X_relative[train_idx], y[train_idx])
        if collect is not None:
            collect.append({"repeat": repeat, "fold": fold, "locator": locator, "model": model, "test_idx": test_idx, "Xtest": X_relative[test_idx]})
        pred[repeat, test_idx] = model.predict(X_relative[test_idx])
        proba[repeat, test_idx] = _align_proba(model, model.predict_proba(X_relative[test_idx]), classes)

    if (pred == None).any(): # noqa: E711 - array object, perbandingan elemen
        raise RuntimeError("Ada run tanpa prediksi out-of-fold")

    return pred, proba, float(np.mean(locator_hits))
 
 
def score_predictions(y_true, y_pred, classes):
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }
    for label, value in zip(classes, recalls):
        row[f"recall_{label}"] = value
    return row
 
 
def metrics_by_repeat(name, y, pred, classes):
    return [
        {"model": name, "repeat": repeat, **score_predictions(y, pred[repeat], classes)}
        for repeat in range(pred.shape[0])
    ]
 
 
def summarize_metrics(by_repeat):
    metric_columns = [c for c in by_repeat.columns if c not in {"model", "repeat"}]
    grouped = by_repeat.groupby("model", sort=False)[metric_columns]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    return summary.reset_index()
 
 
def per_scenario_accuracy(y, pred, groups):
    correct = (pred == y[None, :]).astype(float).mean(axis=0)  # rata-rata antar-repeat per run
    return pd.Series(correct).groupby(groups).mean()
 
 
def aggregate_confusion(y, pred, classes):
    total = np.zeros((len(classes), len(classes)), dtype=int)
    for repeat in range(pred.shape[0]):
        total += confusion_matrix(y, pred[repeat], labels=classes)
    return pd.DataFrame(total, index=[f"true_{c}" for c in classes],
                        columns=[f"pred_{c}" for c in classes])
 
 
def paired_scenario_tests(scenario_acc):
    """Wilcoxon berpasangan pada akurasi per-skenario (unit = 30 skenario, bukan 89 run)."""
    pairs = [("E0", "E1"), ("E1", "E2"), ("E2", "E3"), ("E0", "E3")]
    # Absolut versus relatif, untuk konfigurasi yang keduanya dijalankan.
    pairs += [(config, f"{config}rel") for config in CONFIG_ORDER]
    # Pertingkatan di sisi relatif, sejajar dengan uji E0-E3 di sisi absolut.
    pairs += [("E0rel", "E1rel"), ("E1rel", "E2rel"), ("E2rel", "E3rel"), ("E0rel", "E3rel")]
    rows = []
    for a, b in pairs:
        if a not in scenario_acc or b not in scenario_acc:
            continue
        diff = scenario_acc[b] - scenario_acc[a]
        row = {"comparison": f"{b} vs {a}", "mean_diff_accuracy": diff.mean(),
               "scenarios_better": int((diff > 0).sum()), "scenarios_worse": int((diff < 0).sum()),
               "scenarios_tied": int((diff == 0).sum()), "wilcoxon_p": np.nan}
        if (diff != 0).any():
            try:
                row["wilcoxon_p"] = float(wilcoxon(diff, zero_method="wilcox").pvalue)
            except ValueError:
                pass
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def save_oof(results_dir, config, case_ids, y, pred, proba, classes):
    parts = []
    for repeat in range(pred.shape[0]):
        frame = pd.DataFrame({"case_id": case_ids, "repeat": repeat,
                              "y_true": y, "y_pred": pred[repeat]})
        for column, label in enumerate(classes):
            frame[f"proba_{label}"] = proba[repeat, :, column]
        parts.append(frame)
    pd.concat(parts, ignore_index=True).to_csv(results_dir / f"oof_predictions_{config}.csv", index=False)
 
 
# =====================================================================
# 6. ORKESTRATOR
# =====================================================================
def run(args):
    started = time.time()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
 
    print("\n[Classifier] Memuat berkas training...")
    labels, meta = load_common_tables(data_dir)
    case_ids = labels["case_id"]
    y = labels["severity"].to_numpy(dtype=object)
    groups = meta["scenario"].to_numpy(dtype=object)
 
    print(f"[OK] {len(labels)} run, {len(set(groups))} kelompok skenario")
    print("[OK] Sebaran kelas:", labels["severity"].value_counts().reindex(SEVERITY_ORDER).to_dict())
    if not NATIVE_NAN_SUPPORT:
        print("[WARNING!!] sklearn < 1.4: NaN diisi median (dihitung dari data train per fold)")
 
    splits = build_or_load_splits(
        y, groups, case_ids.tolist(), args.splits, args.repeats, args.seed, results_dir
    )
 
    by_repeat_rows = []
    scenario_acc = {}
    confusion = {}
 
    # ---- Baseline (tanpa fitur berguna; memberi lantai untuk Macro-F1) ----
    dummy_X = np.zeros((len(y), 1))
    for name, strategy in (("baseline_majority", "most_frequent"), ("baseline_stratified", "stratified")):
        pred, _ = cross_validate(lambda s=strategy: make_dummy(s, args.seed),
                                 dummy_X, y, splits, args.repeats, SEVERITY_ORDER)
        by_repeat_rows += metrics_by_repeat(name, y, pred, SEVERITY_ORDER)
 
    # ---- Random Forest E0-E3 ----
    for config in args.configs:
        X, feature_names = load_features(data_dir, config, case_ids)
        print(f"\n[{config}] {X.shape[0]} baris x {X.shape[1]} fitur, sel kosong: {int(np.isnan(X).sum())}")
 
        fold_models = [] if wants_fold_models(args, config) else None
        pred, proba = cross_validate(
            lambda: make_random_forest(args.trees, args.seed, args.jobs),
            X, y, splits, args.repeats, SEVERITY_ORDER, collect=fold_models,
        )
        by_repeat_rows += metrics_by_repeat(f"RF_{config}", y, pred, SEVERITY_ORDER)
        scenario_acc[config] = per_scenario_accuracy(y, pred, groups)
        confusion[config] = aggregate_confusion(y, pred, SEVERITY_ORDER)
        confusion[config].to_csv(results_dir / f"confusion_{config}.csv")
        save_oof(results_dir, config, case_ids, y, pred, proba, SEVERITY_ORDER)
 
        if not args.no_save_models:
            # Model akhir (semua run) hanya untuk inferensi kasus BARU.
            # Untuk menjelaskan 89 run yang ada, pakai oof_predictions + model per-fold,
            # bukan model ini, agar penjelasan tidak in-sample.
            final_model = make_random_forest(args.trees, args.seed, args.jobs).fit(X, y)
            models_dir = results_dir / "models"
            models_dir.mkdir(exist_ok=True)
            joblib.dump({"model": final_model, "feature_names": feature_names, "classes": SEVERITY_ORDER},
                        models_dir / f"rf_{config}.joblib")

        if fold_models is not None:
            models_dir = results_dir / "models"
            models_dir.mkdir(exist_ok=True)
            joblib.dump({"folds": fold_models, "feature_names": feature_names,
                         "classes": SEVERITY_ORDER},
                        models_dir / f"folds_{config}.joblib")

        model_name = f"RF_{config}"
        last = pd.DataFrame(by_repeat_rows).query("model == @model_name")
        print(f"[{config}] Macro-F1 {last['macro_f1'].mean():.3f} ± {last['macro_f1'].std(ddof=1):.3f} | "
              f"Recall High {last['recall_High'].mean():.3f}")

    # ---- Random Forest dengan fitur relatif (penyelarasan ke service terdampak) ----
    if args.relative:
        services = sorted(set(meta["service"]))
        service_true = meta["service"].to_numpy(dtype=object)

        for source in args.relative_configs:
            frame = pd.read_csv(data_dir / f"features_{source}.csv")
            if not frame["case_id"].equals(case_ids):
                raise RuntimeError(f"features_{source}.csv: urutan case_id berbeda dari label")

            modalities = RELATIVE_CONFIG_MODALITIES[source]
            lookup, relative_names = build_relative_lookup(frame, services, modalities)
            X_absolute, _ = load_features(data_dir, source, case_ids)

            config = f"{source}rel"
            print(f"\n[{config}] {lookup.shape[1]} baris x {lookup.shape[2]} fitur relatif "
                  f"(disarikan dari {X_absolute.shape[1]} fitur absolut, "
                  f"{len(services)} kandidat service, modalitas {'+'.join(modalities)})")

            fold_models = [] if wants_fold_models(args, config) else None
            pred, proba, locator_accuracy = cross_validate_relative(
                X_absolute, lookup, services, service_true, y, splits, args.repeats,
                SEVERITY_ORDER,
                lambda: make_service_locator(args.trees, args.seed, args.jobs),
                lambda: make_random_forest(args.trees, args.seed, args.jobs),
                collect=fold_models,
            )

            by_repeat_rows += metrics_by_repeat(f"RF_{config}", y, pred, SEVERITY_ORDER)
            scenario_acc[config] = per_scenario_accuracy(y, pred, groups)
            confusion[config] = aggregate_confusion(y, pred, SEVERITY_ORDER)
            confusion[config].to_csv(results_dir / f"confusion_{config}.csv")
            save_oof(results_dir, config, case_ids, y, pred, proba, SEVERITY_ORDER)
            pd.Series(relative_names, name="feature_name").to_csv(
                results_dir / f"feature_names_{config}.csv", index=False
            )

            models_dir = results_dir / "models"
            models_dir.mkdir(exist_ok=True)

            if fold_models is not None:
                # Dipakai explainer.py: tiap kasus dijelaskan oleh model yang
                # TIDAK pernah melihat kasus itu saat latihan.
                joblib.dump(
                    {"folds": fold_models, "feature_names": relative_names,
                     "services": services, "classes": SEVERITY_ORDER},
                    models_dir / f"folds_{config}.joblib",
                )

            if not args.no_save_models:
                # Model akhir dua tahap, untuk kasus BARU. Bukan untuk menjelaskan
                # 89 run yang ada, karena dilatih pada seluruh data.
                index_of = {service: i for i, service in enumerate(services)}
                true_index = np.array([index_of[service] for service in service_true])
                locator_final = make_service_locator(args.trees, args.seed, args.jobs)
                locator_final.fit(X_absolute, service_true)
                severity_final = make_random_forest(args.trees, args.seed, args.jobs)
                severity_final.fit(take_relative(lookup, true_index), y)
                joblib.dump(
                    {"locator": locator_final, "model": severity_final,
                     "feature_names": relative_names, "services": services,
                     "classes": SEVERITY_ORDER},
                    models_dir / f"rf_{config}.joblib",
                )

            if args.probe_config:
                # Probe yang sama seperti versi absolut, tetapi lewat jalur relatif.
                # Hasilnya berdampingan di metrics_summary.csv untuk dibandingkan.
                target = meta["fault_type"].map(FAULT_CATEGORY).to_numpy(dtype=object)
                probe_classes = sorted(set(target))
                probe_pred, _, _ = cross_validate_relative(
                    X_absolute, lookup, services, service_true, target, splits,
                    args.repeats, probe_classes,
                    lambda: make_service_locator(args.trees, args.seed, args.jobs),
                    lambda: make_random_forest(args.trees, args.seed, args.jobs),
                )
                for repeat in range(probe_pred.shape[0]):
                    by_repeat_rows.append({
                        "model": f"probe_fault_category_{config}", "repeat": repeat,
                        "accuracy": accuracy_score(target, probe_pred[repeat]),
                        "balanced_accuracy": balanced_accuracy_score(target, probe_pred[repeat]),
                        "macro_f1": f1_score(target, probe_pred[repeat],
                                             average="macro", zero_division=0),
                    })

            model_name = f"RF_{config}"
            last = pd.DataFrame(by_repeat_rows).query("model == @model_name")
            print(f"[{config}] Penebak service: {locator_accuracy:.3f} "
                  f"(rata-rata {len(splits)} fold)")
            print(f"[{config}] Macro-F1 {last['macro_f1'].mean():.3f} ± "
                  f"{last['macro_f1'].std(ddof=1):.3f} | "
                  f"Recall High {last['recall_High'].mean():.3f}")

 
    # ---- Probe: seberapa jauh telemetri mengenali service & kategori fault ----
    if args.probe_config:
        X, _ = load_features(data_dir, args.probe_config, case_ids)
        probe_targets = {
            "probe_service": meta["service"].to_numpy(dtype=object),
            "probe_fault_category": meta["fault_type"].map(FAULT_CATEGORY).to_numpy(dtype=object),
        }
        for name, target in probe_targets.items():
            classes = sorted(set(target))
            pred, _ = cross_validate(
                lambda: make_random_forest(args.trees, args.seed, args.jobs),
                X, target, splits, args.repeats, classes,
            )
            for repeat in range(pred.shape[0]):
                by_repeat_rows.append({
                    "model": f"{name}_{args.probe_config}", "repeat": repeat,
                    "accuracy": accuracy_score(target, pred[repeat]),
                    "balanced_accuracy": balanced_accuracy_score(target, pred[repeat]),
                    "macro_f1": f1_score(target, pred[repeat], average="macro", zero_division=0),
                })
 
    # ---- Ringkasan & penyimpanan ----
    by_repeat = pd.DataFrame(by_repeat_rows)
    summary = summarize_metrics(by_repeat)
    tests = paired_scenario_tests(scenario_acc)
 
    by_repeat.to_csv(results_dir / "metrics_by_repeat.csv", index=False)
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)
    if scenario_acc:
        pd.DataFrame(scenario_acc).rename_axis("scenario").to_csv(results_dir / "per_scenario_accuracy.csv")
    tests.to_csv(results_dir / "paired_tests.csv", index=False)
 
    with (results_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "native_nan_support": NATIVE_NAN_SUPPORT,
            "n_runs": int(len(y)), "n_groups": int(len(set(groups))),
            "n_splits": args.splits, "n_repeats": args.repeats, "seed": args.seed,
            "n_trees": args.trees, "class_weight": "balanced_subsample",
            "primary_metric": PRIMARY_METRIC, "configs": list(args.configs),
            "relative_features": bool(args.relative),
            "relative_configs": list(args.relative_configs) if args.relative else None,
            "relative_metrics": list(RELATIVE_METRICS) if args.relative else None,
            "relative_statistics": list(RELATIVE_STATISTICS) if args.relative else None,
            "relative_log_categories": list(RELATIVE_LOG_CATEGORIES) if args.relative else None,
            "relative_trace_aggregation": "max_over_service_operations" if args.relative else None,
            "relative_modalities": (
                {c: list(RELATIVE_CONFIG_MODALITIES[c]) for c in args.relative_configs}
                if args.relative else None
            ),
        }, file, ensure_ascii=False, indent=2)
 
    pd.set_option("display.width", 200)
    show = ["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
            "accuracy_mean", "recall_High_mean"]
    print("\n========== RINGKASAN (rata-rata ± std antar-repeat) ==========")
    print(summary[[c for c in show if c in summary.columns]].round(3).to_string(index=False))
    if len(tests):
        print("\nUji berpasangan per-skenario (akurasi, Wilcoxon; tanpa koreksi multi-uji):")
        print(tests.round(4).to_string(index=False))
    print("\nCatatan: std di atas adalah variasi karena pembagian fold, bukan interval kepercayaan;")
    print("estimasi sesungguhnya hanya bertumpu pada 30 skenario.")
    print(f"Hasil disimpan di: {results_dir}  ({time.time() - started:.0f} detik)")
    return summary
 
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Random Forest severity classifier (E0-E3)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_ORDER), choices=CONFIG_ORDER)
    parser.add_argument("--splits", type=int, default=5, help="jumlah fold per repeat")
    parser.add_argument("--repeats", type=int, default=10, help="jumlah pengulangan CV")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--probe-config", default="E0", choices=[*CONFIG_ORDER, ""],
                        help="konfigurasi untuk probe service/kategori fault; kosong = lewati")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--save-fold-models", nargs="*", default=None,
                        metavar="CONFIG",
                        help="simpan model tiap fold (dibutuhkan explainer.py). "
                             "Sebutkan konfigurasinya, mis. --save-fold-models E0rel. "
                             "Tanpa nilai = semua yang dijalankan (ukurannya besar).")
    parser.add_argument("--relative", action="store_true",
                        help="jalankan juga model dengan fitur relatif "
                             "(penyelarasan ke service terdampak, dua tahap)")
    parser.add_argument("--relative-configs", nargs="+", default=list(CONFIG_ORDER),
                        choices=CONFIG_ORDER,
                        help="konfigurasi mana yang dijalankan versi relatifnya")
    return parser.parse_args()
 
 
def silence_known_warnings():
    """
    Membungkam dua peringatan yang sudah diketahui dan tidak mempengaruhi hasil.

    Yang kedua adalah keluhan sklearn 1.9 tentang dirinya sendiri: modul ensemble
    memanggil joblib lewat utils.parallel, dan utils.parallel mengeluhkan cara
    pemanggilan itu. Muncul sekali per pohon, sehingga pada run penuh jumlahnya
    ratusan ribu baris (log run sebelumnya 53 MB) dan penulisannya ke konsol
    ikut memperlambat eksekusi.

    Disaring lewat pola pesan, bukan seluruh UserWarning, supaya peringatan lain
    yang mungkin penting tetap terlihat.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
        category=UserWarning,
    )


if __name__ == "__main__":
    silence_known_warnings()
    run(parse_arguments())