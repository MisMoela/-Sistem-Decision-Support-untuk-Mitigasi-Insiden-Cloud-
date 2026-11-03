"""
Menjelaskan prediksi severity dengan TreeSHAP.

Memakai model PER FOLD, bukan model akhir. Jadi tiap kasus dijelaskan oleh
model yang belum pernah melihat kasus itu saat latihan. Kalau memakai model
akhir, penjelasannya dihitung di data latihnya sendiri dan jadi tidak jujur.
"""


import argparse
import json
import time
import warnings 
from pathlib import Path

import joblib
import numpy as np 
import pandas as pd
import shap


# =====================================================================
# 1. KONFIGURASI
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "training"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "classifier"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "explainer"

SEVERITY_ORDER = ["Low", "Medium", "High"]

# Batas selisih yang masih ditoleransi pada uji aditivitas
# Nilai wajar sekitar 1e-15, sedangkan 1e-6 sudah sangat longgar
ADDITIVITY_TOLERANCE = 1e-15

# Batas keyakinan tahap 1, diukur pada E0rel dengan rata-rata keyakinan
# per kasus (yang dipakai kode ini):
#   >= 0.7  40 kasus  service benar 1.000  severity benar 0.950
#   <  0.7  49 kasus  service benar 0.816  severity benar 0.714
# Tidak ada satu pun kasus di atas ambang yang service-nya salah tebak.
#
# Angka ini HANYA penanda untuk agent. Jangan dipakai menyaring hasil lalu
# melaporkan angka yang lebih tinggi sebagai mutu model: membuang kasus
# mengubah komposisi kelas, dan macro-F1 justru turun (0.631 vs 0.706).
# Mutu model tetap macro-F1 0.762 pada seluruh kasus.
LOCATOR_CONFIDENCE_FLOOR = 0.7



# =====================================================================
# 2. MEMUAT DATA
# =====================================================================

def load_fold_models(results_dir, config):
    """Membaca folds_<config>.joblib hasil classifier.py."""
    path = results_dir / "models" / f"folds_{config}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada. Jalankan classifier.py dengan "
            f"--save-fold-models {config} lebih dahulu"
        )

    bundle = joblib.load(path)
    folds = bundle["folds"]
    feature_names = list(bundle["feature_names"])

    if not folds:
        raise RuntimeError(f"{path.name}: daftar fold kosong")
    if list(bundle["classes"]) != SEVERITY_ORDER:
        raise RuntimeError(
            f"{path.name}: urutan kelas {bundle['classes']} "
            f"berbeda dari {SEVERITY_ORDER}"
        )

    print(f"[OK] {len(folds)} fold dibaca dari {path.name}")
    return folds, feature_names


def load_oof(results_dir, config):
    """Membaca prediksi out-of-fold; dipakai untuk menentukan kelas yang dijelaskan."""
    path = results_dir / f"oof_predictions_{config}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada")

    oof = pd.read_csv(path)
    case_ids = oof.loc[oof["repeat"] == 0, "case_id"].tolist()
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError(f"{path.name}: case_id duplikat pada repeat 0")

    print(f"[OK] {len(case_ids)} kasus, {oof['repeat'].nunique()} ulangan")
    return oof, case_ids


def load_absolute_features(data_dir, config, case_ids):
    """
    Cadangan untuk konfigurasi ABSOLUT (E0..E3), yang foldnya tidak menyimpan
    matriks fitur karena fiturnya sama di semua fold.
    """

    frame = pd.read_csv(data_dir / f"features_{config}.csv")
    if frame["case_id"].tolist() != list(case_ids):
        raise RuntimeError(f"features_{config}.csv: urutan case_id berbeda")
    columns = [c for c in frame.columns if c != "case_id"]
    return frame[columns].to_numpy(dtype="float64")


def locator_feature_config(config):
    """
    Penebak service (tahap 1) selalu dilatih dari fitur ABSOLUT, bukan relatif.
    Jadi E0rel memakai features_E0.csv, E1rel memakai features_E1.csv.
    """

    return config[:-3] if config.endswith("rel") else config



# =====================================================================
# 3. PERHITUNGAN SHAP
# =====================================================================

def class_reorder_index(model):
    """
    Urutan kelas pada keluaran SHAP mengikuti model.classes_, dan itu urut
    ABJAD: High, Low, Medium. Fungsi ini memberi indeks untuk menukarnya
    menjadi Low, Medium, High.

    Tanpa ini, penjelasan untuk High akan tertempel ke Low tanpa pesan error.
    """

    fitted = list(model.classes_)
    missing = set(SEVERITY_ORDER) - set(fitted)
    if missing:
        raise RuntimeError(f"Model tidak memiliki kelas {sorted(missing)}")
    return [fitted.index(label) for label in SEVERITY_ORDER]


def shap_for_fold(fold, X_absolute=None):
    """
    Menghitung SHAP untuk satu fold.

    Mengembalikan:
        values : (n_uji, n_fitur, 3) sudah diurutkan Low, Medium, High
        base   : (3,) nilai dasar, juga sudah diurutkan
        X_test : (n_uji, n_fitur) nilai fitur yang dipakai
    """

    model = fold["model"]
    test_idx = fold["test_idx"]


    # konfigurasi relatif menyimpan matriksnya sendiri, karena fiturnya
    # berubah tiap fold (ikut service hasil tebakan)

    if "Xtest" in fold:
        X_test = fold["Xtest"]
    elif X_absolute is not None:
        X_test = X_absolute[test_idx]
    else:
        raise RuntimeError("fold tidak punya Xtest dan fitur absolut tidak diberikan")
    
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test, check_additivity=False)

    # shap >= 0.45 mengembalikan array 3 dimensi, versi lama daftar per kelas

    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype="float64")


    base = np.ravel(np.asarray(explainer.expected_value, dtype="float64"))

    order = class_reorder_index(model)
    return values[:, :, order], base[order], X_test


def verify_additivity(values, base, model, X_test, order):
    """
    Uji kejujuran: jumlah seluruh SHAP ditambah nilai dasar harus sama dengan
    probabilitas yang dikeluarkan model. Mengembalikan selisih terbesar.
    """

    predicted = model.predict_proba(X_test)[:, order]
    total = values.sum(axis=1) + base
    return float(np.abs(total - predicted).max())


# =====================================================================
# 4. AGREGASI ANTAR ULANGAN
# =====================================================================

def collect_all_folds(folds, n_cases, n_features, n_repeats, X_absolute=None):
    """
    Menjalankan SHAP di seluruh fold dan menyusunnya menjadi satu array besar.

    Keluaran:
        values : (n_ulangan, n_kasus, n_fitur, 3)
        inputs : (n_ulangan, n_kasus, n_fitur) nilai fitur
        worst  : selisih aditivitas terbesar yang ditemukan
    """

    values = np.full((n_repeats, n_cases, n_features, len(SEVERITY_ORDER)), np.nan)
    inputs = np.full((n_repeats, n_cases, n_features), np.nan)
    worst = 0.0

    for position, fold in enumerate(folds, start = 1):
        fold_values, base, X_test = shap_for_fold(fold, X_absolute)
        order = class_reorder_index(fold["model"])
        worst = max(worst, verify_additivity(fold_values, base, fold["model"], X_test, order))


        repeat = fold["repeat"]
        test_idx = fold["test_idx"]
        values[repeat, test_idx] = fold_values
        inputs[repeat, test_idx] = X_test

        if position % 10 == 0 or position == len(folds):
            print(f"[OK] SHAP {position}/{len(folds)} fold")

    if np.isnan(values).any():
        raise RuntimeError("Ada kasus yang tidak terjelaskan di sebagian ulangan")

    return values, inputs, worst

def summarize(values, inputs):
    """Rata-rata dan simpangan baku antar ulangan."""
    return {
        "shap_mean": values.mean(axis = 0),
        "shap_std": values.std(axis = 0, ddof = 1),
        "value_mean": inputs.mean(axis = 0),
    }


def collect_locator_signals(folds, X_locator, n_cases, n_repeats):
    """
    Mengambil tebakan service dan keyakinan tahap 1 untuk tiap kasus.

    Ini perlu karena fitur relatif diambil dari service hasil tebakan. Kalau
    tebakannya salah, SHAP menjelaskan data milik service yang salah, dan tidak
    ada pesan error apa pun. Keyakinan tahap 1 dapat dihitung tanpa tahu
    jawaban yang benar, jadi bisa dipakai saat sistem berjalan sungguhan.

    Mengembalikan None untuk konfigurasi absolut, yang tidak punya tahap 1.
    """

    if X_locator is None or "locator" not in folds[0]:
        return None

    expected = getattr(folds[0]["locator"], "n_features_in_", None)
    if expected is not None and expected != X_locator.shape[1]:
        raise RuntimeError(
            f"Tahap 1 dilatih dengan {expected} fitur, "
            f"tetapi datanya {X_locator.shape[1]} kolom"
        )

    names = np.empty((n_repeats, n_cases), dtype=object)
    confidence = np.full((n_repeats, n_cases), np.nan)

    for fold in folds:
        locator = fold["locator"]
        test_idx = fold["test_idx"]
        proba = locator.predict_proba(X_locator[test_idx])
        names[fold["repeat"], test_idx] = locator.classes_[proba.argmax(axis=1)]
        confidence[fold["repeat"], test_idx] = proba.max(axis=1)

    if np.isnan(confidence).any():
        raise RuntimeError("Ada kasus tanpa tebakan tahap 1 di sebagian ulangan")

    return {"names": names, "confidence": confidence}


def summarize_locator(signals, n_cases):
    """
    Service yang paling sering ditebak, dan rata-rata keyakinannya.

    Catatan: kesepakatan antar ulangan TIDAK dipakai sebagai penanda. Dua kasus
    terburuk (emailservice_delay-2 dan currencyservice_socket-1) sepakat 10/10
    tetapi sepakat pada service yang salah, jadi kesepakatan justru menandainya
    sebagai paling dapat dipercaya. Keyakinan menangkap keduanya.
    """

    if signals is None:
        return None

    chosen = [
        pd.Series(signals["names"][:, case_index]).value_counts().index[0]
        for case_index in range(n_cases)
    ]
    return {
        "service": chosen,
        "confidence": signals["confidence"].mean(axis=0),
    }


def majority_prediction(oof, case_ids):
    """
    Kelas yang paling sering diprediksi untuk tiap kasus, dari 10 ulangan.
    Inilah kelas yang dijelaskan di file ringkas.
    """

    counts = oof.groupby(["case_id", "y_pred"]).size().rename("n").reset_index()
    top = counts.sort_values(["case_id", "n"], ascending=[True, False])
    top = top.drop_duplicates("case_id").set_index("case_id")["y_pred"]
    truth = oof.drop_duplicates("case_id").set_index("case_id")["y_true"]
    return(
        pd.DataFrame({"y_pred_majority": top, "y_true": truth})
        .reindex(case_ids)
        .reset_index()
        .rename(columns={"index": "case_id"})
    )

# =====================================================================
# 5. KELUARAN
# =====================================================================

def write_full_table(output_dir, config, case_ids, feature_names, summary):
    """Satu baris per kasus per fitur per kelas. Arsip lengkap."""
    rows = []
    for case_index, case_id in enumerate(case_ids):
        for class_index, label in enumerate(SEVERITY_ORDER):
            for feature_index, feature in enumerate(feature_names):
                rows.append({
                    "case_id": case_id,
                    "severity_class": label,
                    "feature": feature,
                    "shap_mean": summary["shap_mean"][case_index, feature_index, class_index],
                    "shap_std": summary["shap_std"][case_index, feature_index, class_index],
                    "feature_value": summary["value_mean"][case_index, feature_index],
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"shap_values_{config}.csv", index=False)
    return frame

def write_top_features(output_dir, config, case_ids, feature_names, summary,
                       predictions, top_k, locator=None):
    """
    Fitur paling berpengaruh untuk kelas yang diprediksi, per kasus.
    File inilah yang dibaca agent_loop.py.
    """
    rows = []
    for case_index, case_id in enumerate(case_ids):
        predicted = predictions.loc[case_index, "y_pred_majority"]
        class_index = SEVERITY_ORDER.index(predicted)

        contribution = summary["shap_mean"][case_index, :, class_index]
        ranking = np.argsort(np.abs(contribution))[::-1][:top_k]

        for rank, feature_index in enumerate(ranking, start=1):
            value = contribution[feature_index]
            baris = {
                "case_id": case_id,
                "y_true": predictions.loc[case_index, "y_true"],
                "y_pred": predicted,
                "rank": rank,
                "feature": feature_names[feature_index],
                "shap_mean": value,
                "shap_std": summary["shap_std"][case_index, feature_index, class_index],
                "feature_value": summary["value_mean"][case_index, feature_index],
                "arah": "mendorong" if value > 0 else "menahan",
            }

            # Dua kolom berikut memberi tahu agent apakah penjelasan ini
            # dibangun dari service yang tepat. Lihat summarize_locator.
            if locator is not None:
                keyakinan = float(locator["confidence"][case_index])
                baris["service_predicted"] = locator["service"][case_index]
                baris["locator_confidence"] = keyakinan
                baris["penjelasan_andal"] = keyakinan >= LOCATOR_CONFIDENCE_FLOOR

            rows.append(baris)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"shap_top_features_{config}.csv", index=False)
    return frame


def write_global_importance(output_dir, config, feature_names, summary):
    """Rata-rata besar pengaruh tiap fitur di seluruh kasus. Untuk tabel di naskah."""
    magnitude = np.abs(summary["shap_mean"])          # (n_kasus, n_fitur, 3)
    rows = []
    for feature_index, feature in enumerate(feature_names):
        row = {
            "feature": feature,
            "mean_abs_shap": magnitude[:, feature_index, :].mean(),
        }
        for class_index, label in enumerate(SEVERITY_ORDER):
            row[f"mean_abs_shap_{label}"] = magnitude[:, feature_index, class_index].mean()
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    frame.to_csv(output_dir / f"shap_global_importance_{config}.csv", index=False)
    return frame


# =====================================================================
# 6. ORKESTRATOR
# =====================================================================
def run(args):
    started = time.time()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Explainer] Konfigurasi {args.config}")

    folds, feature_names = load_fold_models(results_dir, args.config)
    oof, case_ids = load_oof(results_dir, args.config)

    n_cases = len(case_ids)
    n_features = len(feature_names)
    n_repeats = int(oof["repeat"].nunique())

    # Konfigurasi absolut tidak menyimpan matriks fitur di tiap fold.
    X_absolute = None
    if "Xtest" not in folds[0]:
        X_absolute = load_absolute_features(Path(args.data_dir), args.config, case_ids)
        print(f"[OK] Fitur absolut dibaca: {X_absolute.shape[1]} kolom")

    # Tahap 1 membaca fitur absolut, yang untuk konfigurasi relatif tidak ikut
    # tersimpan di fold. Jadi dibaca terpisah dari berkas konfigurasi dasarnya.
    X_locator = X_absolute
    if X_locator is None and "locator" in folds[0]:
        X_locator = load_absolute_features(
            Path(args.data_dir), locator_feature_config(args.config), case_ids
        )
        print(f"[OK] Fitur tahap 1 dibaca: {X_locator.shape[1]} kolom")

    print(f"[Info] {n_cases} kasus x {n_features} fitur x {n_repeats} ulangan")

    values, inputs, worst = collect_all_folds(
        folds, n_cases, n_features, n_repeats, X_absolute
    )

    print(f"[OK] Uji aditivitas: selisih terbesar {worst:.2e}")
    if worst > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"Selisih aditivitas {worst:.2e} melebihi batas {ADDITIVITY_TOLERANCE:.0e}. "
            "Nilai SHAP tidak dapat dipercaya."
        )

    summary = summarize(values, inputs)
    predictions = majority_prediction(oof, case_ids)

    locator = summarize_locator(
        collect_locator_signals(folds, X_locator, n_cases, n_repeats), n_cases
    )
    if locator is not None:
        andal = int((locator["confidence"] >= LOCATOR_CONFIDENCE_FLOOR).sum())
        print(f"[OK] Keyakinan tahap 1: rata-rata {locator['confidence'].mean():.3f}, "
              f"{andal}/{n_cases} kasus di atas {LOCATOR_CONFIDENCE_FLOOR}")

    write_full_table(output_dir, args.config, case_ids, feature_names, summary)
    top = write_top_features(output_dir, args.config, case_ids, feature_names,
                             summary, predictions, args.top_k, locator)
    importance = write_global_importance(output_dir, args.config, feature_names, summary)

    with (output_dir / f"explainer_config_{args.config}.json").open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "config": args.config,
            "shap": shap.__version__,
            "n_cases": n_cases,
            "n_features": n_features,
            "n_repeats": n_repeats,
            "n_folds": len(folds),
            "top_k": args.top_k,
            "severity_order": SEVERITY_ORDER,
            "additivity_max_error": worst,
            "explained_with": "model per fold (out-of-fold)",
            "locator_confidence_floor": LOCATOR_CONFIDENCE_FLOOR,
            "locator_confidence_mean": (
                None if locator is None else float(locator["confidence"].mean())
            ),
            "n_cases_penjelasan_andal": (
                None if locator is None
                else int((locator["confidence"] >= LOCATOR_CONFIDENCE_FLOOR).sum())
            ),
        }, file, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 200)
    print("\n========== 10 FITUR PALING BERPENGARUH ==========")
    print(importance.head(10).round(4).to_string(index=False))

    print("\n========== CONTOH SATU KASUS ==========")
    sample = top[top["case_id"] == case_ids[0]]
    print(f"Kasus: {case_ids[0]}  (sebenarnya {sample['y_true'].iloc[0]}, "
          f"diprediksi {sample['y_pred'].iloc[0]})")
    if locator is not None:
        print(f"Tahap 1 menebak {sample['service_predicted'].iloc[0]} "
              f"dengan keyakinan {sample['locator_confidence'].iloc[0]:.3f} "
              f"-> penjelasan andal: {sample['penjelasan_andal'].iloc[0]}")
    print(sample[["rank", "feature", "shap_mean", "feature_value", "arah"]]
          .round(4).to_string(index=False))

    if locator is not None:
        ragu = top[~top["penjelasan_andal"]].drop_duplicates("case_id")
        print(f"\n========== {len(ragu)} KASUS YANG PENJELASANNYA PERLU HATI-HATI ==========")
        print(ragu.nsmallest(10, "locator_confidence")[
            ["case_id", "y_true", "y_pred", "service_predicted", "locator_confidence"]
        ].round(3).to_string(index=False))

    print(f"\nHasil disimpan di: {output_dir}  ({time.time() - started:.0f} detik)")
    return top


def parse_arguments():
    parser = argparse.ArgumentParser(description="TreeSHAP untuk model severity")
    parser.add_argument("--config", default="E0rel",
                        help="konfigurasi yang dijelaskan, mis. E0rel")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=5,
                        help="berapa fitur teratas per kasus")
    return parser.parse_args()


def silence_known_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
        category=UserWarning,
    )


if __name__ == "__main__":
    silence_known_warnings()
    run(parse_arguments())