from pathlib import Path
import json
import pandas as pd
import argparse
import re
import hashlib
import sys
import sklearn
from collections import defaultdict, Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier



#===================================#
# 1. DATASET CONFIGURATION 
#===================================#   
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "RE2-OB"
    / "RE2-OB"
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OBSERVATION_BEFORE_MINUTES = 5
OBSERVATION_AFTER_MINUTES = 5


TRACE_CHUNK_SIZE = 100_000

#=========== DERIVED SEVERITY CONFIGS (FCW x SCT) ===========#

LABEL_POLICY = {
    # Naikkan policy_version dan set is_frozen=True setelah tabel disepakati
    "policy_version": "fcw_sct_v1",
    "is_frozen": False,
    "method": "fcw_x_sct",

    # Fault Category Weight (FCW)
    # resource : mendegradasi kapasitas satu service (dampak lokal)
    # network  : memutus komunikasi antar service (dampak merambat)
    "fault_category": {
        "cpu": "resource",
        "mem": "resource",
        "disk": "resource",
        "delay": "network",
        "loss": "network",
        "socket": "network",
    },
    "fault_category_weight": {
        "resource": 1,
        "network": 2,
    },

    # Service Criticality Tier (SCT)
    # Tier ditetapkan dari arsitektur Online Boutique, BUKAN dari telemetri,
    # agar tidak beririsan dengan fitur service context pada E3
    "service_tier": {
        "checkoutservice": "A",
        "productcatalogservice": "A",
        "currencyservice": "B",
        "recommendationservice": "C",
        "emailservice": "C",
    },
    "service_tier_weight": {
        "A": 3,
        "B": 2,
        "C": 1,
    },

    # Severity Score = FCW x SCT, dipetakan ke tiga kelas
    "severity_bins": [
        {"min": 1, "max": 2, "label": "Low"},
        {"min": 3, "max": 4, "label": "Medium"},
        {"min": 5, "max": 6, "label": "High"},
    ],
}

# Skema tier alternatif untuk sensitivity analysis:
# menunjukkan seberapa berubah distribusi kelas jika keputusan tier berbeda
LABEL_SENSITIVITY = {
    "alternative_service_tiers": {
        "checkout_only_tier_a": {
            "checkoutservice": "A",
            "productcatalogservice": "B",
            "currencyservice": "B",
            "recommendationservice": "C",
            "emailservice": "C",
        },
        "currency_tier_a": {
            "checkoutservice": "A",
            "productcatalogservice": "A",
            "currencyservice": "A",
            "recommendationservice": "C",
            "emailservice": "C",
        },
    },
}

SEVERITY_ORDER = ["Low", "Medium", "High"]

# Fase 7: apakah keempat time-series juga wajib menutup window penuh?
# False -> hanya raw trace yang wajib (E2/E3 bergantung padanya); bin kosong
#          pada time-series diserahkan ke Fase 8  -> 89 run, sama seperti sekarang
# True  -> semua sumber wajib penuh               -> 88 run, Fase 7 perlu re-run
REQUIRE_FULL_TIMESERIES_COVERAGE = False


#============ FEATURE CLEANING CONFIG (FASE 8) ===============#

CLEANED_TIMESERIES_ROOT = (OUTPUT_ROOT / "cleaned_timeseries")


#Metric kontinue (gauge) : ketika ada missing value maka itu adalah celah pengukuran nantinya akan di interpolasi
GAUGE_METRIC_SUFFIXES = ("_cpu", "_mem", "_socket")


#Bin kosong yang masih ditoleransi per window per modality
MAX_MISSING_BIN_RATIO = 0.10


#Berapaa bin di tepi window yang boleh diisi dari tetangga terdekat
MAX_EDGE_FILL_BINS = 2

#Kolom gauge yang ada di kurang dari sekian bagian run di buang ()
MIN_GAUGE_COLUMN_COVERAGE = 0.5

# Kategori template log. Urutan penting: dicek dari atas; yang tidak cocok -> "routine"
LOG_TEMPLATE_CATEGORIES = (
    ("lifecycle", re.compile(
        r"initializing|listening on port|profil(?:er|ing) disabled|tracing enabled|"
        r"stats enabled|catalog address|env platform|metadata server",
        re.IGNORECASE,
    )),
    ("error", re.compile(
        r"error|fail|exception|unavailable|deadline|timeout|refused|panic|raise|statuscode",
        re.IGNORECASE,
    )),
)

# Cara mengisi nilai kosong per keluarga fitur
#   zero        : kosong/absen berarti "tidak ada kejadian" -> 0
#   interpolate : kosong berarti celah pengukuran -> interpolasi linear dalam run
FILL_POLICY = {
    "continuous_resource": "interpolate",
    "workload_signal": "interpolate",
    "latency_signal": "interpolate",
    "trace_latency": "interpolate",
    "disk_activity": "zero",
    "error_signal": "zero",
    "log_count": "zero",
    "trace_error": "zero",
}

MODALITY_ORDER = ("metric", "log", "trace_latency", "trace_error")

ALIGNED_META_COLUMNS = [
    "case_id", "scenario", "service", "fault_type", "run_number", "inject_time",
    "bin_index", "window_type", "relative_start_seconds", "relative_end_seconds",
    "bin_start", "bin_end",
]


#============ FEATURE EXTRACTION CONFIG (FASE 9) ===============#

# Lima angka ringkasan yang dihitung dari 40 bin untuk setiap kolom sumber.
# Urutan ini menentukan urutan kolom pada tabel fitur.
#   baseline_mean : kondisi normal sebelum injeksi
#   incident_max  : nilai tertinggi saat incident
#   delta_mean    : selisih rata-rata incident terhadap rata-rata baseline
#   zscore_max    : seberapa tidak wajar PUNCAK incident dibanding goyangan normal
#   zscore_mean   : seberapa tidak wajar RATA-RATA incident dibanding goyangan normal
# incident_mean sengaja tidak dihitung karena nilainya persis sama dengan
# baseline_mean + delta_mean (tidak menambah informasi, hanya memecah atribusi SHAP).
FEATURE_STATISTICS = (
    "baseline_mean",
    "incident_max",
    "delta_mean",
    "zscore_max",
    "zscore_mean",
)

# Batas atas/bawah z-score. Dipakai untuk dua hal:
#   1. pengganti nilai ketika sebaran baseline = 0 tetapi nilainya berubah saat incident
#   2. memotong z-score yang terlalu ekstrem karena sebaran baseline sangat kecil
ZSCORE_CAP = 100.0


# Awalan kolom sumber -> konfigurasi tempat fitur itu pertama kali masuk
MODALITY_CONFIG = {
    "metric": "E0",
    "log": "E1",
    "trace_latency": "E2",
    "trace_error": "E2",
    "svc": "E3",   
}


# Fitur service context (E3) dikerjakan terpisah di bagian berikutnya (9.5).
# False = lewati dulu, sehingga Fase 9 menghasilkan fitur untuk E0, E1, dan E2 saja.
INCLUDE_SERVICE_CONTEXT = True


# Kolom yang TIDAK BOLEH muncul di tabel fitur, karena merupakan label
# atau bahan pembentuk label. Diperiksa ulang pada Fase 11.
FORBIDDEN_FEATURE_COLUMNS = (
    "severity", "severity_score", "fcw", "sct", "fault_category", "service_tier",
    "service", "fault_type", "scenario", "run_number", "inject_time",
)


#============ SERVICE CONTEXT CONFIG (FASE 9.5) ===============#

# True = call graph dibaca ulang dari traces.csv (±4 menit) walau cache sudah ada.
# False = kalau service_call_graph.csv sudah ada, pakai itu.
REBUILD_CALL_GRAPH = False

# Pola nama operasi gRPC di Online Boutique: "hipstershop.CartService/GetCart".
# Bagian dalam kurung ("CartService") adalah service yang DIPANGGIL.
CALLEE_PATTERN = re.compile(r"hipstershop\.([A-Za-z]+)/")

# Kolom trace yang bukan service bisnis: health check dan pengiriman trace ke Jaeger
SERVICE_CONTEXT_EXCLUDED_SERVICES = ("health", "traceservice")

# Aturan "service ini anomali": zscore_mean > 3 DAN rata-rata incident > 1,5x baseline.
# Dikunci; tidak diubah setelah eksperimen dimulai.
SERVICE_ANOMALY_ZSCORE_THRESHOLD = 3.0
SERVICE_ANOMALY_RATIO_THRESHOLD = 1.5

# Sebuah hubungan antar-service dianggap "melemah" jika jumlah panggilannya
# saat incident turun lebih dari 20% dibanding baseline
CALL_VOLUME_DROP_THRESHOLD = 0.2

# Proxy audit: fitur svc__ dibuang jika
#   - satu fitur saja sudah menebak tier service dengan balanced accuracy >= 0.85, atau
#   - nilainya hampir tidak berubah di dalam satu service (rasio varians < 0.05),
#     artinya fitur itu hanya "kode nama service", bukan sinyal telemetri
PROXY_AUDIT_MAX_TIER_ACCURACY = 0.85
PROXY_AUDIT_MIN_WITHIN_VARIANCE_RATIO = 0.05
PROXY_AUDIT_TREE_DEPTH = 2




#============ DATASET ASSEMBLY CONFIG (FASE 10) ===============#

# Folder tujuan: seluruh file yang dibaca classifier.py ada di sini, tidak tercampur
# dengan laporan audit
TRAINING_DATASET_ROOT = (OUTPUT_ROOT / "training")

# Urutan konfigurasi. E1 memuat semua kolom E0, E2 memuat semua kolom E1, dst.
CONFIG_ORDER = ("E0", "E1", "E2", "E3")

# Kolom identitas run. Dipakai untuk pengelompokan cross-validation dan evaluasi,
# TIDAK PERNAH menjadi fitur model
CASE_METADATA_COLUMNS = ["case_id", "scenario", "service", "fault_type", "run_number"]




#============ READINESS CONFIG (FASE 11) ===============#

# Tujuh file yang wajib ada di folder training; semuanya diberi sidik jari (md5)
TRAINING_FILES = (
    "features_E0.csv", "features_E1.csv", "features_E2.csv", "features_E3.csv",
    "derived_severity_labels.csv", "case_metadata.csv", "config_definition.json",
)

# Kolom di file label yang merupakan bahan pembentuk severity.
# Fase 11 memastikan tidak ada kolom fitur yang nilainya persis sama dengan salah satunya.
LABEL_INGREDIENT_COLUMNS = ("fcw", "sct", "severity_score")

# Daftar file/folder yang memang dihasilkan kode saat ini di data/processed.
# Apa pun di luar daftar ini (dan file Fase 6 yang namanya bergantung versi)
# dilaporkan sebagai "tidak dikenal" - biasanya sisa dari kode versi lama.
EXPECTED_PROCESSED_ENTRIES = (
    "timeseries_quality_report.csv", "missing_by_column_report.csv",
    "raw_trace_structure_report.csv", "root_span_identity_report.csv",
    "root_service_operation_report.csv", "span_status_code_report.csv",
    "alignment_excluded_runs.csv", "aligned_timeseries_manifest.csv",
    "alignment_quality_report.csv", "alignment_config.json", "aligned_timeseries",
    "feature_column_schema.csv", "feature_eligibility.csv", "cleaning_log.csv",
    "column_harmonization_map.csv", "log_template_catalog.csv",
    "cleaned_timeseries_manifest.csv", "cleaning_config.json", "cleaned_timeseries",
    "features_by_case.csv", "feature_manifest.csv", "feature_extraction_report.json",
    "service_call_graph.csv", "service_call_volume_by_case.csv",
    "service_context_proxy_audit.csv", "training",
    "preprocessing_readiness_report.json", "leakage_audit.json",
)




#============ TIMESERIES ALIGNMENT CONFIG ===============#

ALIGNMENT_INTERVAL_SECONDS = 15

ALIGNMENT_TIMESERIES_ROOT = (OUTPUT_ROOT  / "aligned_timeseries")

ALIGNMENT_SPEC = {
    "metrics" : {
        "prefix": "metric",
        "aggregation": "mean",
    },"logs": {
        "prefix": "log",
        "aggregation": "sum",
    },"trace_latency": {
        "prefix": "trace_latency",
        "aggregation": "mean"
    }, "trace_error": {
        "prefix": "trace_error",
        "aggregation": "sum"
    },
}

EXPECTED_SERVICES = {
    "productcatalogservice",
    "recommendationservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
}



EXPECTED_FAULTS = {
    "cpu",
    "disk",
    "mem",
    "delay",
    "loss",
    "socket",
}

EXPECTED_RUNS = {"1", "2", "3"}

TIMESERIES_FILES = {
    "metrics": "simple_metrics.csv",
    "trace_latency": "tracets_lat.csv",
    "trace_error": "tracets_err.csv",
    "logs": "logts.csv",
}


REQUIRED_FILES = set(
    TIMESERIES_FILES.values()
) | {
   "traces.csv",
    "cluster_info.json",
    "inject_time.txt",
}

TRACE_REQUIRED_COLUMNS = {
   "time",
   "traceID",
   "spanID",
   "serviceName",
   "methodName",
   "operationName",
   "startTimeMillis",
   "startTime",
   "duration",
   "statusCode",
   "parentSpanID",   
}

#===================================#
# 2. VALIDATING DATASET STRUCTURE
#===================================#

def discover_runs():
    #Guna nyauntuk periksa struktur scenario dan nemuin seluruh folder run dalam dataset#

    #Check apakah root dataset tersedia atau tidak#
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root folder not found: {DATASET_ROOT}")

    #Membentuk 30 nama scenario yang seharusnya ada
    expected_scenarios = {
        f"{service}_{fault}"
        for service in EXPECTED_SERVICES
        for fault in EXPECTED_FAULTS
    }

    #Mengambil nama folder scenario yang benar benar ada di dataset
    actual_scenarios = {
        path.name
        for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    }

    #Mencari scenario yang hilang jika ada
    missing_scenarios = (
        expected_scenarios - actual_scenarios
    )

    #Mencari folder scenario yang tidak dikenal
    extra_scenarios = (
        actual_scenarios - expected_scenarios
    )

    if missing_scenarios:
        raise ValueError(
            "Scenario berikut tidak ditemukan dalam dataset: "
            f"{sorted(missing_scenarios)}"
        )

    if extra_scenarios:
        print(
            "[WARNING!!] scenario berikut tidak dikenal dalam dataset: ",
            sorted(extra_scenarios),
        )

    run_paths = []

    #Periksa run 1, 2, dan 3 pada setiap scenario
    for scenario_name in sorted(expected_scenarios):
        scenario_path = (
            DATASET_ROOT / scenario_name
        )

        actual_runs = {
            path.name
            for path in scenario_path.iterdir()
            if path.is_dir()
        }

        missing_runs = (
            EXPECTED_RUNS - actual_runs
        )

        extra_runs = (
            actual_runs - EXPECTED_RUNS
        )

        if missing_runs:
            raise ValueError(
                f"Run pada {scenario_name} tidak lengkap "
                f"Run yang hilang: {sorted(missing_runs)}"
            )

        if extra_runs:
            print(
                f"[WARNING!!] Run tambahan pada:"
                f"{scenario_name}"
                f"{sorted(extra_runs)}"
            )

        for run_number in sorted(EXPECTED_RUNS, key = int,):
            run_path = (
                scenario_path / run_number
            )

            run_paths.append(run_path)

    #Memastikan hasil akhirnya tepat 90 run
    if len(run_paths) != 90:   
        raise ValueError(
            f"Jumlah run yang ditemukan tidak sesuai: {len(run_paths)}"
        )
    
    return run_paths

def validate_run_files(run_path):

    #Digunakan untuk memeriksa tujuh file yang diperlukan ada dalam 1 folder run
    #Validasi ini hanya membaca bagian header CSV nya, bukan loading seluruh dataset

    if not run_path.is_dir():
        raise FileNotFoundError(
            f"Folder run tidak ditemukan: {run_path}"
        )

    #Mengambil nama file yang ada di folder run
    actual_files = {
        path.name
        for path in run_path.iterdir()
        if path.is_file()
    }

    #Membandingkan dengan tujuh file wajib
    missing_files = (
        REQUIRED_FILES - actual_files
    )

    extra_files = (
        actual_files - REQUIRED_FILES
    )

    if missing_files:
        raise FileNotFoundError(
            f"File tidak lengkap pada {run_path}"
            f"File yang hilang: "
            f"{sorted(missing_files)}"
        )

    if extra_files:
        print(
            f"[WARNING!!] File tambahan pada"
            f"{run_path}: "
            F"{sorted(extra_files)}"
        )

    #Memastikan ketujuh file tidak kosong
    for filename in REQUIRED_FILES:
        file_path = run_path / filename

        if file_path.stat().st_size == 0:
            raise ValueError(
                f"File kosong pada {file_path}"
            )

    #Memeriksa header empat CSV timeseries
    for data_name, filename in (
        TIMESERIES_FILES.items()
    ):
        file_path = run_path / filename

        try:
            header = pd.read_csv(
                file_path,
                nrows = 0,
            )

        except Exception as error:
            raise ValueError(
                f"{data_name} tidak dapat dibaca:"
                f"{file_path}. Error: {error}"
            )from error

        if "time" not in header.columns:
            raise ValueError(
                f"{data_name} tidak ditemukan pada:"
                f"{data_name}: {file_path}"
            )

        if len(header.columns) <= 1:
            raise ValueError(
                f"tidak ada kolom fitur pada "
                f"{file_path}"
            )

    #Memeriksa header traces.csv
    traces_path = run_path / "traces.csv"

    try:
        traces_header = pd.read_csv(
            traces_path,
            nrows = 0,
        )

    except Exception as error:
        raise ValueError(
            f"traces.csv tidak dapat dibaca: "
            f"{traces_path}. Error: {error}"
        )from error

    missing_trace_columns = (
        TRACE_REQUIRED_COLUMNS - set(traces_header.columns)
    )

    if missing_trace_columns:
        raise ValueError(
            f"Kolom traces.csv tidak lengkap pada "
            f"{traces_path}. Kolom yang hilang"
            f"{sorted(missing_trace_columns)}"
        )

    #Periksa cluster_info.json
    cluster_path = ( run_path / "cluster_info.json" )

    try:
        with open(cluster_path, encoding = "utf-8") as file:
            cluster_info = json.load(file)

    except (
        OSError, json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"JSON tidak valid atau tidak dapat dibaca: "
            f"{cluster_path}. Error: {error}"
        ) from error

    if not isinstance(cluster_info, dict):
        raise ValueError(
            f"cluster_info.json harus berbentuk dictionary pada :{cluster_path}"
        )

    if not cluster_info:
        raise ValueError(
            f"cluster_info.json kosong pada :{cluster_path}"
        )

    #Memeriksa inject_time.txt
    inject_path = ( run_path / "inject_time.txt" )

    try:
        inject_text = (
            inject_path.read_text(encoding = "utf-8").strip()
        )

        inject_epoch = int(inject_text)

        if inject_epoch <= 0:
            raise ValueError(
                "Unix timestamp harus lebih dari 0"
            )

        pd.to_datetime(
            inject_epoch, unit = "s", utc = True
        )

    except (ValueError, OverflowError, OSError) as error:
        raise ValueError(
            f"inject_time.txt tidak valid :"
            f"{inject_path}. Error: {error}"
        ) from error

    
    return True


# ===================================
# 3. LOADING AND BASIC VALIDATION
# ===================================

# Fase 3.1 - Membaca satu file time series dan membentuk timestamp UTC.
def load_timeseries(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe["time"] = pd.to_numeric(dataframe["time"], errors="raise")

    value_columns = [
        column for column in dataframe.columns if column != "time"
    ]
    dataframe[value_columns] = dataframe[value_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["time"],
        unit="s",
        utc=True,
    )
    return dataframe


# Fase 3.2 - Memuat empat time-series dan metadata untuk satu run.
def load_run(run_path):
    run_path = Path(run_path)
    validate_run_files(run_path)

    scenario_name = run_path.parent.name
    service_name, fault_type = scenario_name.rsplit("_", maxsplit=1)
    timeseries_data = {
        data_name: load_timeseries(run_path / file_name)
        for data_name, file_name in TIMESERIES_FILES.items()
    }

    with open(run_path / "cluster_info.json", encoding="utf-8") as file:
        cluster_info = json.load(file)

    inject_epoch = int(
        (run_path / "inject_time.txt").read_text(encoding="utf-8").strip()
    )
    return {
        "scenario": scenario_name,
        "service": service_name,
        "fault_type": fault_type,
        "run_number": run_path.name,
        "run_path": run_path,
        "inject_epoch": inject_epoch,
        "inject_time": pd.to_datetime(inject_epoch, unit="s", utc=True),
        "timeseries": timeseries_data,
        "traces_path": run_path / "traces.csv",
        "cluster_info": cluster_info,
    }


# Fase 3.3 - Memastikan hasil loading memenuhi struktur minimum untuk audit.
def validate_loaded_run_basic(run_data):
    if not isinstance(run_data, dict):
        raise TypeError("run_data harus berupa dictionary")

    required_keys = {
        "scenario",
        "service",
        "fault_type",
        "run_number",
        "run_path",
        "inject_time",
        "timeseries",
        "traces_path",
        "cluster_info",
    }
    missing_keys = required_keys - set(run_data)
    if missing_keys:
        raise KeyError(f"Key run_data tidak lengkap: {sorted(missing_keys)}")

    inject_time = run_data["inject_time"]
    if pd.isna(inject_time):
        raise ValueError(f"Inject time tidak valid: {run_data['run_path']}")

    timeseries = run_data["timeseries"]
    if not isinstance(timeseries, dict):
        raise TypeError("timeseries harus berupa dictionary")

    missing_data = set(TIMESERIES_FILES) - set(timeseries)
    if missing_data:
        raise KeyError(f"Time-series belum dimuat: {sorted(missing_data)}")

    for data_name, dataframe in timeseries.items():
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError(f"Data {data_name} kosong atau bukan DataFrame")

        missing_columns = {"time", "timestamp"} - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Kolom {data_name} tidak lengkap: {sorted(missing_columns)}"
            )

        valid_timestamps = dataframe["timestamp"].dropna()
        if valid_timestamps.empty:
            raise ValueError(f"{data_name} tidak memiliki timestamp valid")
        if not valid_timestamps.min() <= inject_time <= valid_timestamps.max():
            raise ValueError(f"Inject time di luar rentang {data_name}")

    if not Path(run_data["traces_path"]).is_file():
        raise FileNotFoundError(
            f"Raw trace tidak ditemukan: {run_data['traces_path']}"
        )
    if not isinstance(run_data["cluster_info"], dict):
        raise TypeError("cluster_info harus berupa dictionary")
    return True


# Fase 3.4 - Menjalankan loading dan basic validation dalam satu pemanggilan.
def load_validated_run(run_path):
    run_data = load_run(run_path)
    validate_loaded_run_basic(run_data)
    return run_data


# Fase 3.5 - Memuat banyak run bergantian agar penggunaan memori tetap aman.
def load_all_runs(run_paths):
    total_runs = len(run_paths)
    for index, run_path in enumerate(run_paths, start=1):
        try:
            run_data = load_validated_run(run_path)
        except Exception as error:
            raise RuntimeError(
                f"Gagal memuat atau memvalidasi run {index}/{total_runs}: "
                f"{run_path}"
            ) from error

        print(
            f"[OK] {index}/{total_runs}: "
            f"{run_data['scenario']}/run-{run_data['run_number']}"
        )
        yield run_data


# Fase 3.6 - Menampilkan sampel hasil loading tanpa mencetak seluruh dataset.
def show_run_summary(run_data):
    print("\n========== RUN SUMMARY ==========")
    for key in ("scenario", "service", "fault_type", "run_number", "inject_time"):
        print(f"{key.replace('_', ' ').title():12}: {run_data[key]}")

    for data_name, dataframe in run_data["timeseries"].items():
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {"time", "timestamp"}
        ]
        print(f"\n--- {data_name} ---")
        print("Shape:", dataframe.shape)
        print(
            "Range:",
            dataframe["timestamp"].min(),
            "sampai",
            dataframe["timestamp"].max(),
        )
        print(
            dataframe[["timestamp", *feature_columns[:4]]]
            .head(3)
            .to_string(index=False)
        )

    trace_preview = pd.read_csv(
        run_data["traces_path"],
        usecols=[
            "traceID",
            "spanID",
            "parentSpanID",
            "serviceName",
            "operationName",
            "startTimeMillis",
            "duration",
            "statusCode",
        ],
        nrows=5,
    )
    print("\n--- traces preview ---")
    print(trace_preview.to_string(index=False))

#===================================#
# 4. SUMMARY TIMESERIES QUALITY AUDIT
#===================================#


# Phase 4.1 - Run Level Quality Audit
def audit_run_quality(run_data):

    #Membuat laporan kualitas empat timeseries dalam satu run tanpa langsung menghentikan program

    records = []

    inject_time = run_data["inject_time"]

    required_start = (
        inject_time
        - pd.Timedelta(
            minutes=OBSERVATION_BEFORE_MINUTES
        )
    )

    required_end = (
        inject_time
        + pd.Timedelta(
            minutes=OBSERVATION_AFTER_MINUTES
        )
    )

    for data_name, dataframe in (
        run_data["timeseries"].items()
    ):
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {"time", "timestamp"}
        ]

        invalid_timestamp_count = int(
            dataframe["timestamp"].isna().sum()
        )

        missing_feature_count = int(
            dataframe[
                feature_columns
            ].isna().sum().sum()
        )

        valid_time_data = dataframe.dropna(
            subset=["timestamp"]
        )

        duplicate_timestamp_count = int(
            valid_time_data[
                "timestamp"
            ].duplicated().sum()
        )

        if valid_time_data.empty:
            start_time = pd.NaT
            end_time = pd.NaT
            seconds_before_injection = None
            seconds_after_injection = None
            window_covered = False

        else:
            start_time = valid_time_data[
                "timestamp"
            ].min()

            end_time = valid_time_data[
                "timestamp"
            ].max()

            seconds_before_injection = (
                inject_time - start_time
            ).total_seconds()

            seconds_after_injection = (
                end_time - inject_time
            ).total_seconds()

            window_covered = bool(
                start_time <= required_start
                and end_time >= required_end
            )

        issues = []

        if invalid_timestamp_count > 0:
            issues.append(
                "invalid_timestamp"
            )

        if missing_feature_count > 0:
            issues.append(
                "missing_feature"
            )

        if duplicate_timestamp_count > 0:
            issues.append(
                "duplicate_timestamp"
            )

        if not window_covered:
            issues.append(
                "insufficient_window"
            )

        status = (
            "valid"
            if not issues
            else "|".join(issues)
        )

        records.append({
            "scenario": run_data["scenario"],
            "service": run_data["service"],
            "fault_type": run_data["fault_type"],
            "run_number": run_data["run_number"],
            "data_name": data_name,
            "row_count": len(dataframe),
            "feature_count": len(feature_columns),
            "invalid_timestamp_count": (
                invalid_timestamp_count
            ),
            "missing_feature_count": (
                missing_feature_count
            ),
            "duplicate_timestamp_count": (
                duplicate_timestamp_count
            ),
            "start_time": start_time,
            "end_time": end_time,
            "inject_time": inject_time,
            "seconds_before_injection": (
                seconds_before_injection
            ),
            "seconds_after_injection": (
                seconds_after_injection
            ),
            "window_covered": window_covered,
            "status": status,
        })

    return records


def audit_all_runs(run_paths):

    #Melakukan Audit seluruh run tanpa berhenti ketika menemukan suatu masalah pada kualitas data

    quality_records = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(
        run_paths,
        start=1,
    ):

        try:
            run_data = load_validated_run(run_path)
            run_records = audit_run_quality(run_data)
            quality_records.extend(run_records)

            print(
                f"[AUDIT] {index}/{total_runs} "
                f"{run_data['scenario']}/"
                f"run-{run_data['run_number']}"
            )

        except (
            FileNotFoundError,
            OSError,
            ValueError,
            KeyError,
            pd.errors.ParserError,
        ) as error:
            quality_records.append({
                "scenario": run_path.parent.name,
                "run_number": run_path.name,
                "data_name": "run",
                "window_covered": False,
                "status": f"load_error: {error}"
            })

            print(
                f"[ERROR] {index}/{total_runs} "
                f"{run_path}: {error}"
            )
    
    return pd.DataFrame(
        quality_records
    )

#Phase 4.2 - Column Level Missing data audit

def longest_missing_streak(missing_mask):
    #Menghitung jumlah missing value terpanjang yang terjadi secara berturut

    if not missing_mask.any():
        return 0

    group_ids = missing_mask.ne(
        missing_mask.shift(
            fill_value = False
        )
    ).cumsum()

    streak_lengths = missing_mask.groupby(
        group_ids
    ).sum()

    return int(
        streak_lengths.max()
    )


def audit_missing_by_column(run_data):
    #Mencatat missing value per kolom untuk satu run, hanya kolom yang memiliki missing valu yang dicatat

    records = []

    inject_time = run_data["inject_time"]

    for data_name, dataframe in (
        run_data["timeseries"].items()
    ):
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {"time", "timestamp"}
        ]

        for column_name in feature_columns:
            missing_mask = dataframe[column_name].isna()

            missing_count = int(
                missing_mask.sum()
            )

            if missing_count == 0:
                continue

            missing_percentage = (
                missing_count / len(dataframe) * 100
            )

            valid_timestamp_mask = dataframe["timestamp"].notna()

            valid_missing_times = dataframe.loc[
                missing_mask & valid_timestamp_mask, "timestamp"
            ]

            missing_before_injection = int(
                (
                    missing_mask & valid_timestamp_mask & (
                        dataframe["timestamp"] < inject_time
                    )
                ).sum()
            )

            missing_after_injection = int(
                (
                    missing_mask & valid_timestamp_mask & (
                        dataframe["timestamp"] >= inject_time
                    )
                ).sum()
            )

            missing_on_invalid_timestamp = int(
                (
                    missing_mask & ~valid_timestamp_mask
                ).sum()
            )

            first_missing_time = (
                valid_missing_times.min()
                if not valid_missing_times.empty
                else pd.NaT
            )

            last_missing_time = (
                valid_missing_times.max()
                if not valid_missing_times.empty
                else pd.NaT
            )

            records.append({
                "scenario": run_data["scenario"],
                "service": run_data["service"],
                "fault_type": run_data["fault_type"],
                "run_number": run_data["run_number"],
                "data_name": data_name,
                "column_name": column_name,
                "row_count": len(dataframe),
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_percentage
                ),
                "longest_missing_streak": (
                    longest_missing_streak(
                        missing_mask
                    )
                ),
                "missing_before_injection": (
                    missing_before_injection
                ),
                "missing_after_injection": (
                    missing_after_injection
                ),
                "missing_on_invalid_timestamp": (
                    missing_on_invalid_timestamp
                ),
                "first_missing_time": (
                    first_missing_time
                ),
                "last_missing_time": (
                    last_missing_time
                ),
            })
    return records


def audit_all_missing_columns(run_paths):
    #Menjalankan audit missing per column terhadap seluruh 90 run

    records = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(
        run_paths, 
        start=1,
    ):
        run_data = load_validated_run(run_path)
        run_records = audit_missing_by_column(run_data)
        records.extend(run_records)

        print(
            f"[COLUMN AUDIT] {index}/{total_runs} "
            f"{run_data['scenario']}/"
            f"run-{run_data['run_number']}"
        )

    return pd.DataFrame(
        records
    )



#==================================#
# 5. RAW TRACE LOADING 
#    AND SEMANTIC AUDIT
#==================================# 

#5.1 Raw Trace Chunk Loading

def iterate_trace_chunks(traces_path, chunksize=TRACE_CHUNK_SIZE):

    #Berfungsi untuk membaca traces.csv secara bertahap

    traces_path = Path(traces_path)

    if not traces_path.is_file():
        raise FileNotFoundError(
            f"File traces tidak ditemukan: "
            f"{traces_path}"
        )

    string_columns = {
        "traceID": "string",
        "spanID": "string",
        "serviceName": "string",
        "methodName": "string",
        "operationName": "string",
        "parentSpanID": "string",
    }

    trace_reader = pd.read_csv(
        traces_path,
        chunksize=chunksize,
        dtype=string_columns,

    )

    for chunk_number, chunk in enumerate(trace_reader, start=1):
        missing_columns = (
            TRACE_REQUIRED_COLUMNS - set(chunk.columns)
        )

        if missing_columns:
            raise ValueError(
                "Kolom raw trace tidak lengkap. "
                f"Kolom yang hilang: "
                f"{sorted(missing_columns)}"
            )

        chunk["startTimeMillis"] = (
            pd.to_numeric(
                chunk["startTimeMillis"],
                errors="coerce",
            )
        )

        chunk["startTime"] = pd.to_numeric(
            chunk["startTime"],
            errors="coerce",
        )

        chunk["duration"] = pd.to_numeric(
            chunk["duration"],
            errors="coerce",
        )

        chunk["statusCode"] = pd.to_numeric(
            chunk["statusCode"],
            errors="coerce",
        )

        chunk["timestamp"] = pd.to_datetime(
            chunk["startTimeMillis"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

        yield chunk_number, chunk

def count_missing_string(series):
    """
    Menghitung nilai string yang kosong atau NaN.
    """

    missing_mask = (
        series.isna()
        | series.str.strip().eq("")
    )

    return int(
        missing_mask.sum()
    )



# 5.2 Raw Trace Structural Audit
def audit_trace_structure(run_data):
    #mengaudit struktur seluruh raw trace pada satu run tanpa load seluruh file sekaligus

    total_chunk_count = 0
    total_span_count = 0

    unique_trace_ids = set()
    seen_span_ids = set()

    duplicate_span_id_count = 0

    missing_trace_id_count = 0
    missing_span_id_count = 0
    missing_service_count = 0
    missing_operation_count = 0
    missing_status_code_count = 0

    invalid_timestamp_count = 0
    invalid_duration_count = 0
    negative_duration_count = 0

    trace_start_time = None
    trace_end_time = None

    for chunk_number, chunk in (iterate_trace_chunks(run_data["traces_path"])):
        total_chunk_count += 1
        total_span_count += len(chunk)

        #Missing identifier dan metadata

        missing_trace_id_count += (
            count_missing_string(chunk["traceID"])
        )

        missing_span_id_count += (
            count_missing_string(chunk["spanID"])
        )

        missing_service_count += (
            count_missing_string(chunk["serviceName"])
        )

        missing_operation_count += (
            count_missing_string(chunk["operationName"])
        )

        missing_status_code_count += int(
            chunk["statusCode"].isna().sum()
        )

        #Validasi timestamp dan durasi

        invalid_timestamp_count += int(
            chunk["timestamp"].isna().sum()
        )

        invalid_duration_count += int(
            chunk["duration"].isna().sum()
        )

        negative_duration_count += int(
            (
                chunk["duration"] < 0
            ).sum()
        )


        valid_timestamps = (
            chunk["timestamp"].dropna()
        )

        if not valid_timestamps.empty:
            chunk_start = (
                valid_timestamps.min()
            )

            chunk_end = (
                valid_timestamps.max()
            )

            if (
                trace_start_time is None
                or chunk_start < trace_start_time
            ):
                trace_start_time = (
                    chunk_start
                )

            if (
                trace_end_time is None
                or chunk_end > trace_end_time
            ):
                trace_end_time = (
                    chunk_end
                )

        # ---------------------------------
        # Unique trace ID
        # ---------------------------------

        valid_trace_ids = (
            chunk["traceID"]
            .dropna()
            .str.strip()
        )

        valid_trace_ids = valid_trace_ids[
            valid_trace_ids != ""
        ]

        unique_trace_ids.update(
            valid_trace_ids.tolist()
        )

        # ---------------------------------
        # Duplicate span ID
        # ---------------------------------

        valid_span_ids = (
            chunk["spanID"]
            .dropna()
            .str.strip()
        )

        valid_span_ids = valid_span_ids[
            valid_span_ids != ""
        ]

        span_id_list = (
            valid_span_ids.tolist()
        )

        span_id_set = set(
            span_id_list
        )

        # Duplikasi dalam chunk yang sama.
        duplicate_span_id_count += (
            len(span_id_list)
            - len(span_id_set)
        )

        # Duplikasi dengan chunk sebelumnya.
        duplicate_span_id_count += len(
            span_id_set
            & seen_span_ids
        )

        seen_span_ids.update(
            span_id_set
        )

        print(
            f"[TRACE AUDIT] "
            f"{run_data['scenario']}/"
            f"run-{run_data['run_number']} "
            f"chunk-{chunk_number}"
        )

    issues = []

    if total_span_count == 0:
        issues.append(
            "empty_trace"
        )

    if missing_trace_id_count > 0:
        issues.append(
            "missing_trace_id"
        )

    if missing_span_id_count > 0:
        issues.append(
            "missing_span_id"
        )

    if invalid_timestamp_count > 0:
        issues.append(
            "invalid_timestamp"
        )

    if invalid_duration_count > 0:
        issues.append(
            "invalid_duration"
        )

    if negative_duration_count > 0:
        issues.append(
            "negative_duration"
        )

    if duplicate_span_id_count > 0:
        issues.append(
            "duplicate_span_id"
        )

    status = (
        "valid"
        if not issues
        else "|".join(issues)
    )

    return {
        "scenario": run_data["scenario"],
        "service": run_data["service"],
        "fault_type": run_data["fault_type"],
        "run_number": run_data["run_number"],
        "trace_file": str(
            run_data["traces_path"]
        ),
        "chunk_count": total_chunk_count,
        "span_count": total_span_count,
        "unique_trace_count": len(
            unique_trace_ids
        ),
        "unique_span_count": len(
            seen_span_ids
        ),
        "missing_trace_id_count": (
            missing_trace_id_count
        ),
        "missing_span_id_count": (
            missing_span_id_count
        ),
        "missing_service_count": (
            missing_service_count
        ),
        "missing_operation_count": (
            missing_operation_count
        ),
        "missing_status_code_count": (
            missing_status_code_count
        ),
        "invalid_timestamp_count": (
            invalid_timestamp_count
        ),
        "invalid_duration_count": (
            invalid_duration_count
        ),
        "negative_duration_count": (
            negative_duration_count
        ),
        "duplicate_span_id_count": (
            duplicate_span_id_count
        ),
        "trace_start_time": trace_start_time,
        "trace_end_time": trace_end_time,
        "status": status,
    }



def audit_all_trace_structures(run_paths):
    # Menjalankan audit struktur raw trace untuk seluruh run

    records = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(run_paths, start = 1):
        try:
            run_data = load_validated_run(run_path)

            audit_result = audit_trace_structure(run_data)

            records.append(audit_result)

            print(
                f"[OK] Raw Trace audit"
                f"{index}/{total_runs}: "
                f"{run_data['scenario']}/"
                f"run-{run_data['run_number']}"
            )

        except Exception as error:
            records.append(
                {
                    "scenario": run_path.parent.name,
                    "run_number": run_path.name,
                    "trace_file": str(
                        run_path / "traces.csv"
                    ),
                    "status": (f"audit_error: {error}"),
                }
            )

            print(
                f"[ERROR] Raw Trace audit "
                f"{index}/{total_runs}: "
                f"{run_path}: {error}"
            )
    return pd.DataFrame(records)

#5.3 Root Span and Request Identity Audit
def audit_root_span_identity(run_data):

    #Mengaudit jumlah root span pada setiap traceID dalam satu run

    unique_trace_ids = set()

    #Menyimpan root spanID unik untuk setiap traceID
    root_span_ids_by_trace = defaultdict(set)

    root_missing_trace_id_count = 0
    root_missing_span_id_count = 0

    for chunk_number, chunk in iterate_trace_chunks(run_data["traces_path"]):

        trace_ids = (chunk["traceID"].astype("string").str.strip())
        span_ids = (chunk["spanID"].astype("string").str.strip())
        parent_span_ids = (chunk["parentSpanID"].astype("string").str.strip())
        valid_trace_mask = (trace_ids.notna() & trace_ids.ne(""))
        valid_span_mask = (span_ids.notna() & span_ids.ne(""))
        valid_parent_mask = (parent_span_ids.notna() & parent_span_ids.ne(""))

        # Root span tidak mempunyai parent
        root_mask = ~valid_parent_mask

        unique_trace_ids.update(trace_ids.loc[valid_trace_mask].tolist())
        valid_root_mask = (root_mask & valid_trace_mask & valid_span_mask)
        root_trace_ids = trace_ids.loc[valid_root_mask]
        root_span_ids = span_ids.loc[valid_root_mask]

        for trace_id, span_id in zip(
            root_trace_ids,
            root_span_ids,
        ):
            root_span_ids_by_trace[trace_id].add(span_id)

        root_missing_trace_id_count += int(
            (
                root_mask & ~valid_trace_mask
            ).sum()
        )

        root_missing_span_id_count += int(
            (
                root_mask & ~valid_span_mask
            ).sum()
        )

    root_counts_by_trace = {
        trace_id: len(span_id_set)
        for trace_id, span_id_set in root_span_ids_by_trace.items()
    }

    trace_ids_with_root = set(
        root_counts_by_trace
    )

    zero_root_trace_count = len(
        unique_trace_ids - trace_ids_with_root
    )

    single_root_trace_count = sum(
        root_count == 1
        for root_count in root_counts_by_trace.values()
    )

    multiple_root_trace_count = sum(
        root_count > 1
        for root_count
        in root_counts_by_trace.values()
    )

    unique_root_span_count = sum(
        root_counts_by_trace.values()
    )

    categorized_trace_count = (
        zero_root_trace_count + single_root_trace_count + multiple_root_trace_count
    )

    identity_balance_valid = (
        categorized_trace_count
        == len(unique_trace_ids)
    )

    issues = []

    if not unique_trace_ids:
        issues.append(
            "empty_trace_identity"
        )

    if zero_root_trace_count > 0:
        issues.append(
            "zero_root_trace"
        )

    if multiple_root_trace_count > 0:
        issues.append(
            "multiple_root_trace"
        )

    if root_missing_trace_id_count > 0:
        issues.append(
            "root_missing_trace_id"
        )

    if root_missing_span_id_count > 0:
        issues.append(
            "root_missing_span_id"
        )

    if not identity_balance_valid:
        issues.append(
            "identity_balance_mismatch"
        )

    status = (
        "valid"
        if not issues
        else "|".join(issues)
    )

    single_root_ratio = (
        single_root_trace_count
        / len(unique_trace_ids)
        if unique_trace_ids
        else 0.0
    )

    return {
        "scenario": run_data["scenario"],
        "service": run_data["service"],
        "fault_type": run_data["fault_type"],
        "run_number": run_data["run_number"],
        "trace_file": str(
            run_data["traces_path"]
        ),
        "unique_trace_count": len(
            unique_trace_ids
        ),
        "unique_root_span_count": (
            unique_root_span_count
        ),
        "single_root_trace_count": (
            single_root_trace_count
        ),
        "zero_root_trace_count": (
            zero_root_trace_count
        ),
        "multiple_root_trace_count": (
            multiple_root_trace_count
        ),
        "root_missing_trace_id_count": (
            root_missing_trace_id_count
        ),
        "root_missing_span_id_count": (
            root_missing_span_id_count
        ),
        "single_root_ratio": (
            single_root_ratio
        ),
        "identity_balance_valid": (
            identity_balance_valid
        ),
        "status": status,
    }


def audit_all_root_span_identities(run_paths):
    #Menjalankan root span identity untuk seluruh run (90 run)

    records = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(run_paths, start = 1,):
        try:
            run_data = load_validated_run(run_path)
            audit_result = (audit_root_span_identity(run_data))
            records.append(audit_result)

            print(
                f"[OK] Root identity audit "
                f"{index}/{total_runs}: "
                f"{run_data['scenario']}/"
                f"run-{run_data['run_number']}"
            )

        except Exception as error:
            records.append({
                "scenario": (run_path.parent.name),
                "run_number": (run_path.name),
                "trace_file": str(run_path / "traces.csv"),
                "status": (f"audit_error: {error}")
            })

            print(
                f"[ERROR] Root identity audit "
                f"{index}/{total_runs}: "
                f"{run_path}: {error}" 
            )
    return pd.DataFrame(records)            


#5.4 Service and Operation Audit
def audit_root_service_operations(run_data):
    #Menghitung distribusi service Name dan Operation Name pada root span satu run

    service_operation_counts = Counter()
    seen_root_span_ids = set()

    for chunk_number, chunk in iterate_trace_chunks(run_data["traces_path"]):
        trace_ids = (
            chunk["traceID"].astype("string").str.strip()
        )

        span_ids = (
            chunk["spanID"].astype("string").str.strip()
        )

        parent_span_ids = (
            chunk["parentSpanID"].astype("string").str.strip()
        )

        service_names = (
            chunk["serviceName"].astype("string").str.strip()
        )

        operation_names = (
            chunk["operationName"].astype("string").str.strip()
        )

        valid_trace_mask = (
            trace_ids.notna() & trace_ids.ne("")
        )

        valid_span_mask = (
            span_ids.notna() & span_ids.ne("")
        )

        valid_parent_mask = (
            parent_span_ids.notna() & parent_span_ids.ne("")
        )

        root_mask = ( ~valid_parent_mask )
        valid_root_mask = (root_mask & valid_trace_mask & valid_span_mask)

        root_rows = pd.DataFrame({
            "span_id": span_ids.loc[valid_root_mask],
            "service_name": service_names.loc[valid_root_mask],
            "operation_name": operation_names.loc[valid_root_mask],
        })

        for row in root_rows.itertuples(index=False):
            #Jangan hitung duplicate SpanID lebih dari satu kali

            if row.span_id in seen_root_span_ids:
                continue

            seen_root_span_ids.add(
                row.span_id
            )

            service_name = (
                row.service_name 
                if pd.notna(row.service_name) and row.service_name != ""
                else "<missing>"
            )

            operation_name = (
                row.operation_name 
                if pd.notna(row.operation_name) and row.operation_name != ""
                else "<missing>"
            )

            service_operation_counts[service_name, operation_name] += 1

    total_root_count = len(
        seen_root_span_ids
    )

    records = []

    for (service_name, operation_name, ), root_count in sorted(
        service_operation_counts.items(), key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        root_share = (
            root_count / total_root_count
            if total_root_count > 0
            else 0.0
        )

        records.append({
            "scenario": run_data["scenario"],
            "service": run_data["service"],
            "fault_type": run_data["fault_type"],
            "run_number": run_data["run_number"],
            "root_service": service_name,
            "root_operation": operation_name,
            "root_count": root_count,
            "root_share": root_share,
            "total_root_count": (
                total_root_count
            ),
        })

    return pd.DataFrame(
        records,
        columns=[
            "scenario",
            "service",
            "fault_type",
            "run_number",
            "root_service",
            "root_operation",
            "root_count",
            "root_share",
            "total_root_count",
        ],

    )


def audit_all_root_service_operations(run_paths):
    #Menjalankan service operation audit untuk seluruh run

    reports = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(run_paths, start=1):
        try:
            run_data = load_validated_run(run_path)
            run_report = (audit_root_service_operations(run_data))

            if run_report.empty:
                raise ValueError(
                    "Tidak ditemukan root span"
                    "yang dapat diaudit"
                )

            run_report["status"] = "valid"
            run_report["error_message"] = ""

            reports.append(run_report)

            print(
                f"[OK] Service operation audit "
                f"{index}/{total_runs}: "
                f"{run_data['scenario']}/"
                f"run-{run_data['run_number']}"
            )

        except Exception as error:
            scenario = (run_path.parent.name)
            service_name, fault_type = (scenario.rsplit("_", maxsplit=1,))

            error_report = pd.DataFrame([{
                "scenario": scenario,
                "service": service_name,
                "fault_type": fault_type,
                "run_number": run_path.name,
                "root_service": "<audit_error>",
                "root_operation": "<audit_error>",
                "root_count": 0,
                "root_share": 0.0,
                "total_root_count": 0,
                "status": "audit_error",
                "error_message": str(error),
            }])

            reports.append(error_report)

            print(
                f"[ERROR] Service operation audit"
                f"{index}/{total_runs}: "
                f"{run_path}: {error}"
            )

    return pd.concat(reports, ignore_index=True,)

#5.5 Status Code Audit

def audit_span_status_codes(run_data):
    #Mengaudit distribusi status Code pada root dan child span dalam satu run

    status_counts = Counter()
    trace_ids_by_group = defaultdict(set)
    seen_span_ids = set()

    for chunk_number, chunk in iterate_trace_chunks(run_data["traces_path"]):
        trace_ids = (chunk["traceID"].astype("string").str.strip())
        span_ids = (chunk["spanID"].astype("string").str.strip())
        parent_span_ids = (chunk["parentSpanID"].astype("string").str.strip())
        service_names = (chunk["serviceName"].astype("string").str.strip())
        operation_names = (chunk["operationName"].astype("string").str.strip())
        valid_span_mask = (span_ids.notna() & span_ids.ne(""))

        audit_rows = pd.DataFrame({
            "trace_id": trace_ids,
            "span_id": span_ids,
            "parent_span_id": parent_span_ids,
            "service_name": service_names,
            "operation_name": operation_names,
            "status_code": chunk["statusCode"],
        }).loc[valid_span_mask]

        for row in audit_rows.itertuples(index=False):
            if row.span_id in seen_span_ids:
                continue

            seen_span_ids.add(row.span_id)

            trace_id = (
                row.trace_id
                if pd.notna(row.trace_id) and row.trace_id != ""
                else "<missing>"
            )

            service_name = (
                row.service_name
                if pd.notna(row.service_name) and row.service_name != ""
                else "<missing>"
            )

            operation_name = (
                row.operation_name
                if pd.notna(row.operation_name) and row.operation_name != ""
                else "<missing>"
            )

            span_scope = (
                "root"
                if pd.isna(row.parent_span_id)
                or row.parent_span_id == ""
                else "child"
            )

            if pd.isna(row.status_code):
                status_code = "<missing>"
                status_category = "missing"

            elif float(row.status_code) == 0.0:
                status_code = "0"
                status_category = "zero"

            else:
                numeric_status = float(row.status_code)

                status_code = (
                    str(int(numeric_status))
                    if numeric_status.is_integer()
                    else str(numeric_status)
                )

                status_category = "nonzero"

            group_key = (
                span_scope,
                service_name,
                operation_name,
                status_code,
                status_category,
            )

            status_counts[
                group_key
            ] += 1

            if trace_id != "<missing>":
                trace_ids_by_group[
                    group_key
                ].add(trace_id)

    total_unique_span_count = len(seen_span_ids)

    records = []

    for(
        span_scope,
        service_name,
        operation_name,
        status_code,
        status_category,
    ), span_count in sorted(
        status_counts.items(), key=lambda item: (
            item[0][0],
            -item[1],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        span_share = (
            span_count / total_unique_span_count 
            if total_unique_span_count > 0
            else 0.0
        )

        records.append({
            "scenario": run_data["scenario"],
            "service": run_data["service"],
            "fault_type": run_data["fault_type"],
            "run_number": run_data["run_number"],
            "span_scope": span_scope,
            "span_service": service_name,
            "span_operation": operation_name,
            "status_code": status_code,
            "status_category": status_category,
            "span_count": span_count,
            "trace_count": len(
                trace_ids_by_group[
                    (
                        span_scope,
                        service_name,
                        operation_name,
                        status_code,
                        status_category,
                    )
                ]
            ),
            "span_share": span_share,
            "total_unique_span_count": (
                total_unique_span_count
            ),
        })

    return pd.DataFrame(records, columns=[
        "scenario",
        "service",
        "fault_type",
        "run_number",
        "span_scope",
        "span_service",
        "span_operation",
        "status_code",
        "status_category",
        "span_count",
        "trace_count",
        "span_share",
        "total_unique_span_count",
    ],)


def audit_all_span_status_codes(run_paths):
    #Menjalankan status code audit

    reports = []
    total_runs = len(run_paths)

    for index, run_path in enumerate (run_paths, start = 1,):
        try:
            run_data = load_validated_run(run_path)
            run_report = (audit_span_status_codes(run_data))

            if run_report.empty:
                raise ValueError(
                    "Tidak ditemukan span "
                    "yang dapat diaudit"
                )

            run_report["audit_status"] = "valid"
            run_report["error_message"] = ""

            reports.append(run_report)

            print(
                f"[OK] Status code audit "
                f"{index}/{total_runs}: "
                f"{run_data['scenario']}/"
                f"run-{run_data['run_number']}"
            )

        except Exception as error:
            scenario = (run_path.parent.name)
            service_name, fault_type = (scenario.rsplit("_", maxsplit = 1, ))

            error_report = pd.DataFrame([{
                "scenario": scenario,
                "service": service_name,
                "fault_type": fault_type,
                "run_number": run_path.name,
                "span_scope": "<audit_error>",
                "span_service": "<audit_error>",
                "span_operation": "<audit_error>",
                "status_code": "<audit_error>",
                "status_category": "<audit_error>",
                "span_count": 0,
                "trace_count": 0,
                "span_share": 0.0,
                "total_unique_span_count": 0,
                "audit_status": "audit_error",
                "error_message": str(error),
            }])

            reports.append(error_report)

            print(
                f"[ERROR] Status code audit "
                f"{index}/{total_runs}: "
                f"{run_path}: {error}"
            )

    return pd.concat(reports, ignore_index = True, )


#===================================#
# 6. DERIVED SEVERITY PROXY
#    (FCW x SCT)
#===================================#

#6.1  Label Policy Validation
def validate_label_policy(policy, sensitivity):
    #Befungsi untuk memastukan tabel FCW, SCT dan bin severity lengkap dan konsisten sebelum satu label pun dibentuk

    required_key = {
        "policy_version",
        "is_frozen",
        "method",
        "fault_category",
        "fault_category_weight",
        "service_tier",
        "service_tier_weight",
        "severity_bins",
    }

    missing_keys = required_key - set(policy)

    if missing_keys:
        raise ValueError(
            f"Konfigurasi label belum lengkap: {sorted(missing_keys)}"
        )
    
    if not str(policy["policy_version"]).strip():
        raise ValueError("Policy version tidak boleh kosong")

    #Setiap fault di dataset harus punya kategori
    #dan setiap kategori harus punya bobot

    fault_category = policy["fault_category"]
    unmapped_faults = EXPECTED_FAULTS - set(fault_category)

    if unmapped_faults:
        raise ValueError(
            f"Fault belum dipetakan ke kategor: {sorted(unmapped_faults)}"
        )

    fault_weights = policy["fault_category_weight"]
    unweighted_categories = set(fault_category.values()) - set(fault_weights)

    if unweighted_categories:
        raise ValueError(
            f"Kategori fault belum punya bobot: {sorted(unweighted_categories)}"
        )


    tier_weights = policy["service_tier_weight"]
    _validate_tier_scheme(policy["service_tier"], tier_weights, "service_tier")

    #semua bobot harus bilangan bulat positif (boolean ditolak karena subclass int) 
    for name, weights in (
        ("fault_category_weight", fault_weights),
        ("service_tier_weight", tier_weights),
    ): 
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in weights.values()
        ):
            raise ValueError(f"Seluruh {name} harus bilangan bulat positif")


    #bin severity harus menutup setiap skot yang mungkin muncul
    possible_scores = {
        fault_weight * tier_weight
        for fault_weight in fault_weights.values()
        for tier_weight in tier_weights.values()
    }

    score_lookup = _build_score_lookup(policy["severity_bins"])
    uncovered_scores = possible_scores - set(score_lookup)

    if uncovered_scores:
        raise ValueError(
            f"Skor berikut tidak tertcakup severity_bins: {sorted(uncovered_scores)}"
        )

    #scema tier alternatif untuk sensitivity juga harus lengkap
    for scheme_name, scheme in sensitivity.get(
        "alternative_service_tiers", {}
    ).items():
        _validate_tier_scheme(scheme,tier_weights, f"sensitivity[{scheme_name}]")



def _validate_tier_scheme(service_tier, tier_weights, scheme_name):
    #Befungsi untuk memeriksa satu skema tier: semua service terpetakan, semua tier berbobot

    unmapped_services = EXPECTED_SERVICES - set(service_tier)

    if unmapped_services:
        raise ValueError(
            f"{scheme_name}: service belum dipetakan ke tier: "
            f"{sorted(unmapped_services)}"
        )

    unweighted_tiers = set(service_tier.values()) - set(tier_weights)

    if unweighted_tiers:
        raise ValueError(
            f"{scheme_name}: tier belum punya bobot: {sorted(unweighted_tiers)}"
        )


def _build_score_lookup(severity_bins):
    #Mengubah daftar bin menjadi dixt {skor: label}
    #[{"min":1,"max":2,"label":"Low"}, ...] -> {1:"Low", 2:"Low", 3:"Medium", ...}
    #Dict ini akan dipakai .map() untuk menghasilkan pemetaan spt brikut : skor->kelas 0(1) perbaris
    
    lookup = {}
    for bin_spec in severity_bins:
        label = bin_spec["label"]
        if label not in SEVERITY_ORDER:
            raise ValueError(f"Label severity tidak dikenal: {label}")

        for score in range(int(bin_spec["min"]), int(bin_spec["max"]) + 1):
            if score in lookup:
                raise ValueError(f"Skor {score} masuk lebih dari 1 bin / duplikat")
            lookup[score] = label

    return lookup



#6.2 Run Metadata Loading
def load_run_metadata(run_paths):
    #Membentuk satu baris metadata per run dari nama folder
    #Hanya ini yang boleh mnejadi bahan label: scenario, service, fault_type, run_number.


    records = []
    for run_path in run_paths:
        scenario = run_path.parent.name
        service_name, fault_type = scenario.rsplit("_", maxsplit=1)
        records.append({
            "case_id": f"{scenario}-{run_path.name}",
            "scenario": scenario,
            "service": service_name,
            "fault_type": fault_type,
            "run_number": run_path.name,
        }) 

    metadata = pd.DataFrame(records)

    if metadata["case_id"].duplicated().any():
        raise ValueError("Ditemukan case_id duplikat pada metadata run")

    return metadata


#6.3 Label Construction
def build_derived_severity_labels(metadata, policy, service_tier=None):
    #Menghitung FCW, SCT, Severity Score, dan kelas severity untuk setiap run
    #service_tier dapat diganti untuk sensitivity analysis, namum defaultnya memakai policy

    service_tier = policy["service_tier"] if service_tier is None else service_tier
    score_lookup = _build_score_lookup(policy["severity_bins"])

    labels = metadata.copy()

    #FCW: Fault_type -> kategori -> bobot
    labels["fault_category"] = labels["fault_type"].map(policy["fault_category"])
    labels["fcw"] = labels["fault_category"].map(policy["fault_category_weight"])

    #SCT: service -> tier -> bobot
    labels["service_tier"] = labels["service"].map(service_tier)
    labels["sct"] = labels["service_tier"].map(policy["service_tier_weight"])

    #Severity Score = FCW x SCT, lalu dipetakan ke kelas
    labels["severity_score"] = labels["fcw"] * labels["sct"]   
    labels["severity"] = labels["severity_score"].map(score_lookup)

    # .map() menghasilkan NaN untuk nilai yang tidak terpetakan
    # ditangkap di sini supaya tudak lolos dia diam ke tahap berikutnya

    unmapped = labels[["fault_category", "fcw", "service_tier", "sct", "severity"]].isna().any(axis=1)

    if unmapped.any():
        print(
            labels.loc[unmapped, ["case_id", "fault_type", "service"]].to_string(index=False)
        )
        raise ValueError(
            "Terdapat run yang tidak dapat dipetakan ke FCW/SCT"
        )

    labels[["fcw", "sct", "severity_score"]] = (labels[["fcw", "sct", "severity_score"]].astype("int64"))
    labels["label_policy_version"] = policy["policy_version"]
    labels["label_status"] = "final" if policy["is_frozen"] else "candidate"


    return labels[[
        "case_id", "scenario", "service", "fault_type", "run_number",
        "fault_category", "fcw", "service_tier", "sct",
        "severity_score", "severity",
        "label_policy_version", "label_status",
    ]]



#6.4 Distribution Summary
def summarize_label_distribution(labels):
    #Menghitung sebaran kelas: keseluruhan, perservice, dan perfault.
    #Dipakai untuk audit keseimbangan kelas sebelum training
    
    overall = (
        labels["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0).rename_axis("severity").reset_index(name="run_count")
    )


    overall.insert(0, "breakdown", "overall")
    overall.insert(1, "group", "all")


    parts = [overall]
    for group_column in ("service", "fault_type"):
        table = (
            pd.crosstab(labels[group_column], labels["severity"])
            .reindex(columns=SEVERITY_ORDER, fill_value=0)
            .stack()
            .rename("run_count")
            .reset_index()
        )

        table.columns = ["group","severity", "run_count"]
        table.insert(0, "breakdown", group_column)
        parts.append(table)

    return pd.concat(parts, ignore_index =True)


#6.5  Sensitivity Analysis
def build_label_sensitivity_reports(metadata, policy, sensitivity):
    #Menghitung ulang label dengan skema tier alternatif untuk melihat
    #seberapa sensitif distribusi kelas terhadapa keputusan tier

    schemes = {
        "primary": policy["service_tier"],
        **sensitivity.get("alternative_service_tiers", {}),
    }

    detail_parts = []
    summary_records = []

    for scheme_name, service_tier in schemes.items():
        detail = build_derived_severity_labels(metadata, policy, service_tier)
        detail.insert(0, "tier_scheme", scheme_name)
        detail_parts.append(detail)

        counts = detail["severity"].value_counts()
        summary_records.append({
            "tier_scheme": scheme_name,
            "run_count": len(detail),
            "low_count": int(counts.get("Low", 0)),
            "medium_count": int(counts.get("Medium", 0)),
            "high_count": int(counts.get("High", 0)),
            "high_scenario_count": int(detail.loc[detail["severity"].eq("High"), "scenario"].nunique())
        })

    return (pd.concat(detail_parts, ignore_index=True), pd.DataFrame(summary_records),)


#6.6 Label Validation
def validate_derived_severity_labels(labels, metadata, policy):
    #Berfungsi untuk memastikan label lengkap, unik, konsisten dengan rumus, dan identik
    # untuk tiga repetisi dalam satu scenario

    if len(labels) != len(metadata):
        raise RuntimeError("Jumlah label tidak sama dengan jumlah run")

    if labels["case_id"].duplicated().any():
        raise RuntimeError("Ditemukan Case Id duplikat pada derived label")

    if set(labels["case_id"]) != set(metadata["case_id"]):
        raise RuntimeError("Case Id pada label tidak sama dengan metadata run")
    
    if labels.isna().any().any():
        raise RuntimeError("Derived label masih memilki nilai kosong (Missing Values)")

    if not labels["severity_score"].eq(labels["fcw"] * labels["sct"]).all():
        raise RuntimeError("Severity score tidak konsisten dengan FCW * SCT")

    score_lookup = _build_score_lookup(policy["severity_bins"])
    if not labels["severity"].eq(labels["severity_score"].map(score_lookup)).all():
        raise RuntimeError("Kelas severity tidak konsisten dengan severity_bins")

    if not set(labels["severity"]).issubset(SEVERITY_ORDER):
        raise RuntimeError("Ditemukan kelas severity yang tidak dikenal")

    # Label adalah fungsi dari (service, fault_type), sehingga
    # setiap scenario wajib memiliki tepat satu nilai severity

    if not labels.groupby("scenario")["severity"].nunique().eq(1).all():
        raise RuntimeError("Ada scenario dengan severity berbeda antar repetisi")

    if labels["severity"].nunique() < len(SEVERITY_ORDER):
        print("[WARNING!!] Tidak semua kelas severity muncul pada label")

    return True

#6.7 Policy Document
def save_label_policy_document(policy, sensitivity, policy_path, run_count, output_files):
    #Menyimpan seluruh aturan pembentukan label sebagai dokumen JSON
    # agar keputusan label dapat diaudit dan direproduksi

    policy_document = {
        "phase": "6 - Derived Severity Proxy (FCW x SCT)",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "policy": policy,
        "sensitivity_analysis": sensitivity,
        "run_count": int(run_count),
        "label_source": (
            "metadata (service, fault_type) dari nama folder scenario; "
            "tidak ada data observability yang dipakai"
        ),
        "output_files": output_files,
        "method_note": (
            "severity adalah derived proxy dari Fault Category Weight x "
            "Service Criticality Tier, ditetapkan sebelum training, "
            "bukan severity ground truth produksi"
        ),
    }

    with policy_path.open("w", encoding="utf-8") as file:
        json.dump(policy_document, file, ensure_ascii=False, indent=2)



#6.8 Phase 6 Orchestrator 

def run_phase_6():
    # Menjalankan seluruh fase 6 tanpa membaca satupun file telemetri

    print("\n[Pipeline] Memulai Fase 6: Derived Severity Proxy (FCW x SCT)...")

    validate_label_policy(LABEL_POLICY, LABEL_SENSITIVITY)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = load_run_metadata(discover_runs())
    labels = build_derived_severity_labels(metadata, LABEL_POLICY)
    validate_derived_severity_labels(labels, metadata, LABEL_POLICY)

    distribution = summarize_label_distribution(labels)
    sensitivity_detail, sensitivity_summary = build_label_sensitivity_reports(metadata, LABEL_POLICY, LABEL_SENSITIVITY)

    version = str(LABEL_POLICY["policy_version"])
    suffix = "" if LABEL_POLICY["is_frozen"] else f"_{version}"

    label_path = OUTPUT_ROOT / f"derived_severity_labels{suffix}.csv"
    policy_path = OUTPUT_ROOT / f"label_policy{suffix}.json"
    distribution_path = OUTPUT_ROOT / f"derived_severity_distribution{suffix}.csv"
    sensitivity_detail_path = (OUTPUT_ROOT / f"derived_severity_sensitivity_by_run{suffix}.csv")
    sensitivity_summary_path = (OUTPUT_ROOT / f"derived_severity_sensitivity_summary{suffix}.csv")

    labels.to_csv(label_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    sensitivity_detail.to_csv(sensitivity_detail_path, index=False)
    sensitivity_summary.to_csv(sensitivity_summary_path, index=False)

    save_label_policy_document(
        policy=LABEL_POLICY,
        sensitivity=LABEL_SENSITIVITY,
        policy_path=policy_path,
        run_count=len(labels),
        output_files={
            "derived_labels": label_path.name,
            "distribution": distribution_path.name,
            "sensitivity_by_run": sensitivity_detail_path.name,
            "sensitivity_summary": sensitivity_summary_path.name,
        },
    )

    print("\n========== PHASE 6 SUMMARY ==========")
    print("Policy version:", version)
    print("Policy status:", "FINAL" if LABEL_POLICY["is_frozen"] else "CANDIDATE")
    print("Total run berlabel:", len(labels))

    print("\nDistribusi severity:")
    print(labels["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0))

    print("\nSeverity per service:")
    print(pd.crosstab(labels["service"], labels["severity"]).reindex(columns=SEVERITY_ORDER, fill_value=0))

    print("\nSeverity per fault:")
    print(pd.crosstab(labels["fault_type"], labels["severity"]).reindex(columns=SEVERITY_ORDER, fill_value=0))

    print("\nSensitivity skema tier:")
    print(sensitivity_summary.to_string(index=False))

    print("\nDerived label disimpan di:", label_path)
    print("Label policy disimpan di:", policy_path)
    print("[OK] Fase 6 selesai")

    return {
        "labels": labels,
        "metadata": metadata,
        "distribution": distribution,
        "sensitivity_detail": sensitivity_detail,
        "sensitivity_summary": sensitivity_summary,
    }




#===================================#
# 7. OBSERVATION WINDOW EXTRACTION 
#    AND TIMESERIES ALIGNMENT
#===================================#

#7.0 Run eligibility (kelayakan fitur, bukan label)

def _parse_required_boolean(series, column_name):
    #Mengubh TRUE/FALSE dari CSV menjadi boolean asli dan menolak nilai lain

    normalized = series.astype("string").str.strip().str.lower()
    invalid_mask = normalized.isna() | ~normalized.isin(["true", "false"])

    if invalid_mask.any():
        invalid_values = series.loc[invalid_mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"Nilai boolean tidak valid pada {column_name}: {invalid_values}"
        )

    return normalized.eq("true")

def _to_utc(series):
    # Mengubah kolom timestamp hasil CSV (string ISO, tz-aware) menjadi datetime UTC.
    # format="ISO8601" menerima campuran dengan/tanpa fractional second.

    return pd.to_datetime(series, utc=True, format = "ISO8601")


def load_run_eligibility(timeseries_quality_path, trace_structure_path):
    # Menentukan run yang layak masuk eksperimen dari dua laporan audit yang sudah ada:
    #   Fase 4   -> window_covered tiap modality time-series + inject_time
    #   Fase 5.2 -> trace_start_time / trace_end_time raw trace
    # Tidak ada ketergantungan pada label maupun pada Fase 5.6-5.8.


    for path in (timeseries_quality_path, trace_structure_path):
        if not path.exists():
            raise FileNotFoundError(f"Laporan audit belum tersedia: {path}")
        
    key = ["scenario", "run_number"]
    key_dtype = {"scenario": "string", "run_number": "string"}

    # ---- Fase 4: cakupan window per modality ----
    quality = pd.read_csv(timeseries_quality_path, dtype= key_dtype)

    if quality["status"].astype("string").str.startswith("load_error").any():
        raise RuntimeError("Terdapat run yang gagal dimuat pada Fase 4")

    quality["window_covered"] = _parse_required_boolean(
        quality["window_covered"], "window_covered"
    )
    quality["inject_time"] = _to_utc(quality["inject_time"])

    # Daftar modality yang tidak menutup window, per run (kosong jika semua tertutup)
    uncovered = (
        quality.loc[~quality["window_covered"]]
        .groupby(key)["data_name"]
        .agg(lambda names: "|".join(sorted(names)))
        .rename("uncovered_modalities")
    )

    coverage = (
        quality.groupby(key, as_index=False)
        .agg(
            service=("service", "first"),
            fault_type=("fault_type", "first"),
            inject_time=("inject_time", "first"),
            timeseries_window_covered=("window_covered", "all"),
        )
        .merge(uncovered, on=key, how="left")
    )
    coverage["uncovered_modalities"] = coverage["uncovered_modalities"].fillna("")

    # ---- Fase 5.2: rentang waktu raw trace ----
    traces = pd.read_csv(trace_structure_path, dtype=key_dtype)

    if traces["status"].astype("string").str.startswith("audit_error").any():
        raise RuntimeError("Terdapat run yang gagal pada audit raw trace Fase 5.2")

    traces = traces[key + ["trace_start_time", "trace_end_time"]].copy()
    traces["trace_start_time"] = _to_utc(traces["trace_start_time"])
    traces["trace_end_time"] = _to_utc(traces["trace_end_time"])

    expected_run_count = quality[key].drop_duplicates().shape[0]
    coverage = coverage.merge(traces, on=key, how="inner", validate="one_to_one")

    if len(coverage) != expected_run_count or len(coverage) != len(traces):
        raise RuntimeError("Laporan Fase 4 dan Fase 5.2 tidak mencakup run yang sama")

    # ---- Kriteria kelayakan ----
    before = pd.Timedelta(minutes=OBSERVATION_BEFORE_MINUTES)
    after = pd.Timedelta(minutes=OBSERVATION_AFTER_MINUTES)

    coverage["trace_window_covered"] = (
        coverage["trace_start_time"].le(coverage["inject_time"] - before)
        & coverage["trace_end_time"].ge(coverage["inject_time"] + after)
    )

    # Alasan exclusion menjadi satu-satunya sumber kebenaran:
    # run eligible <=> tidak punya alasan exclusion
    def _exclusion_reason(row):
        reasons = []
        if not row.trace_window_covered:
            reasons.append("trace_window_incomplete")
        if REQUIRE_FULL_TIMESERIES_COVERAGE and not row.timeseries_window_covered:
            reasons.append(
                f"timeseries_window_incomplete[{row.uncovered_modalities}]"
            )
        return "|".join(reasons)

    coverage["exclusion_reason"] = coverage.apply(_exclusion_reason, axis=1)
    is_eligible = coverage["exclusion_reason"].eq("")

    eligible_runs = coverage.loc[is_eligible].drop(columns="exclusion_reason").copy()
    excluded_runs = coverage.loc[~is_eligible].copy()

    if eligible_runs.empty:
        raise RuntimeError("Tidak ada run yang memenuhi cakupan observation window")

    return eligible_runs, excluded_runs







#7.1 Build and Validate Alignment Configuration
def build_alignment_config():
    # Berfungsi untuk membentuk konfigurasi alignment dan memvalidasinya satu kali di awal fase 7

    interval_seconds = int(ALIGNMENT_INTERVAL_SECONDS)
    before_seconds = int(OBSERVATION_BEFORE_MINUTES * 60)
    after_seconds = int(OBSERVATION_AFTER_MINUTES * 60)

    if interval_seconds <= 0:
        raise ValueError(
            "Alignment inteval harus lebih beasar dari 0 detik"
        )     

    if before_seconds <= 0:
        raise ValueError(
            "Baseline window harus lebih besar dari 0 detik"
        )

    if before_seconds % interval_seconds != 0:
        raise ValueError(
            "Baseline window tidak dapat dibagi rata dengan alignment interval"
        )

    if after_seconds % interval_seconds != 0:
        raise ValueError(
            "Incident window tidak dapat dibagi rata dengan alignment interval"
        )

    expected_modalities = set(TIMESERIES_FILES)
    configured_modalities = set(ALIGNMENT_SPEC)

    if(expected_modalities != configured_modalities):

        missing_modalities = ( expected_modalities - configured_modalities )
        extra_modalities = (configured_modalities - expected_modalities)

        raise ValueError(
            "ALIGMENT SPEC tidak dikenali"
            f"Missing: {sorted(missing_modalities)}"
            f"Extra: {sorted(extra_modalities)}"
        )

    supported_aggregations = {"mean", "sum",}

    for data_name, specification in (ALIGNMENT_SPEC.items()):

        prefix = str(specification.get("prefix", "",)).strip()
        aggregation = specification.get("aggregation")

        if not prefix:
            raise ValueError(
                f"Prefix kosong untuk {data_name} "
            )

        if (aggregation not in supported_aggregations):
            raise ValueError(
                f"Aggregasi tidak dikenal "
                f"untuk {data_name}: "
                f"{aggregation}"
            )

    baseline_bin_count = (before_seconds // interval_seconds)
    incident_bin_count = (after_seconds // interval_seconds)

    bin_indexes = tuple(
        range(
            -baseline_bin_count,
            incident_bin_count,
        )
    )

    return {
        "interval_seconds": (interval_seconds),
        "before_seconds": (before_seconds),
        "after_seconds": (after_seconds),
        "baseline_bin_count": (baseline_bin_count),
        "incident_bin_count": (incident_bin_count),
        "total_bin_count": len(bin_indexes),
        "bin_indexes": bin_indexes,
    }


#7.2 Observation Grid
def build_observation_grid(inject_time, config):

    #Berfungsi untuk membuat grid waktu 15 detik yang berpusat pada inject_time

    grid = pd.DataFrame({
        "bin_index": list(config["bin_indexes"]),
    })

    grid["window_type"] = [
        (
            "baseline" if bin_index < 0 
            else "incident"
        )
        for bin_index in (
            grid["bin_index"]
        )
    ]

    grid["relative_start_seconds"] = (
        grid["bin_index"] * config["interval_seconds"]
    )

    grid["relative_end_seconds"] = ( 
        grid["relative_start_seconds"] + config["interval_seconds"]
    )

    grid["bin_start"] = (
        inject_time + pd.to_timedelta(grid["relative_start_seconds"], unit = "s",)
    )

    grid["bin_end"] = (
        inject_time + pd.to_timedelta(grid["relative_end_seconds"], unit = "s",)
    )

    return grid.set_index("bin_index")


#7.3 One Modality windowing and Aggregation
def aggregate_modality_to_grid(dataframe, data_name, inject_time, config):

    #Berfungsi untuk memotong satu time series ke observation window lalu mengagregasikannya ke grid 15 detik

    spesification = ALIGNMENT_SPEC[data_name]
    prefix = spesification["prefix"]
    aggregation_method = (spesification["aggregation"])
    window_start = (inject_time - pd.Timedelta(seconds=config["before_seconds"]))
    window_end = (inject_time + pd.Timedelta(seconds=config["after_seconds"]))
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {"time", "timestamp",}
    ]

    if not feature_columns:
        raise ValueError(
            f"{data_name} tidak memiliki kolom fitur "
        )

    if pd.Index(feature_columns).duplicated().any():
        raise ValueError(
            f"{data_name} memiliki nama fitur duplikat "
        )

    window_mask = (dataframe["timestamp"].ge(window_start) & dataframe["timestamp"].lt(window_end))
    window_data = dataframe.loc[window_mask, ["timestamp", *feature_columns,],].copy()

    if window_data.empty:
        raise ValueError(
            f"{data_name} tidak memiliki data di observation window"
        )

    relative_seconds = (window_data["timestamp"] - inject_time).dt.total_seconds()
    window_data["bin_index"] = (relative_seconds // config["interval_seconds"]).astype("int64")
    unknown_bin_mask = (~window_data["bin_index"].isin(config["bin_indexes"]))

    if unknown_bin_mask.any():
        raise RuntimeError(
            f"{data_name} menghasilkan bin_index di luar grid"
        )

    grouped = window_data.groupby("bin_index", sort = True,)

    if aggregation_method == "mean":
        aggregated_values = (
            grouped[feature_columns].mean()
        )

    elif aggregation_method == "sum":
        aggregated_values = (
            grouped[feature_columns].sum(min_count=1)
        )

    else:
        raise ValueError(
            "Aggregation tidak dikukung: "
            f"{aggregation_method}"
        )

    #Mendeteksi bin yang hanya memiliki sebagaian sampel valid
    valid_counts = (grouped[feature_columns].count())
    group_sizes = (grouped.size())
    incomplete_cells = (valid_counts.ne(group_sizes, axis = 0,))


    #Melakukan check terhadap value missing karena jika tidak lengkap, hasil aggregasi tetap NaN untuk fase 8
    aggregated_values = (aggregated_values.mask(incomplete_cells))
    aggregated_values = (
        aggregated_values.rename(
            columns ={
                column: (
                    f"{prefix}__{column}"
                )
                for column in (
                    feature_columns
                )
            }
        )
    )

    sample_count_column = (
        f"quality__{data_name}"
        "_sample_count"
    )


    sample_counts = (group_sizes.rename(sample_count_column))
    observed_bin_count = int(aggregated_values.index.nunique())

    aggregated = (
        aggregated_values
        .join(
            sample_counts,
            how="outer",
        )
        .reindex(
            config["bin_indexes"]
        )
    )

    aggregated.index.name = (
        "bin_index"
    )

    aggregated[sample_count_column] = (
        aggregated[sample_count_column].fillna(0).astype("int64")
    )

    output_feature_columns = [
        column
        for column in (aggregated.columns)
        if column.startswith(
            f"{prefix}__"
        )
    ]

    quality_record = {
        "data_name": data_name,
        "aggregation_method": (aggregation_method),
        "source_row_count": len(window_data),
        "source_feature_count": len(feature_columns),
        "observed_bin_count": observed_bin_count,
        "missing_bin_count": int(aggregated[sample_count_column].eq(0).sum()),
        "source_missing_value_count": int(window_data[feature_columns].isna().sum().sum()),
        "incomplete_aggregated_cell_count": int(incomplete_cells.sum().sum()),
        "aligned_missing_feature_cell_count": int(aggregated[output_feature_columns].isna().sum().sum()),
    }

    return (
        aggregated, quality_record,
    )


#7.4 One Run Alignment
def align_run_timeseries(run_data, config,): 
    #Menyelaraskan 4 modality untuk satu run ke dalam grid yang sama
   
    inject_time = run_data["inject_time"]
    case_id = (
        f"{run_data['scenario']}-"
        f"{run_data['run_number']}"
    )

    aligned = build_observation_grid(inject_time, config,)

    quality_records = []

    for data_name in TIMESERIES_FILES:
        dataframe = run_data["timeseries"][data_name]

        (
            modality_data,
            quality_record,
        ) = aggregate_modality_to_grid(
            dataframe = dataframe,
            data_name= data_name,
            inject_time = inject_time,
            config = config,
        )

        aligned = aligned.join(
            modality_data,
            how="left",
            validate="one_to_one",
        )

        quality_record.update({
            "case_id": case_id,
            "scenario": run_data["scenario"],
            "service": run_data["service"],
            "fault_type": run_data["fault_type"],
            "run_number": run_data["run_number"],
        })

        quality_records.append(quality_record)

    aligned = (aligned.reset_index())
    aligned.insert(0, "case_id", case_id,)
    aligned.insert(1, "scenario", run_data["scenario"],)
    aligned.insert(2, "service", run_data["service"],)
    aligned.insert(3, "fault_type", run_data["fault_type"],)
    aligned.insert(4, "run_number", run_data["run_number"],)
    aligned.insert(5, "inject_time", inject_time,)

    validate_aligned_run(
        aligned = aligned,
        run_data = run_data,
        config = config,
    )

    return (
        aligned, 
        pd.DataFrame(quality_records),
    )


#7.5 One run Validation
def validate_aligned_run(aligned, run_data, config):

    #memastikan hasol satu run mempunyai grid

    expected_case_id = (
        f"{run_data['scenario']}-"
        f"{run_data['run_number']}"
    )

    if (len(aligned) != config["total_bin_count"]):
        raise RuntimeError(
            "Jumlah aligned bin tidak sesuai "
            f"pada {expected_case_id}."
        )

    if aligned["bin_index"].duplicated().any():
        raise RuntimeError(
            "Ditemukan bin_index duplikat "
            f"pada {expected_case_id}."
        )

    if (aligned["bin_index"].tolist() != list(config["bin_indexes"])):
        raise RuntimeError(
            "Urutan bin_index tidak sesuai "
            f"pada {expected_case_id}."
        )

    baseline_count = int(aligned["window_type"].eq("baseline").sum())
    incident_count = int(aligned["window_type"].eq("incident").sum())

    if(baseline_count != config["baseline_bin_count"]):
        raise RuntimeError(
            "Jumlah baseline bin tidak sesuai "
            f"pada {expected_case_id}."
        )

    if(incident_count != config["incident_bin_count"]):
        raise RuntimeError(
            "Jumlah incident bin tidak sesuai "
            f"pada {expected_case_id}."
        )

    if aligned["case_id"].nunique() != 1:
        raise RuntimeError(
            "satu aligned run memiliki lebih dari satu case_id"
        )

    if (aligned["case_id"].iloc[0] != expected_case_id):
        raise RuntimeError(
            "case_id hasil alignment tidak sesuai. "
        )

    if aligned.columns.duplicated().any():
        raise RuntimeError(
            "Ditemukan nama kolom duplikat "
            f"pada {expected_case_id}."
        )

    if aligned[
        ["bin_start", "bin_end"]
    ].isna().any().any():
        raise RuntimeError(
            "Timestamp grid kosong pada "
            f"{expected_case_id}."
        )

    for (data_name, spesification) in ALIGNMENT_SPEC.items():
        prefix = spesification["prefix"]

        feature_columns = [
            column 
            for column in aligned.columns 
            if column.startswith(f"{prefix}__") 
        ]


        if not feature_columns:
            raise RuntimeError(
                f"Fitur {data_name} tidak "
                f"tersedia pada {expected_case_id}."
            )

        non_numeric_columns = [
            column for column in feature_columns 
            if not pd.api.types.is_numeric_dtype(aligned[column])
        ]

        if non_numeric_columns:
            raise RuntimeError(
                "Ditemukan fitur nonnumerik "
                f"pada {expected_case_id}: "
                f"{non_numeric_columns[:10]}"
            )

        sample_count_column = (
            f"quality__{data_name}"
            "_sample_count"
        )

        if (sample_count_column not in aligned.columns):
            raise RuntimeError(
                "Sample count column tidak "
                f"tersedia untuk {data_name}."
            )

        if aligned[sample_count_column].lt(0).any():
            raise RuntimeError(
                "Sample count tidak boleh negatif "
            )

    return True


#7.6 All Eligible run Alignment

def align_all_eligible_runs(eligible_runs, config, ):

    #Memproses 89 run secara bergantian dan menyimpan setiap hasil langsung ke disk

    ALIGNMENT_TIMESERIES_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    quality_reports = []

    total_runs = len(eligible_runs)

    for index, row in enumerate(eligible_runs.itertuples(index=False), start=1, ):

        scenario = str(row.scenario)
        run_number = str(row.run_number)
        case_id = (f"{scenario}-{run_number}")
        run_path = (DATASET_ROOT / scenario / run_number)
        output_directory = (ALIGNMENT_TIMESERIES_ROOT / scenario)
        output_path = (output_directory / f"run_{run_number}.csv")

        try:
            run_data = (
                load_validated_run(run_path)
            )

            (
                aligned,
                run_quality,
            ) = align_run_timeseries(run_data=run_data, config=config,)

            output_directory.mkdir(parents=True, exist_ok=True)
            aligned.to_csv(output_path, index=False)
            quality_reports.append(run_quality)

            feature_columns = [
                column
                for column in (aligned.columns)      
                if any(
                    column.startswith(
                        f"{spesification['prefix']}__"
                    )
                    for spesification in (ALIGNMENT_SPEC.values())
                )
            ]

            manifest_records.append({
                "case_id": case_id,
                "scenario": scenario,
                "service": str(row.service),
                "fault_type": str(row.fault_type),
                "run_number": run_number,
                "aligned_path": str(output_path.relative_to(PROJECT_ROOT)),
                "row_count": len(aligned),
                "feature_column_count": len(feature_columns),
                "baseline_bin_count": int(aligned["window_type"].eq("baseline").sum()),
                "incident_bin_count": int(aligned["window_type"].eq("incident").sum()),
                "missing_feature_cell_count": int(aligned[feature_columns].isna().sum().sum()),
                "status": "valid",
                "error_message": ""
            })

            print(
                f"[OK] Alignment "
                f"{index}/{total_runs}: "
                f"{case_id}"
            )

        except Exception as error:
            manifest_records.append({
                "case_id": case_id,
                "scenario": scenario,
                "service": str(row.service),
                "fault_type": str(row.fault_type),
                "run_number": run_number,
                "aligned_path": "",
                "row_count": 0,
                "feature_column_count": 0,
                "baseline_bin_count": 0,
                "incident_bin_count": 0,
                "missing_feature_cell_count": 0,
                "status": "alignment_error",
                "error_message": str(error)
            })

            print(
                f"[ERROR] Alignment "
                f"{index}/{total_runs}: "
                f"{case_id}: {error}"
            )

    manifest = pd.DataFrame(manifest_records)
    quality_reports = (pd.concat(quality_reports, ignore_index=True,) if quality_reports else pd.DataFrame())

    return (manifest, quality_reports)


#7.7 Final Phase 7 validation
def validate_phase_7_results(manifest, quality_report, eligible_runs, config, ):
    #Memastikan seluruh run eligible berhasil fiproses dan seluruh output tersedia

    expected_run_count= len(eligible_runs)

    if len(manifest) != expected_run_count:
        raise RuntimeError(
            "Jumlah manifest tidak sama dengan jumlah run eligible. "
        )

    error_mask = (manifest["status"].ne("valid"))

    if error_mask.any():
        print(
            manifest.loc[error_mask, ["case_id", "error_message", ],].to_string(index=False)
        )

        raise RuntimeError(
            "Terdapat alignment error "
            "pada fase 7."
        )

    if manifest["case_id"].duplicated().any():
        raise RuntimeError(
            "Ditemukan Case Id yang duplikat pada alignment manifest."
        )

    expected_case_ids = set(eligible_runs["scenario"]+ "-" + eligible_runs["run_number"].astype("string"))

    actual_case_ids = set(manifest["case_id"])

    if (expected_case_ids != actual_case_ids):
        raise RuntimeError(
            "Aligned case tidak sama dengan run eligible."
        )

    if not manifest["row_count"].eq(config["total_bin_count"]).all():
        raise RuntimeError(
            "Tidak semua run memiliki jumlah bin yang benar."
        )

    if not manifest["baseline_bin_count"].eq(config["baseline_bin_count"]).all():
        raise RuntimeError(
            "Baseline bin tidak konsisten. "
        )

    if not manifest["incident_bin_count"].eq(config["incident_bin_count"]).all():
        raise RuntimeError(
            "Incident bin tidak konsisten. "
    )

    expected_quality_count = (expected_run_count * len(ALIGNMENT_SPEC))

    if (len(quality_report) != expected_quality_count):
        raise RuntimeError(
            "Jumlah quality record tidak sesuai"
        )
    
    if quality_report.duplicated(
        ["case_id", "data_name", ]
    ).any():
        raise RuntimeError(
            "Ditemukan Quality record duplikat."
        )

    if set(quality_report["data_name"]) != set(ALIGNMENT_SPEC):
        raise RuntimeError(
            "Daftar modality pada quality report tidak sesuai"
        )

    quality_count_per_case = (
        quality_report.groupby("case_id").size()
    )

    if not quality_count_per_case.eq(len(ALIGNMENT_SPEC)).all():
        raise RuntimeError(
            "Tidak semua case mempunyai empat quality record"
        )

    for path_text in manifest["aligned_path"]:
        output_path = (PROJECT_ROOT / Path(path_text))

        if not output_path.is_file():
            raise FileNotFoundError(
                "Aligned output tidak "
                f"ditemukan: {output_path}"
            )
    
    return True



#7.8 Phase 7 Orchestrator
def run_phase_7():
    #Berfungsi untuk menjalankan seluruh fase 7 menggunakan hasil eligibility di fase 5

    print(
        "\n[Pipeline] Memulai Fase 7: "
        "Observation-window extraction "
        "dan time-series alignment..."
    )

    #Konfigurasi dibentuk dan divalidasi hanya satu kali
    config = (build_alignment_config())

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    eligible_runs, excluded_runs = load_run_eligibility(
        timeseries_quality_path = OUTPUT_ROOT / "timeseries_quality_report.csv",
        trace_structure_path = OUTPUT_ROOT / "raw_trace_structure_report.csv"
    )
    excluded_runs.to_csv(OUTPUT_ROOT / "alignment_excluded_runs.csv", index=False,)

    (
        manifest,
        quality_report,
    ) = align_all_eligible_runs(
        eligible_runs=eligible_runs,
        config=config,
    )
    

    manifest_path = (OUTPUT_ROOT / "aligned_timeseries_manifest.csv")
    quality_report_path = (OUTPUT_ROOT / "alignment_quality_report.csv")
    config_path = (OUTPUT_ROOT / "alignment_config.json")


    #Menyimpan laporan terlebih dahulu agar tetap dapat diperiksa jika validasi akhir menemukan masalah
    manifest.to_csv(
        manifest_path,
        index=False,
    )

    quality_report.to_csv(
        quality_report_path,
        index=False,
    )

    config_document = {
        "generated_at_utc": (pd.Timestamp.now(tz="UTC").isoformat()),
        "interval_seconds": config["interval_seconds"],
        "before_seconds": config["before_seconds"],
        "after_seconds": config["after_seconds"],
        "baseline_bin_count": config["baseline_bin_count"],
        "incident_bin_count": config["incident_bin_count"],
        "total_bin_count": config["total_bin_count"],
        "bin_indexes": list(config["bin_indexes"]),
        "modality_specification": (ALIGNMENT_SPEC),
        "window_rule": {
            "baseline": ("[inject-before, inject)"),
            "incident": ("[inject, inject+after)"),
        },
        "missing_value_policy": ("preserve_for_phase_8"),
    }


    with config_path.open("w", encoding="utf-8",) as file:
        json.dump(
            config_document,
            file,
            ensure_ascii=False,
            indent=2,
        )

    validate_phase_7_results(
        manifest=manifest,
        quality_report=quality_report,
        eligible_runs=eligible_runs,
        config=config,
    )
    
    print(
        "\n========== PHASE 7 SUMMARY =========="
    )

    print(
        "Alignment interval:",
        config["interval_seconds"],
        "detik",
    )

    print(
        "Eligible run:",
        len(eligible_runs),
    )

    print(
        "Excluded run:",
        len(excluded_runs),
    )

    print(
        "Baseline bin per run:",
        config[
            "baseline_bin_count"
        ],
    )

    print(
        "Incident bin per run:",
        config[
            "incident_bin_count"
        ],
    )

    print(
        "Total bin per run:",
        config[
            "total_bin_count"
        ],
    )

    print(
        "Aligned run:",
        manifest["status"]
        .eq("valid")
        .sum(),
    )

    print(
        "\nMissing bin per modality:"
    )

    print(
        quality_report
        .groupby(
            "data_name"
        )["missing_bin_count"]
        .sum()
    )

    print(
        "\nManifest disimpan di:",
        manifest_path,
    )

    print(
        "Quality report disimpan di:",
        quality_report_path,
    )

    print(
        "Alignment config disimpan di:",
        config_path,
    )

    print(
        "Aligned files disimpan di:",
        ALIGNMENT_TIMESERIES_ROOT,
    )

    print(
        "[OK] Fase 7 selesai"
    )

    return {
        "config": config,
        "manifest": manifest,
        "quality_report": (
            quality_report
        ),
        "eligible_runs": eligible_runs,
        "excluded_runs": excluded_runs,
    }

#===================================#
# 8. FEATURE DATA CLEANING 
#    AND RUN ELIGIBILITY
#===================================#

#8.1 Missing Value Classification
def classify_feature_family(column_name):
    # Tugas: menentukan jenis satu kolom dari namanya.
    # Kenapa perlu: nilai kosong harus diisi dengan cara berbeda tergantung jenis kolom.
    #   - Kolom hitungan (jumlah error, jumlah log, aktivitas disk): kosong berarti
    #     tidak ada kejadian, jadi nanti diisi 0.
    #   - Kolom pengukuran (cpu, mem, socket, workload, latency): kosong berarti
    #     alat ukurnya tidak mencatat, jadi nanti diisi dari nilai sebelum/sesudahnya.
    # Cara kerja: nama kolom dipecah di tanda "__" menjadi awalan (metric/log/trace_...)
    # dan sisa nama, lalu dicocokkan dengan akhiran seperti "_cpu", "_error", "_diskio".
    # Hasilnya adalah nama jenis yang menjadi kunci di FILL_POLICY.
    # Jika nama kolom tidak cocok pola mana pun, program sengaja berhenti (error)
    # supaya tidak ada kolom yang diperlakukan salah tanpa disadari.

    prefix, name = column_name.split("__", 1)
    lowered = name.lower()

    if prefix == "trace_latency" :
        return "trace_latency"
    if prefix == "trace_error" :
        return "trace_error"
    if prefix == "log" :
        return "log_count"
    if prefix == "metric":
        if lowered.endswith(GAUGE_METRIC_SUFFIXES):
            return "continuous_resource"
        if lowered.endswith("_diskio"):
            return "disk_activity"
        if lowered.endswith("_error"):
            return "error_signal"
        if lowered.endswith("_workload"):
            return "workload_signal"
        if "latency" in lowered :
            return "latency_signal"
        
    raise ValueError(f"Kolom tidak dapat diklasifikasi: {column_name}")

def classify_log_template(template):
    # Tugas: menentukan kategori satu kalimat log (template) berdasarkan kata kuncinya.
    # Tiga kategori yang mungkin:
    #   - "lifecycle": pesan saat service mulai/restart (mengandung "initializing", dll.)
    #   - "error"    : pesan kegagalan (mengandung "error", "fail", "unavailable", dll.)
    #   - "routine"  : pesan operasi normal (tidak mengandung kata kunci mana pun)
    # Daftar kata kunci ada di LOG_TEMPLATE_CATEGORIES dan diperiksa berurutan:
    # "lifecycle" dicek lebih dulu supaya pesan startup yang kebetulan memuat kata
    # "unavailable" tidak salah masuk ke "error".

    for category, pattern in LOG_TEMPLATE_CATEGORIES:
        if pattern.search(template):
            return category
    return "routine"
    

#8.2 Column Harmonization
def load_cluster_info(scenario, run_number):
    # Tugas: membuka file cluster_info.json milik satu run dan mengembalikannya sebagai dict.
    # Isi file: pemetaan nomor cluster -> kalimat log,
    # contoh {"2": {"template": "frontend request started", "container": ["frontend"]}}.
    # Harus dibaca per run, karena nomor yang sama (misalnya "2") berarti kalimat yang
    # berbeda di run yang berbeda.


    path = DATASET_ROOT / scenario / str(run_number) / "cluster_info.json"
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _split_log_column(column):
    # Tugas: memecah nama kolom log menjadi nama service dan nomor cluster.
    # Contoh: "log__frontend_2" -> ("frontend", "2").
    # Nomor cluster inilah yang nanti dicari di cluster_info.json.

    service_name, cluster_id = column[len("log__"):].rsplit("_", 1)
    return service_name, cluster_id

def harmonize_log_columns(aligned, cluster_info, case_id):
    # Tugas: mengganti nama kolom log yang memakai nomor (arti nomor berbeda tiap run)
    # menjadi nama yang memakai kategori (arti sama di semua run).
    # Contoh: "log__frontend_2" (= "frontend request started") -> "log__frontend__routine".
    # Langkah:
    #   1. Untuk tiap kolom log: pecah nama -> cari nomor di cluster_info -> dapat kalimat
    #      -> tentukan kategori -> susun nama baru "log__<service>__<kategori>".
    #   2. Kolom-kolom yang mendapat nama baru yang sama dijumlahkan per baris (per bin),
    #      sehingga hasilnya satu kolom per pasangan (service, kategori).
    #   3. Setiap terjemahan dicatat ke map_records untuk laporan column_harmonization_map.csv,
    #      termasuk jumlah kemunculan kalimat itu di window baseline dan incident.


    groups = defaultdict(list)
    map_records = []
    basline_mask = aligned["window_type"].eq("baseline")
    incident_mask = aligned["window_type"].eq("incident")

    for column in aligned.columns:
        if not column.startswith("log__"):
            continue

        service_name, cluster_id = _split_log_column(column)
        info = cluster_info.get(cluster_id)
        if info is None:
            raise KeyError(f"{case_id}: cluster_id {cluster_id} tidak ditemukan di cluster info json")

        template = info["template"]
        category = classify_log_template(template)
        harmonized = f"log__{service_name}__{category}"
        groups[harmonized].append(column)


        map_records.append({
            "case_id": case_id,
            "source_column": column,
            "service": service_name,
            "cluster_id": cluster_id,
            "template": template,
            "category": category,
            "harmonized_column": harmonized,
            "baseline_sum": float(aligned.loc[basline_mask, column].sum()),
            "incident_sum": float(aligned.loc[incident_mask, column].sum()),
        })

    harmonized_frame = pd.DataFrame(index=aligned.index)
    for harmonized, sources in groups.items():
        # min_count=1 artinya: jika semua kolom sumber pada satu bin kosong (NaN),
        # hasil penjumlahan tetap NaN, bukan 0. Dengan begitu bin yang benar-benar
        # tidak punya data tetap terlihat sebagai kosong dan tercatat di cleaning log.
        harmonized_frame[harmonized] = aligned[sources].sum(axis= 1, min_count = 1)

    return harmonized_frame, map_records


def build_feature_schema(manifest):
    # Tugas: menyusun daftar kolom final yang akan dipakai SEMUA run, sebelum ada satu
    # run pun yang diproses. Daftar ini disebut "schema".
    # Langkah:
    #   1. Baca hanya baris judul (header) dari setiap file aligned, tidak membaca isinya.
    #   2. Nama kolom log diterjemahkan dulu ke nama kategori (lihat harmonize_log_columns),
    #      supaya yang dihitung adalah nama final, bukan nama bernomor.
    #   3. Hitung tiap kolom muncul di berapa run (run_coverage = jumlah run / total run).
    #   4. Tentukan jenis kolom (classify_feature_family) dan kebijakan pengisiannya.
    #   5. Kolom pengukuran yang muncul di kurang dari MIN_GAUGE_COLUMN_COVERAGE run
    #      dibuang, karena tidak ada cara jujur mengisi run yang tidak memilikinya.
    #      Kolom hitungan tidak pernah dibuang: run yang tidak memilikinya cukup diisi 0.
    #   6. Urutkan kolom secara tetap (metric, log, trace_latency, trace_error, lalu abjad)
    #      supaya urutan kolom identik di semua file hasil.

    total_runs = len(manifest)
    coverage = Counter()

    for row in manifest.itertuples(index=False):
        columns = pd.read_csv(PROJECT_ROOT / Path(row.aligned_path), nrows=0).columns
        cluster_info = load_cluster_info(row.scenario, row.run_number)  
        run_columns = set()

        for column in columns:
            if column.startswith("log__"):
                service_name, cluster_id = _split_log_column(column)
                category = classify_log_template(cluster_info[cluster_id]["template"])
                run_columns.add(f"log__{service_name}__{category}")
            elif "__" in column and not column.startswith("quality__"):
                run_columns.add(column)

        coverage.update(run_columns)

    records = []
    for column, count in coverage.items():
        family = classify_feature_family(column)
        fill_policy = FILL_POLICY[family]
        run_coverage = count / total_runs
        keep = True
        drop_reason = ""

        # Hanya kolom pengukuran (kebijakan "interpolate") yang bisa dibuang.
        # Contoh yang terbuang: metric__istio-init_mem, hanya ada di 1 dari 89 run.
        if fill_policy == "interpolate" and run_coverage < MIN_GAUGE_COLUMN_COVERAGE:
            keep = False
            drop_reason = "gauge_coverage_below_threshold"

        records.append({
            "column": column,
            "modality": column.split("__", 1)[0],
            "family": family,
            "fill_policy": fill_policy,
            "run_coverage": run_coverage,
            "keep": keep,
            "drop_reason": drop_reason,
        })

    schema = pd.DataFrame(records)
    schema["modality_rank"] = schema["modality"].map(
        {name: rank for rank, name in enumerate(MODALITY_ORDER)}
    )
    schema = (
        schema.sort_values(["modality_rank", "column"])
        .drop(columns="modality_rank")
        .reset_index(drop=True)
    )

    return schema


#8.3 Run Eligibility


def assess_run_eligibility(aligned, case_id):
    # Tugas: memutuskan apakah data satu run masih cukup utuh untuk dijadikan fitur.
    # Dasar penilaian: kolom quality__<sumber>_sample_count dari Fase 7. Nilai 0 pada
    # satu bin berarti bin itu sama sekali tidak punya data asli (bukan sekadar satu sel kosong).
    # Langkah: untuk tiap sumber data (metrics, logs, trace_latency, trace_error) dan tiap
    # window (baseline, incident), hitung berapa bin yang bernilai 0 dari 20 bin.
    # Jika rasionya melebihi MAX_MISSING_BIN_RATIO (0.10 = lebih dari 2 bin), run dinyatakan
    # tidak layak dan alasannya ditulis. Angka rasio tiap sumber tetap disimpan untuk laporan.

    record = {"case_id": case_id}
    reasons = []


    for data_name in TIMESERIES_FILES:
        sample_column = f"quality__{data_name}_sample_count"
        for window in ("baseline", "incident"):
            mask = aligned["window_type"].eq(window)
            missing =  int(aligned.loc[mask, sample_column].eq(0).sum())
            ratio = missing / int(mask.sum())
            record[f"{data_name}_{window}_missing_bins"] = missing
            record[f"{data_name}_{window}_missing_ratio"] = round(ratio,4)

            if ratio > MAX_MISSING_BIN_RATIO:
                reasons.append(f"{data_name}_{window}_missing_ratio={ratio:.2f}")

    record["eligible"] = not reasons
    record["exclusion_reason"] = "|".join(reasons)
    return record


#8.4 Data Cleaning
def fill_missing_values(series, fill_policy):
    # Tugas: mengisi nilai kosong (NaN) pada satu kolom di satu run.
    # Ada dua kebijakan, ditentukan oleh jenis kolom (lihat FILL_POLICY):
    #   "zero"        -> semua kosong diganti 0 (dipakai untuk kolom hitungan).
    #   "interpolate" -> kosong ditebak dari nilai tetangganya (dipakai untuk kolom pengukuran).
    # Fungsi mengembalikan kolom yang sudah diisi dan nama metode yang dipakai (untuk log).

    if fill_policy == "zero":
        return series.fillna(0.0), "zero_fill"


    # Kebijakan "interpolate", tiga tahap:
    #   1. Kosong yang diapit dua nilai diisi garis lurus di antaranya
    #      (contoh: 0.40, kosong, 0.50 -> kosong menjadi 0.45). limit_area="inside"
    #      memastikan hanya kosong yang diapit yang diisi.
    #   2. Kosong di tepi awal/akhir window disalin dari nilai terdekat, maksimal
    #      MAX_EDGE_FILL_BINS bin (ffill = dari sebelumnya, bfill = dari sesudahnya).
    #   3. Kosong yang masih tersisa dibiarkan NaN. Tidak dipaksa diisi, karena tidak ada
    #      dasar yang jujur; jumlahnya akan tercatat sebagai missing_after di cleaning log.

    filled = series.interpolate(method = "linear", limit_area= "inside")
    filled = filled.ffill(limit=MAX_EDGE_FILL_BINS).bfill(limit=MAX_EDGE_FILL_BINS)
    return filled, "linear_interpolate_edge_fill"


def clean_run(aligned, schema, cluster_info, case_id):
    # Tugas: membentuk satu file bersih untuk satu run, dengan kolom persis sesuai schema.
    # Langkah:
    #   1. Samakan nama kolom log (harmonize_log_columns), lalu gabungkan dengan kolom asli
    #      menjadi tabel "source" yang memuat semua kolom yang mungkin ada.
    #   2. Untuk setiap kolom di schema (urutan tetap):
    #      - jika kolom itu ada di source, ambil nilainya;
    #      - jika tidak ada di run ini, buat kolom berisi NaN seluruhnya;
    #      - hitung jumlah kosong, isi sesuai kebijakan (fill_missing_values), hitung lagi;
    #      - jika ada yang diisi atau kolomnya tadinya tidak ada, catat ke cleaning_records.
    #   3. Tambahkan kolom identitas (case_id, bin_index, dll.) dan kolom quality__ apa adanya.
    #   4. Gabungkan semua kolom menjadi satu DataFrame sekaligus.

    harmonized_logs, map_records = harmonize_log_columns(aligned, cluster_info, case_id)
    source = pd.concat([aligned, harmonized_logs], axis=1)


    # Kolom hasil dikumpulkan dulu ke dalam dict, lalu digabung sekali dengan pd.concat.
    # Alasan: menambahkan kolom satu per satu ke DataFrame membuat pandas lambat dan
    # memunculkan peringatan "DataFrame is highly fragmented".
    cleaned_columns = {}
    cleaning_records = []


    for row in schema.loc[schema["keep"]].itertuples(index=False):
        column_absent = row.column not in source.columns
        series = (
            pd.Series(float("nan"), index = aligned.index, dtype="float64")
            if column_absent
            else source[row.column].astype("float64")
        )

        missing_before = int(series.isna().sum())
        if missing_before: 
            filled, method = fill_missing_values(series, row.fill_policy)
        else:
            filled, method = series, "none"
        missing_after = int(filled.isna().sum())

        cleaned_columns[row.column] = filled

        if missing_before or column_absent:
            cleaning_records.append({
                "case_id": case_id,
                "column": row.column,
                "modality": row.modality,
                "family": row.family,
                "fill_policy": row.fill_policy,
                "column_absent_in_run": column_absent,
                "missing_before": missing_before,
                "method": method,
                "filled_count": missing_before - missing_after,
                "missing_after": missing_after
            })
    
    for data_name in TIMESERIES_FILES:
        sample_column = f"quality__{data_name}_sample_count"
        cleaned_columns[sample_column] = aligned[sample_column]


    cleaned = pd.concat(
        [aligned[ALIGNED_META_COLUMNS], pd.DataFrame(cleaned_columns, index=aligned.index)],
        axis = 1,
    )
    return cleaned, cleaning_records, map_records


#8.5 Cleaned Run Validation
def validate_cleaned_run(cleaned, schema, case_id):
    # Tugas: memeriksa satu file bersih sebelum disimpan. Empat pemeriksaan:
    #   1. Daftar dan urutan kolom harus persis: kolom identitas + kolom schema + kolom quality.
    #   2. Harus tepat 40 baris (20 baseline + 20 incident) tanpa bin_index ganda.
    #   3. Kolom berkebijakan "zero" tidak boleh masih memuat NaN.
    #   4. Semua kolom fitur harus bertipe angka.
    # Jika satu pemeriksaan gagal, program berhenti dan menyebut case_id yang bermasalah,
    # supaya file yang bentuknya salah tidak pernah tersimpan.

    expected_columns = (
        ALIGNED_META_COLUMNS + schema.loc[schema["keep"]]["column"].tolist()
        + [f"quality__{data_name}_sample_count" for data_name in TIMESERIES_FILES]
    )

    if cleaned.columns.tolist() != expected_columns:
        raise RuntimeError(f"{case_id}: urutan / daftar kolom tidak sesuai dengan schema")

    if len(cleaned) != 40 or cleaned["bin_index"].duplicated().any():
        raise RuntimeError(f"{case_id}: jumlah bin tidak 40 atau ada duplikat")

    zero_policy_columns = schema.loc[
        schema["keep"] & schema["fill_policy"].eq("zero"), "column"
    ]

    if cleaned[zero_policy_columns].isna().any().any():
        raise RuntimeError(f"{case_id}: masih ada NaN pada kolom berkebijakan zero")


    feature_columns = schema.loc[schema["keep"], "column"]
    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(cleaned[column])
    ]

    if non_numeric:
        raise RuntimeError(f"{case_id}: kolom non numerik {non_numeric[:5]}")

    return True

#8.6 Phase 8 Orchestrator
def run_phase_8():
    # Tugas: menjalankan seluruh Fase 8 secara berurutan.
    #   1. Baca daftar file aligned dari manifest Fase 7.
    #   2. Susun schema (daftar kolom final) sekali untuk semua run.
    #   3. Untuk tiap run: nilai kelayakan -> jika layak, bersihkan -> periksa -> simpan.
    #   4. Tulis laporan: schema, kelayakan, cleaning log, peta kolom log, katalog template,
    #      manifest file bersih, dan konfigurasi yang dipakai.

    print("\n[Pipeline] Memulai Fase 8: Feature cleaning dan run eligibility...")

    manifest_path = OUTPUT_ROOT / "aligned_timeseries_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest Fase 7 belum tersedia: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype={"run_number": "string"})
    if manifest["status"].ne("valid").any():
        raise RuntimeError("Manifest Fase 7 memuat run yang tidak valid")

    CLEANED_TIMESERIES_ROOT.mkdir(parents=True, exist_ok=True)

    schema = build_feature_schema(manifest)
    print(
        "[OK] Schema:", int(schema["keep"].sum()), "kolom dipertahankan,",
        int((~schema["keep"]).sum()), "kolom dibuang",
    )

    eligibility_records = []
    cleaning_records = []
    map_records = []
    cleaned_manifest_records = []
    total_runs = len(manifest)

    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        aligned = pd.read_csv(
            PROJECT_ROOT / Path(row.aligned_path),
            dtype={"run_number": "string"},
        )
        eligibility = assess_run_eligibility(aligned, row.case_id)
        eligibility_records.append(eligibility)

        if not eligibility["eligible"]:
            print(f"[SKIP] {index}/{total_runs} {row.case_id}: {eligibility['exclusion_reason']}")
            continue

        cluster_info = load_cluster_info(row.scenario, row.run_number)
        cleaned, run_cleaning, run_map = clean_run(aligned, schema, cluster_info, row.case_id)
        validate_cleaned_run(cleaned, schema, row.case_id)

        output_directory = CLEANED_TIMESERIES_ROOT / row.scenario
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"run_{row.run_number}.csv"
        cleaned.to_csv(output_path, index=False)

        cleaning_records.extend(run_cleaning)
        map_records.extend(run_map)
        feature_columns = schema.loc[schema["keep"], "column"]
        cleaned_manifest_records.append({
            "case_id": row.case_id,
            "scenario": row.scenario,
            "service": row.service,
            "fault_type": row.fault_type,
            "run_number": row.run_number,
            "cleaned_path": str(output_path.relative_to(PROJECT_ROOT)),
            "row_count": len(cleaned),
            "feature_column_count": len(feature_columns),
            "unfilled_cell_count": int(cleaned[feature_columns].isna().sum().sum()),
        })
        print(f"[OK] Cleaning {index}/{total_runs}: {row.case_id}")

    eligibility = pd.DataFrame(eligibility_records)
    cleaning_log = pd.DataFrame(cleaning_records)
    harmonization_map = pd.DataFrame(map_records)
    cleaned_manifest = pd.DataFrame(cleaned_manifest_records)

    template_catalog = (
        harmonization_map.groupby(["template", "category"])["case_id"]
        .nunique()
        .rename("run_count")
        .reset_index()
        .sort_values(["category", "run_count"], ascending=[True, False])
    )

    schema.to_csv(OUTPUT_ROOT / "feature_column_schema.csv", index=False)
    eligibility.to_csv(OUTPUT_ROOT / "feature_eligibility.csv", index=False)
    cleaning_log.to_csv(OUTPUT_ROOT / "cleaning_log.csv", index=False)
    harmonization_map.to_csv(OUTPUT_ROOT / "column_harmonization_map.csv", index=False)
    template_catalog.to_csv(OUTPUT_ROOT / "log_template_catalog.csv", index=False)
    cleaned_manifest.to_csv(OUTPUT_ROOT / "cleaned_timeseries_manifest.csv", index=False)

    with (OUTPUT_ROOT / "cleaning_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "max_missing_bin_ratio": MAX_MISSING_BIN_RATIO,
            "max_edge_fill_bins": MAX_EDGE_FILL_BINS,
            "min_gauge_column_coverage": MIN_GAUGE_COLUMN_COVERAGE,
            "gauge_metric_suffixes": list(GAUGE_METRIC_SUFFIXES),
            "fill_policy": FILL_POLICY,
            "log_template_categories": {
                name: pattern.pattern for name, pattern in LOG_TEMPLATE_CATEGORIES
            },
            "input_run_count": int(total_runs),
            "eligible_run_count": int(eligibility["eligible"].sum()),
            "kept_column_count": int(schema["keep"].sum()),
            "dropped_columns": schema.loc[~schema["keep"], "column"].tolist(),
        }, file, ensure_ascii=False, indent=2)

    if len(cleaned_manifest) != int(eligibility["eligible"].sum()):
        raise RuntimeError("Jumlah run bersih tidak sama dengan jumlah run eligible")

    print("\n========== PHASE 8 SUMMARY ==========")
    print("Run masuk        :", total_runs)
    print("Run eligible     :", int(eligibility["eligible"].sum()))
    print("Run dikeluarkan  :", int((~eligibility["eligible"]).sum()))
    print("Kolom fitur final:", int(schema["keep"].sum()))
    print("\nKolom per modality:")
    print(schema.loc[schema["keep"]].groupby("modality").size().reindex(MODALITY_ORDER))
    print("\nKolom dibuang:")
    print(schema.loc[~schema["keep"], ["column", "run_coverage", "drop_reason"]].to_string(index=False))
    print("\nPengisian per kebijakan (jumlah sel):")
    if not cleaning_log.empty:
        print(cleaning_log.groupby(["fill_policy", "method"])["filled_count"].sum())
        print("\nSel yang tetap kosong setelah cleaning:", int(cleaning_log["missing_after"].sum()))
    print("\nKategori template log:")
    print(template_catalog.groupby("category")["template"].nunique())
    print("\nCleaned files disimpan di:", CLEANED_TIMESERIES_ROOT)
    print("[OK] Fase 8 selesai")

    return {
        "schema": schema,
        "eligibility": eligibility,
        "cleaning_log": cleaning_log,
        "harmonization_map": harmonization_map,
        "cleaned_manifest": cleaned_manifest,
    }



#===================================#
# 9. FEATURE EXTRACTION
#===================================#

#9.1 Ringkasan Statistik Satu Run
def _cap_constant_baseline(delta, constant_baseline, raw_zscore):
    # Tugas: menangani kolom yang nilai baseline-nya konstan (sebaran = 0).
    # Pada kolom seperti itu z-score tidak bisa dihitung karena pembaginya nol.
    # Aturan penggantinya:
    #   - selisih NaN (kolom kosong seluruhnya) -> tetap NaN
    #   - selisih 0   (nilai tidak berubah)     -> 0
    #   - selisih > 0 (naik dari konstan)       -> +ZSCORE_CAP
    #   - selisih < 0 (turun dari konstan)      -> -ZSCORE_CAP
    # Terakhir seluruh nilai dipotong ke rentang +-ZSCORE_CAP.

    replacement = delta.map(
        lambda value: (
            float("nan") if pd.isna(value)
            else 0.0 if value == 0
            else ZSCORE_CAP if value > 0
            else -ZSCORE_CAP
        )
    )
    return raw_zscore.where(~constant_baseline, replacement).clip(-ZSCORE_CAP, ZSCORE_CAP)



def summarize_run(cleaned, feature_columns):
    # Tugas: meringkas 40 baris satu run menjadi 5 angka untuk setiap kolom sumber.
    # Input : cleaned = tabel 40 baris hasil Fase 8, feature_columns = daftar kolom sumber.
    # Output: dict {nama_fitur: angka}, contoh
    #         {"metric__checkoutservice_cpu__baseline_mean": 0.30, ...}
    # Seluruh perhitungan dilakukan untuk semua kolom sekaligus: setiap variabel
    # (baseline_mean, incident_max, dan seterusnya) berisi satu angka per kolom,
    # sehingga tidak perlu perulangan per kolom.
    # Nilai kosong diabaikan saat menghitung; kolom yang kosong seluruhnya
    # menghasilkan NaN dan jumlahnya dilaporkan oleh validasi.

    baseline = cleaned.loc[cleaned["window_type"].eq("baseline"), feature_columns]
    incident = cleaned.loc[cleaned["window_type"].eq("incident"), feature_columns]

    baseline_mean = baseline.mean()
    baseline_std = baseline.std(ddof = 0)
    incident_max = incident.max()
    delta_mean = incident.mean() - baseline_mean

    
    # safe_std: sebaran 0 diubah menjadi NaN supaya pembagian tidak menghasilkan
    # nilai tak hingga; hasil NaN itu kemudian diganti oleh _cap_constant_baseline.
    safe_std = baseline_std.replace(0, float("nan"))
    constant_baseline = baseline_std.eq(0)
    peak_delta = incident_max - baseline_mean


    zscore_max = _cap_constant_baseline(peak_delta, constant_baseline, peak_delta / safe_std)
    zscore_mean = _cap_constant_baseline(delta_mean, constant_baseline, delta_mean / safe_std)

    statistics = {
        "baseline_mean": baseline_mean,
        "incident_max": incident_max,
        "delta_mean": delta_mean,
        "zscore_max": zscore_max,
        "zscore_mean": zscore_mean,
    }

    row = {}
    for column in feature_columns:
        for statistic_name in FEATURE_STATISTICS:
            row[f"{column}__{statistic_name}"] = float(statistics[statistic_name][column])
    return row



#9.2 Ekstraksi fitur satu run
def extract_run_features(cleaned_path, feature_columns):
    # Tugas: membaca satu file bersih, memeriksa bentuknya, lalu mengembalikan
    # satu baris fitur lengkap dengan case_id.
    # Pemeriksaan di awal mencegah file rusak atau schema yang tidak cocok
    # lolos diam-diam ke tabel fitur.

    cleaned = pd.read_csv(cleaned_path, dtype={"run_number": "string"})

    if len(cleaned) != 40:
        raise RuntimeError(f"{cleaned_path}: jumlah bin bukan 40")

    missing_columns = set(feature_columns) -  set(cleaned.columns)
    if missing_columns:
        raise RuntimeError(
            f"{cleaned_path}: kolom sumber tidak ditemukan {sorted(missing_columns)[:5]}"
        )

    row = {"case_id": cleaned["case_id"].iloc[0]}
    row.update(summarize_run(cleaned, feature_columns))
    return row


#9.3 Fitur kalimat log baru
def build_log_novelty_features(harmonization_map, case_ids):
    # Tugas: menghitung, per run dan per service, berapa kalimat log yang muncul
    # saat incident tetapi tidak pernah muncul sebelum injeksi.
    # Informasi ini tidak bisa diperoleh dari lima ringkasan biasa, karena lima
    # ringkasan itu menghitung jumlah kemunculan, bukan kemunculan kalimat baru.
    # Sumber data: column_harmonization_map.csv dari Fase 8, yang sudah mencatat
    # baseline_sum dan incident_sum untuk setiap kalimat di setiap run.
    # Langkah:
    #   1. Tandai kalimat "baru": baseline_sum = 0 dan incident_sum > 0.
    #   2. Hitung jumlahnya per (run, service), lalu ubah menjadi tabel
    #      baris = run, kolom = service.
    #   3. Pastikan seluruh run ada; run tanpa kalimat baru diisi 0.
    #   4. Tambahkan satu kolom total seluruh service.

    novelty = harmonization_map.copy()
    novelty["baseline_sum"] = novelty["baseline_sum"].fillna(0)
    novelty["incident_sum"] = novelty["incident_sum"].fillna(0)
    novelty["is_new"] = novelty["baseline_sum"].eq(0) & novelty["incident_sum"].gt(0)

    per_service = (
        novelty.groupby(["case_id", "service"])["is_new"].sum()
        .unstack("service", fill_value=0)
        .reindex(case_ids, fill_value=0)
    )

    per_service.columns = [
        f"log__{service}__new_template_count" for service in per_service.columns
    ]
    per_service["log__all__new_template_count"] = per_service.sum(axis=1)

    return per_service.astype("int64").reset_index()



#9.4 Pembuangan Fitur Konstan
def drop_constant_features(features):
    # Tugas: membuang kolom fitur yang nilainya sama persis di seluruh 89 run.
    # Kolom seperti itu tidak mungkin dipakai model untuk membedakan apa pun,
    # tetapi ikut memecah atribusi SHAP dan menggelembungkan jumlah fitur.
    # Syarat dibuang dibuat ketat: tidak boleh ada satu pun nilai kosong.
    # Kolom yang punya nilai kosong di sebagian run TIDAK dibuang, karena
    # pola "ada / tidak ada" itu sendiri dapat menjadi informasi bagi model.
    # Kolom yang hampir konstan (misalnya bernilai sama di 86 dari 89 run) juga
    # tidak dibuang, karena tiga run yang berbeda itu bisa jadi justru sinyalnya.

    feature_columns = [c for c in features.columns if c != "case_id"]
    constant = [
        column for column in feature_columns
        if features[column].notna().all() and features[column].nunique() <= 1
    ]
    return features.drop(columns=constant), constant


#9.5 Service Context Features (E3)

#9.5a Call graph dari traces.csv
def _window_bounds(inject_epoch):
    # Tugas: mengubah waktu injeksi (epoch detik) menjadi tiga batas waktu:
    # awal baseline, waktu injeksi, akhir incident. Sama seperti Fase 7.

    inject_time = pd.to_datetime(inject_epoch, unit="s", utc=True)
    return (
        inject_time - pd.Timedelta(minutes=OBSERVATION_BEFORE_MINUTES),
        inject_time,
        inject_time + pd.Timedelta(minutes=OBSERVATION_AFTER_MINUTES),
    )


def extract_call_volume(run_path, case_id):
    # Tugas: menghitung, untuk satu run, berapa kali service A memanggil service B
    # pada window baseline dan pada window incident.
    # Cara mengenali panggilan: setiap span punya operationName seperti
    # "hipstershop.CartService/GetCart". Service di dalam nama itu (CartService)
    # adalah yang dipanggil; serviceName pemilik span adalah yang memanggil.
    # Jika keduanya sama, span itu span sisi server (bukan panggilan keluar) -> dilewati.
    # Hanya tiga kolom yang dibaca dari traces.csv, sehingga cukup cepat (±2,5 detik).

    inject_epoch = int((run_path / "inject_time.txt").read_text(encoding="utf-8").strip())
    baseline_start, inject_time, incident_end = _window_bounds(inject_epoch)

    spans = pd.read_csv(
        run_path / "traces.csv",
        usecols=["serviceName", "operationName", "startTimeMillis"],
        dtype={"serviceName": "string", "operationName": "string"},
    )
    callee = spans["operationName"].str.extract(CALLEE_PATTERN, expand=False).str.lower()
    is_client_call = callee.notna() & callee.ne(spans["serviceName"])

    calls = pd.DataFrame({
        "caller": spans.loc[is_client_call, "serviceName"],
        "callee": callee[is_client_call],
        "timestamp": pd.to_datetime(
            spans.loc[is_client_call, "startTimeMillis"], unit="ms", utc=True, errors="coerce"
        ),
    })
    calls["window"] = "outside"
    calls.loc[calls["timestamp"].ge(baseline_start) & calls["timestamp"].lt(inject_time), "window"] = "baseline"
    calls.loc[calls["timestamp"].ge(inject_time) & calls["timestamp"].lt(incident_end), "window"] = "incident"

    volume = (
        calls.loc[calls["window"].ne("outside")]
        .groupby(["caller", "callee", "window"]).size()
        .unstack("window", fill_value=0)
        .reindex(columns=["baseline", "incident"], fill_value=0)
        .reset_index()
        .rename(columns={"baseline": "baseline_calls", "incident": "incident_calls"})
    )
    volume.insert(0, "case_id", case_id)
    return volume


def build_call_graph(cleaned_manifest):
    # Tugas: menjalankan extract_call_volume untuk seluruh run, lalu menghasilkan dua tabel:
    #   - service_call_graph.csv        : peta statis siapa memanggil siapa (union semua run)
    #   - service_call_volume_by_case.csv: jumlah panggilan per hubungan per run per window
    # Karena butuh ±4 menit, hasilnya disimpan dan dipakai ulang kecuali REBUILD_CALL_GRAPH=True.

    graph_path = OUTPUT_ROOT / "service_call_graph.csv"
    volume_path = OUTPUT_ROOT / "service_call_volume_by_case.csv"

    if graph_path.exists() and volume_path.exists() and not REBUILD_CALL_GRAPH:
        print("[OK] Call graph dibaca dari cache:", graph_path.name)
        return pd.read_csv(graph_path), pd.read_csv(volume_path)

    volumes = []
    total_runs = len(cleaned_manifest)
    for index, row in enumerate(cleaned_manifest.itertuples(index=False), start=1):
        run_path = DATASET_ROOT / row.scenario / str(row.run_number)
        volumes.append(extract_call_volume(run_path, row.case_id))
        print(f"[OK] Call graph {index}/{total_runs}: {row.case_id}")

    volume = pd.concat(volumes, ignore_index=True)
    volume["total_calls"] = volume["baseline_calls"] + volume["incident_calls"]
    graph = (
        volume.groupby(["caller", "callee"])
        .agg(run_count=("case_id", "nunique"), total_calls=("total_calls", "sum"))
        .reset_index()
    )
    volume = volume.drop(columns="total_calls")

    graph.to_csv(graph_path, index=False)
    volume.to_csv(volume_path, index=False)
    return graph, volume


#9.5b Anomali per service dari fitur trace
def _service_of_trace_column(feature_name):
    # "trace_latency__checkoutservice_PlaceOrder__zscore_mean" -> "checkoutservice"

    operation = feature_name.split("__", 2)[1]
    return operation.split("_", 1)[0]


def _is_anomalous(zscore_mean, baseline_mean, delta_mean):
    # Tugas: memutuskan satu operasi anomali atau tidak, dengan aturan yang dikunci.
    # Kasus khusus: baseline 0 (tidak ada aktivitas sebelum injeksi) -> anomali
    # jika ada aktivitas apa pun saat incident, karena rasio tidak bisa dihitung.

    incident_mean = baseline_mean + delta_mean
    if pd.isna(zscore_mean):
        return False
    if baseline_mean == 0:
        return incident_mean > 0
    return (
        zscore_mean > SERVICE_ANOMALY_ZSCORE_THRESHOLD
        and (incident_mean / baseline_mean) > SERVICE_ANOMALY_RATIO_THRESHOLD
    )


def compute_service_anomalies(features_row, modality):
    # Tugas: dari satu baris fitur, menentukan service mana yang anomali dan seberapa besar.
    # Input : features_row = satu baris tabel fitur; modality = "trace_latency" atau "trace_error".
    # Output: dict {nama_service: besar anomali}. Besar anomali = zscore_mean tertinggi
    #         di antara operasi milik service itu yang lolos aturan; 0 jika tidak ada.
    # Catatan: dipanggil SEBELUM pembuangan fitur konstan, karena membutuhkan
    # kolom baseline_mean yang bisa saja konstan (misalnya trace_error selalu 0).

    prefix = f"{modality}__"
    zscore_columns = [
        c for c in features_row.index
        if c.startswith(prefix) and c.endswith("__zscore_mean")
    ]
    magnitude = {}
    for column in zscore_columns:
        service = _service_of_trace_column(column)
        if service in SERVICE_CONTEXT_EXCLUDED_SERVICES:
            continue
        base = column[: -len("__zscore_mean")]
        anomalous = _is_anomalous(
            features_row[column],
            features_row[f"{base}__baseline_mean"],
            features_row[f"{base}__delta_mean"],
        )
        score = float(features_row[column]) if anomalous else 0.0
        magnitude[service] = max(magnitude.get(service, 0.0), score)
    return magnitude


#9.5c Fitur konteks service per run
def _longest_anomalous_chain(anomalous, callees):
    # Tugas: mencari rantai pemanggilan terpanjang yang seluruh service-nya anomali.
    # Contoh: frontend -> checkout -> payment semua anomali -> kedalaman 3.
    # Dihitung dengan penelusuran rekursif; pengaman "visiting" mencegah loop tak berujung.

    memo = {}

    def depth(node, visiting):
        if node in memo:
            return memo[node]
        if node in visiting:
            return 0
        visiting.add(node)
        best = 0
        for child in callees.get(node, ()):
            if child in anomalous:
                best = max(best, depth(child, visiting))
        visiting.discard(node)
        memo[node] = best + 1
        return memo[node]

    return max((depth(node, set()) for node in anomalous), default=0)


def build_service_context_features(features, graph, volume):
    # Tugas: menghitung 15 fitur svc__ untuk setiap run. Tiga kelompok:
    #   1. Berapa service anomali (dari fitur trace, tanpa peta)
    #   2. Bagaimana sebarannya di peta (butuh call graph): hubungan yang kedua ujungnya
    #      anomali, rantai terpanjang, service "sumber" (anomali tapi pemanggilnya tidak),
    #      seberapa "menular" (tetangga yang ikut anomali), apakah sampai ke frontend
    #   3. Perubahan jumlah panggilan antar-service saat incident (dari volume)

    nodes = sorted(set(graph["caller"]) | set(graph["callee"]))
    edges = list(zip(graph["caller"], graph["callee"]))
    callees = {}
    callers = {}
    for caller, callee in edges:
        callees.setdefault(caller, set()).add(callee)
        callers.setdefault(callee, set()).add(caller)

    volume_by_case = {case_id: frame for case_id, frame in volume.groupby("case_id")}
    records = []

    for _, row in features.set_index("case_id").iterrows():
        case_id = row.name
        latency = compute_service_anomalies(row, "trace_latency")
        error = compute_service_anomalies(row, "trace_error")
        anomalous = {s for s, score in latency.items() if score > 0}

        anomalous_edges = [(a, b) for a, b in edges if a in anomalous and b in anomalous]
        roots = [s for s in anomalous if not (callers.get(s, set()) & anomalous)]

        contagion = []
        for service in anomalous:
            neighbors = callers.get(service, set()) | callees.get(service, set())
            if neighbors:
                contagion.append(len(neighbors & anomalous) / len(neighbors))

        case_volume = volume_by_case.get(case_id)
        if case_volume is not None and len(case_volume):
            base = case_volume["baseline_calls"].astype(float)
            inc = case_volume["incident_calls"].astype(float)
            drop = ((base - inc) / base.replace(0, float("nan"))).fillna(0)
            drop_share = float((drop > CALL_VOLUME_DROP_THRESHOLD).mean())
            max_drop = float(drop.max())
            total_change = float((inc.sum() - base.sum()) / base.sum()) if base.sum() else 0.0
        else:
            drop_share, max_drop, total_change = 0.0, 0.0, 0.0

        records.append({
            "case_id": case_id,
            "svc__anomalous_service_count": len(anomalous),
            "svc__anomalous_service_share": len(anomalous) / len(nodes),
            "svc__latency_anomaly_magnitude": float(sum(latency.values())),
            "svc__error_service_count": sum(1 for s in error.values() if s > 0),
            "svc__anomalous_edge_count": len(anomalous_edges),
            "svc__anomalous_edge_share": len(anomalous_edges) / len(edges),
            "svc__propagation_depth": _longest_anomalous_chain(anomalous, callees),
            "svc__anomalous_root_count": len(roots),
            "svc__contagion_ratio": float(sum(contagion) / len(contagion)) if contagion else 0.0,
            "svc__frontend_anomalous": int("frontendservice" in anomalous),
            "svc__max_in_degree_anomalous": max((len(callers.get(s, ())) for s in anomalous), default=0),
            "svc__max_out_degree_anomalous": max((len(callees.get(s, ())) for s in anomalous), default=0),
            "svc__call_volume_drop_share": drop_share,
            "svc__call_volume_max_drop": max_drop,
            "svc__call_volume_total_change": total_change,
        })

    return pd.DataFrame(records)


#9.5d Proxy audit: apakah fitur svc__ hanya menyalin tier?
def audit_service_context_proxy(svc_features, cleaned_manifest):
    # Tugas: menguji setiap fitur svc__ terhadap dua tanda kebocoran, lalu memutuskan
    # lolos atau dibuang. Nama service dan tier dipakai HANYA di sini, untuk audit;
    # keduanya tidak pernah masuk ke tabel fitur.
    #   Uji 1: pohon keputusan kedalaman 2 pada fitur itu saja -> seberapa akurat
    #          menebak tier (A/B/C). Jika >= 0.85, fitur itu praktis menyalin tier.
    #   Uji 2: rasio varians di dalam satu service terhadap varians total. Jika < 0.05,
    #          fitur hampir tidak berubah antar-run dalam satu service, artinya dia
    #          hanya "kode nama service", bukan sinyal dari telemetri.

    tier_by_service = LABEL_POLICY["service_tier"]
    meta = cleaned_manifest.set_index("case_id")
    aligned = svc_features.set_index("case_id")
    service = meta.loc[aligned.index, "service"]
    tier = service.map(tier_by_service)

    records = []
    for column in aligned.columns:
        x = aligned[column].astype(float)
        total_var = float(x.var(ddof=0))
        if total_var == 0:
            records.append({"feature_name": column, "tier_balanced_accuracy": float("nan"),
                            "within_service_variance_ratio": float("nan"),
                            "decision": "rejected", "reason": "constant"})
            continue

        within_var = float(x.groupby(service.values).var(ddof=0).mean())
        within_ratio = within_var / total_var

        tree = DecisionTreeClassifier(max_depth=PROXY_AUDIT_TREE_DEPTH, random_state=0)
        tree.fit(x.to_frame(), tier)
        tier_acc = float(balanced_accuracy_score(tier, tree.predict(x.to_frame())))

        reasons = []
        if tier_acc >= PROXY_AUDIT_MAX_TIER_ACCURACY:
            reasons.append("encodes_tier")
        if within_ratio < PROXY_AUDIT_MIN_WITHIN_VARIANCE_RATIO:
            reasons.append("service_identifier")

        records.append({
            "feature_name": column,
            "tier_balanced_accuracy": round(tier_acc, 4),
            "within_service_variance_ratio": round(within_ratio, 4),
            "decision": "rejected" if reasons else "passed",
            "reason": "|".join(reasons),
        })
    return pd.DataFrame(records)


def add_service_context_features(features, cleaned_manifest):
    # Tugas: merangkai 9.5a-9.5d. Membangun call graph, menghitung fitur svc__,
    # mengaudit, membuang yang gagal, dan menempelkan yang lolos ke tabel fitur.
    # Mengembalikan tabel fitur baru, status audit tiap fitur svc__ untuk manifest,
    # tabel audit lengkap untuk laporan, dan call graph.

    graph, volume = build_call_graph(cleaned_manifest)
    svc_features = build_service_context_features(features, graph, volume)
    audit = audit_service_context_proxy(svc_features, cleaned_manifest)

    rejected = audit.loc[audit["decision"].eq("rejected"), "feature_name"].tolist()
    kept = svc_features.drop(columns=rejected)
    features = features.merge(kept, on="case_id", how="left", validate="one_to_one")

    audit_status = {name: "passed" for name in kept.columns if name != "case_id"}
    return features, audit_status, audit, graph




#9.6 Feature Manifest
def build_feature_manifest(feature_names, audit_status_by_feature=None):
    # Tugas: membuat satu baris keterangan untuk setiap kolom fitur.
    # Manifest inilah yang dipakai Fase 10 untuk memilih kolom mana yang masuk
    # E0, E1, E2, atau E3, dan dipakai Fase 11 sebagai daftar fitur resmi.
    # Nama fitur selalu berpola "<modality>__<sumber>__<ringkasan>":
    #   - bagian sebelum "__" pertama  = modality  -> menentukan konfigurasi
    #   - bagian setelah "__" terakhir = ringkasan -> jenis statistik
    # Pemecahan dilakukan dari kanan (rsplit) supaya nama sumber yang sendirinya
    # memuat "__", seperti log__frontend__error, tidak terpotong salah.


    records = []
    for name in feature_names:
        modality = name.split("__", 1)[0]
        source_column, statistic = name.rsplit("__", 1)
        if statistic == "new_template_count":
            source_column = "column_harmonization_map.csv"

        records.append({
            "feature_name": name,
            "modality": modality,
            "config": MODALITY_CONFIG[modality],
            "source_column": source_column,
            "statistic": statistic,
            "origin": "derived",
            "audit_status": (audit_status_by_feature or {}).get(name, "n/a"),
        })
    return pd.DataFrame(records)


#9.7 Validasi Tabel Fitur
def validate_feature_table(features, manifest, expected_case_ids):
    # Tugas: memastikan tabel fitur layak diteruskan ke Fase 10. Empat pemeriksaan:
    #   1. Satu baris per run; case_id unik dan sama persis dengan daftar run Fase 8.
    #   2. Tidak ada kolom terlarang, yaitu label atau bahan pembentuk label.
    #   3. Semua kolom fitur berisi angka dan tidak ada nilai tak hingga.
    #      Nilai tak hingga bisa muncul dari pembagian, karena itu ditolak di sini.
    #   4. Setiap kolom fitur punya satu baris di manifest, dan sebaliknya.
    # Nilai kosong (NaN) tidak ditolak, karena memang ada kolom yang absen di
    # sebagian run. Jumlahnya dikembalikan agar dicatat di laporan.

    if features["case_id"].duplicated().any():
        raise RuntimeError("Ditemukan case_id duplikat pada tabel fitur")
    if set(features["case_id"]) != set(expected_case_ids):
        raise RuntimeError("case_id tabel fitur tidak sama dengan daftar run fase 8")

    forbidden = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in features.columns]
    if forbidden:
        raise RuntimeError(f"Kolom terlarang ditemukan di tabel fitur: {forbidden}")

    feature_columns = [c for c in features.columns if c != "case_id"]
    non_numeric = [
        c for c in feature_columns if not pd.api.types.is_numeric_dtype(features[c])
    ]
    if non_numeric:
        raise RuntimeError(f"Kolom fitur non numerik: {non_numeric}")

    values = features[feature_columns]
    if values.abs().eq(float("inf")).any().any():
        raise RuntimeError("Nilai tak hingga ditemukan di tabel fitur")

    if set(manifest["feature_name"]) != set(feature_columns):
        raise RuntimeError("Manifest tidak sesuai dengan kolom tabel fitur")
    
    if manifest["feature_name"].duplicated().any():
        raise RuntimeError("Ditemukan feature_name duplikat pada manifest")

    return int(values.isna().sum().sum())



#9.8 Phase 9 Orchestrator
def run_phase_9():
    # Tugas: menjalankan seluruh Fase 9 secara berurutan.
    #   1. Baca tiga keluaran Fase 8: daftar file bersih, daftar kolom (schema),
    #      dan peta kolom log.
    #   2. Untuk setiap run: baca file bersih, ringkas menjadi satu baris.
    #   3. Tambahkan fitur kalimat log baru.
    #   4. Buang fitur yang konstan di seluruh run.
    #   5. Susun manifest, validasi, simpan tabel fitur dan laporan.

    print("\n[Pipeline] Memulai Fase 9: Feature extraction...")

    cleaned_manifest_path = OUTPUT_ROOT / "cleaned_timeseries_manifest.csv"
    schema_path = OUTPUT_ROOT / "feature_column_schema.csv"
    harmonization_path = OUTPUT_ROOT / "column_harmonization_map.csv"
    for path in (cleaned_manifest_path, schema_path, harmonization_path):
        if not path.exists():
            raise FileNotFoundError(f"Output Fase 8 belum tersedia: {path}")

    cleaned_manifest = pd.read_csv(cleaned_manifest_path, dtype={"run_number": "string"})
    schema = pd.read_csv(schema_path)
    harmonization_map = pd.read_csv(harmonization_path, dtype={"cluster_id": "string"})

    feature_columns = schema.loc[schema["keep"], "column"].tolist()
    total_runs = len(cleaned_manifest)
    rows = []

    for index, row in enumerate(cleaned_manifest.itertuples(index=False), start=1):
        rows.append(extract_run_features(PROJECT_ROOT / Path(row.cleaned_path), feature_columns))
        print(f"[OK] Extraction {index}/{total_runs}: {row.case_id}")

    features = pd.DataFrame(rows)

    novelty = build_log_novelty_features(harmonization_map, features["case_id"].tolist())
    features = features.merge(novelty, on="case_id", how="left", validate="one_to_one")

    # Service context dihitung SEBELUM pembuangan fitur konstan, karena membutuhkan
    # kolom baseline_mean trace yang bisa saja konstan di seluruh run
    audit_status_by_feature = {}
    proxy_audit = pd.DataFrame()
    if INCLUDE_SERVICE_CONTEXT:
        features, audit_status_by_feature, proxy_audit, call_graph = (
            add_service_context_features(features, cleaned_manifest)
        )
        proxy_audit.to_csv(OUTPUT_ROOT / "service_context_proxy_audit.csv", index=False)


    raw_feature_count = len(features.columns) - 1
    features, dropped_constant = drop_constant_features(features)

    feature_names = [c for c in features.columns if c != "case_id"]
    manifest = build_feature_manifest(feature_names, audit_status_by_feature)
    nan_cell_count = validate_feature_table(features, manifest, cleaned_manifest["case_id"])

    features_path = OUTPUT_ROOT / "features_by_case.csv"
    manifest_path = OUTPUT_ROOT / "feature_manifest.csv"
    report_path = OUTPUT_ROOT / "feature_extraction_report.json"

    features.to_csv(features_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    nan_by_column = features[feature_names].isna().sum()
    per_config = manifest.groupby("config").size()

    with report_path.open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "row_count": int(len(features)),
            "feature_count": int(len(feature_names)),
            "feature_count_before_drop": int(raw_feature_count),
            "dropped_constant_count": len(dropped_constant),
            "dropped_constant_features": dropped_constant,
            "feature_count_by_config": {k: int(v) for k, v in per_config.items()},
            "statistics": list(FEATURE_STATISTICS),
            "zscore_cap": ZSCORE_CAP,
            "include_service_context": INCLUDE_SERVICE_CONTEXT,
            "service_context_rejected": (
                proxy_audit.loc[proxy_audit["decision"].eq("rejected"), "feature_name"].tolist()
                if len(proxy_audit) else []
            ),
            "service_context_passed": [k for k, v in audit_status_by_feature.items() if v == "passed"],
            "nan_cell_count": nan_cell_count,
            "columns_with_nan": nan_by_column[nan_by_column.gt(0)].to_dict(),
        }, file, ensure_ascii=False, indent=2)

    print("\n========== PHASE 9 SUMMARY ==========")
    print("Run (baris)             :", len(features))
    print("Fitur sebelum dibuang   :", raw_feature_count)
    print("Fitur konstan dibuang   :", len(dropped_constant))
    print("Fitur final (kolom)     :", len(feature_names))
    if len(proxy_audit):
        print("\nProxy audit service context:")
        print(proxy_audit[["feature_name", "tier_balanced_accuracy",
                           "within_service_variance_ratio", "decision", "reason"]].to_string(index=False))
    print("\nFitur per konfigurasi:")
    print(per_config.reindex(["E0", "E1", "E2", "E3"], fill_value=0))
    print("\nSel NaN pada tabel fitur:", nan_cell_count)
    if nan_cell_count:
        print(nan_by_column[nan_by_column.gt(0)].to_string())
    print("\nContoh 5 fitur dengan zscore_mean tertinggi pada run pertama:")
    first = features.iloc[0]
    zscore_columns = [c for c in feature_names if c.endswith("__zscore_mean")]
    print(first[zscore_columns].sort_values(ascending=False).head(5).to_string())
    print("\nTabel fitur disimpan di :", features_path)
    print("Manifest disimpan di    :", manifest_path)
    print("[OK] Fase 9 selesai")

    return {"features": features, "manifest": manifest, "dropped": dropped_constant}

#===================================#
# 10. FINAL DATASET ASSEMBLY
#===================================#

#10.1 Memuat output fase 6,8, dan 9
def resolve_label_path():
    # Tugas: menentukan nama file label yang ditulis Fase 6.
    # Fase 6 memberi akhiran versi selama policy masih "candidate"
    # (derived_severity_labels_fcw_sct_v1.csv) dan tanpa akhiran setelah dikunci.
    # Aturan yang sama diulang di sini supaya keduanya selalu sepakat.

    version = str(LABEL_POLICY["policy_version"])
    suffix = "" if LABEL_POLICY["is_frozen"] else f"_{version}"
    return OUTPUT_ROOT / f"derived_severity_labels{suffix}.csv"


def load_assembly_inputs():
    # Tugas: membaca empat file masukan dan berhenti dengan pesan jelas jika ada yang belum ada.
    #   features_by_case.csv           -> tabel fitur (Fase 9)
    #   feature_manifest.csv           -> kamus fitur, penentu E0-E3 (Fase 9)
    #   derived_severity_labels_*.csv  -> label (Fase 6)
    #   cleaned_timeseries_manifest.csv-> identitas run (Fase 8)

    paths = {
        "features": OUTPUT_ROOT / "features_by_case.csv",
        "manifest": OUTPUT_ROOT / "feature_manifest.csv",
        "labels": resolve_label_path(),
        "cleaned_manifest": OUTPUT_ROOT / "cleaned_timeseries_manifest.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Masukan Fase 10 ({name}) belum tersedia: {path}")

    features = pd.read_csv(paths["features"])
    manifest = pd.read_csv(paths["manifest"])
    labels = pd.read_csv(paths["labels"], dtype={"run_number": "string"})
    cleaned_manifest = pd.read_csv(paths["cleaned_manifest"], dtype={"run_number": "string"})
    return features, manifest, labels, cleaned_manifest




#10.2 Metadata Run
def build_case_metadata(cleaned_manifest, case_ids):
    # Tugas: membuat tabel identitas, satu baris per run, dengan urutan baris
    # persis sama seperti tabel fitur. Kolom scenario nanti menjadi "group"
    # pada cross-validation supaya tiga repetisi satu scenario tidak terpisah.

    metadata = cleaned_manifest[CASE_METADATA_COLUMNS].set_index("case_id")
    return metadata.loc[case_ids].reset_index()



#10.3 Penyelarasan Label
def align_labels(labels, case_ids):
    # Tugas: mengambil label untuk setiap run yang punya fitur, dalam urutan yang sama.
    # Dua aturan:
    #   - setiap run berfitur WAJIB punya label -> jika tidak, berhenti (error)
    #   - label tanpa fitur (run yang gugur di Fase 7) BOLEH ada -> dicatat, bukan error
    # Penggabungan selalu lewat case_id, tidak pernah lewat urutan baris.

    label_index = labels.set_index("case_id")

    missing = [case_id for case_id in case_ids if case_id not in label_index.index]
    if missing:
        raise RuntimeError(f"Run berikut punya fitur tetapi tidak punya label: {missing[:5]}")

    without_features = sorted(set(label_index.index) - set(case_ids))
    aligned = label_index.loc[case_ids].reset_index()
    return aligned, without_features


#10.4 Pemilihan Kolom per Konfigurasi
def select_features(features, manifest, config):
    # Tugas: mengambil kolom fitur yang boleh dilihat model pada konfigurasi tertentu.
    # Konfigurasi bersifat bertingkat: E2 = kolom E0 + E1 + E2. Daftar kolom diambil
    # dari manifest (kolom "config"), bukan dari tebakan awalan nama, dan kolom yang
    # ditolak audit tidak pernah ikut walaupun ada di tabel fitur.


    allowed = CONFIG_ORDER[: CONFIG_ORDER.index(config) + 1]
    selected = manifest.loc[
        manifest["config"].isin(allowed) & manifest["audit_status"]
        .ne("rejected"), "feature_name",
    ].tolist()
    return features[["case_id"] + selected]



#10.5 Penulisan Enam File Training
def write_training_dataset(features, manifest, labels, metadata):
    # Tugas: menulis empat file fitur, satu file label, dan satu file metadata ke
    # folder training. Mengembalikan ringkasan (nama file dan jumlah fitur) per konfigurasi.

    TRAINING_DATASET_ROOT.mkdir(parents=True, exist_ok = True)
    written = {}

    for config in CONFIG_ORDER:
        frame = select_features(features, manifest, config)
        path = TRAINING_DATASET_ROOT / f"features_{config}.csv"
        frame.to_csv(path, index=False)
        written[config] = {"file": path.name, "feature_count": int(frame.shape[1] - 1)}

    labels.to_csv(TRAINING_DATASET_ROOT / "derived_severity_labels.csv", index=False)
    metadata.to_csv(TRAINING_DATASET_ROOT / "case_metadata.csv", index=False)
    return written



#10.6 Pemeriksaan Keterpaduan X-y
def validate_training_dataset(case_ids, labels, metadata):
    # Tugas: memeriksa file yang SUDAH TERTULIS di disk (bukan variabel di memori),
    # karena itulah yang akan dibaca classifier.py. Lima pemeriksaan:
    #   1. case_id unik, dan urutannya identik di fitur, label, dan metadata
    #   2. label hanya berisi Low / Medium / High
    #   3. tiap file E memuat semua kolom file E sebelumnya (bertingkat)
    #   4. tidak ada kolom terlarang di file fitur mana pun
    #   5. semua kolom fitur berupa angka

    
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("Ditemukan case_id duplikat")
    if labels["case_id"].tolist() != case_ids:
        raise RuntimeError("Urutan atau isi case_id pada label tidak sama dengan fitur")
    if metadata["case_id"].tolist() != case_ids:
        raise RuntimeError("Urutan atau isi case_id pada metadata tidak sama dengan fitur")
    if not set(labels["severity"]).issubset(SEVERITY_ORDER):
        raise RuntimeError("Label memuat kelas severity yang tidak dikenal")

    
    previous_columns = set()
    for config in CONFIG_ORDER:
        frame = pd.read_csv(TRAINING_DATASET_ROOT / f"features_{config}.csv")

        if frame["case_id"].tolist() != case_ids:
            raise RuntimeError(f"features_{config}.csv: urutan case_id tidak sama dengan label")

        columns = set(frame.columns) - {"case_id"}
        if not previous_columns.issubset(columns):
            raise RuntimeError(f"features_{config}.csv tidak memuat seluruh kolom konfigurasi sebelumnya")

        forbidden = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in frame.columns]
        if forbidden:
            raise RuntimeError(f"features_{config}.csv memuat kolom terlarang: {forbidden}")

        non_numeric = frame.drop(columns=["case_id"]).select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise RuntimeError(f"features_{config}.csv memuat kolom non-numerik: {non_numeric[:5]}")

        previous_columns = columns
    
    return True


#10.7 Dokumen Definisi Konfigurasi
def save_config_definition(written, labels, metadata, labels_without_features):
    # Tugas: menulis satu file JSON yang menjelaskan isi folder training:
    # berapa baris, versi label, sebaran kelas, modality tiap konfigurasi, jumlah fitur,
    # dan run mana yang punya label tetapi tidak punya fitur. Ini sumber angka untuk
    # tabel "konfigurasi x jumlah fitur" di Bab 4.

    def modalities_for(config):
        limit = CONFIG_ORDER.index(config)
        return [m for m, cfg in MODALITY_CONFIG.items() if CONFIG_ORDER.index(cfg) <= limit]

    distribution = labels["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)

    definition = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": int(len(labels)),
        "label_policy_version": str(LABEL_POLICY["policy_version"]),
        "label_status": "final" if LABEL_POLICY["is_frozen"] else "candidate",
        "severity_distribution": {k: int(v) for k, v in distribution.items()},
        "group_column": "scenario",
        "group_count": int(metadata["scenario"].nunique()),
        "configurations": {
            config: {"modalities": modalities_for(config), **written[config]}
            for config in CONFIG_ORDER
        },
        "labels_without_features": labels_without_features,
        "files": {
            "labels": "derived_severity_labels.csv",
            "metadata": "case_metadata.csv",
            "target_column": "severity",
        },
    }

    with (TRAINING_DATASET_ROOT / "config_definition.json").open("w", encoding="utf-8") as file:
        json.dump(definition, file, ensure_ascii=False, indent=2)
    return definition


#10.8 Phase 10 Orchestrator
def run_phase_10():
    # Tugas: menjalankan seluruh Fase 10 berurutan.
    #   1. Baca fitur, manifest, label, metadata.
    #   2. Urutkan run berdasarkan case_id supaya urutan tetap dan sama di semua file.
    #   3. Selaraskan label dan metadata ke urutan itu.
    #   4. Tulis enam file training.
    #   5. Periksa ulang file yang tertulis, lalu simpan definisi konfigurasi.

    print("\n[Pipeline] Memulai Fase 10: Final dataset assembly...")

    features, manifest, labels, cleaned_manifest = load_assembly_inputs()

    features = features.sort_values("case_id").reset_index(drop=True)
    case_ids = features["case_id"].tolist()

    labels_aligned, labels_without_features = align_labels(labels, case_ids)
    metadata = build_case_metadata(cleaned_manifest, case_ids)

    written = write_training_dataset(features, manifest, labels_aligned, metadata)
    validate_training_dataset(case_ids, labels_aligned, metadata)
    definition = save_config_definition(written, labels_aligned, metadata, labels_without_features)

    print("\n========== PHASE 10 SUMMARY ==========")
    print("Run (baris)             :", definition["row_count"])
    print("Label tanpa fitur       :", len(labels_without_features), labels_without_features)
    print("Versi label             :", definition["label_policy_version"],
          f"({definition['label_status']})")
    print("Distribusi severity     :", definition["severity_distribution"])
    print("Group CV (scenario)     :", definition["group_count"])
    print("\nFitur per konfigurasi (kumulatif):")
    for config in CONFIG_ORDER:
        info = definition["configurations"][config]
        print(f"  {config}: {info['feature_count']:4d} fitur  <- {', '.join(info['modalities'])}")
    print("\nFolder training         :", TRAINING_DATASET_ROOT)
    print("[OK] Fase 10 selesai")

    return definition



#===================================#
# 11. FINAL VALIDATION AND HANDOFF
#===================================#

#11.1 Memuat dan Memeriksa Skema Dataset Training
def load_training_dataset():
    # Tugas: membaca ketujuh file di folder training ke dalam satu dict.
    # Kunci "E0".."E3" = tabel fitur, "labels", "metadata", dan "definition" (isi JSON).
    # Berhenti dengan pesan jelas jika ada file yang belum ada.

    for name in TRAINING_FILES:
        path  = TRAINING_DATASET_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"File training belum tersedia: {path}")

    training = {
        config: pd.read_csv(TRAINING_DATASET_ROOT / f"features_{config}.csv")
        for config in CONFIG_ORDER
    }

    training["labels"] = pd.read_csv(TRAINING_DATASET_ROOT / "derived_severity_labels.csv", dtype={"run_number": "string"})
    training["metadata"] = pd.read_csv(TRAINING_DATASET_ROOT / "case_metadata.csv", dtype={"run_number": "string"})

    with (TRAINING_DATASET_ROOT / "config_definition.json").open("r", encoding="utf-8") as file:
        training["definition"] = json.load(file)

    return training



def validate_training_schema(training):
    # Tugas: memastikan bentuk tiap file sesuai yang dijanjikan.
    #   - file fitur: kolom pertama case_id, sisanya angka, jumlahnya persis
    #     seperti yang tercatat di config_definition.json
    #   - file label: punya case_id dan severity
    #   - file metadata: punya lima kolom identitas

    for config in CONFIG_ORDER:
        frame = training[config]
        if frame.columns[0] != "case_id":
            raise RuntimeError(f"features_{config}.csv: kolom pertama harus case_id")

        non_numeric = frame.drop(columns=["case_id"]).select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise RuntimeError(f"features_{config}.csv: kolom non numerik {non_numeric[:5]}")

        expected = int(training["definition"]["configurations"][config]["feature_count"])
        if frame.shape[1] - 1 != expected:
            raise RuntimeError(
                f"features_{config}.csv: {frame.shape[1] - 1} fitur, config_definition menyebut {expected}"
            )

    for column in ("case_id", "severity"):
        if column not in training["labels"].columns:
            raise RuntimeError(f"derived_severity_labels.csv: kolom {column} tidak ada")

    missing_metadata = set(CASE_METADATA_COLUMNS) - set(training["metadata"].columns)
    if missing_metadata:
        raise RuntimeError(f"case_metadata.csv: kolom {sorted(missing_metadata)} tidak ada /hilang ")

    return True


#11.2 Pemeriksaan Kualitas Numerik
def validate_numeric_quality(training):
    # Tugas: memastikan angka di file fitur layak dipakai model.
    #   Ditolak : nilai tak hingga, kolom yang kosong seluruhnya, kolom konstan
    #   Diizinkan: NaN pada sebagian sel (kolom yang absen di beberapa run),
    #              tetapi jumlah dan nama kolomnya dicatat untuk laporan


    quality = {}
    for config in CONFIG_ORDER:
        values = training[config].drop(columns="case_id")

        inf_count = int(values.abs().eq(float("inf")).sum().sum())
        if inf_count:
            raise RuntimeError(f"features_{config}.csv: {inf_count} nilai tak hingga")

        all_nan_columns = values.columns[values.isna().all()].tolist()
        if all_nan_columns:
            raise RuntimeError(f"features_{config}.csv: kolom kosong seluruhnya {all_nan_columns[:5]}")

        constant_columns = [
            c for c in values.columns if values[c].notna().all() and values[c].nunique() <= 1
        ]
        if constant_columns:
            raise RuntimeError(f"features_{config}.csv: kolom konstan {constant_columns[:5]}")

        quality[config] = {
            "feature_count": int(values.shape[1]),
            "nan_cell_count": int(values.isna().sum().sum()),
            "columns_with_nan": values.columns[values.isna().any()].tolist(),
        }


    return quality


#11.3 Pemeriksaan Konsistensi case_id
def validate_case_consistency(training):
    # Tugas: memastikan keenam file berbicara tentang run yang sama, dalam urutan
    # yang sama. features_E0.csv dijadikan acuan; file lain harus identik dengannya.
    # Juga memastikan jumlah baris cocok dengan config_definition.json dan label
    # hanya berisi tiga kelas yang dikenal. Mengembalikan angka ringkasan untuk laporan.


    reference = training["E0"]["case_id"].tolist()
    if len(set(reference)) != len(reference):
        raise RuntimeError("case_id duplikat pada features_E0.csv")

    for name in (*CONFIG_ORDER, "labels", "metadata"):
        if training[name]["case_id"].tolist() != reference:
            raise RuntimeError(f"{name}: daftar atau urutan case_id tidak identik dengan features_E0.csv")

    if len(reference) != int(training["definition"]["row_count"]):
        raise RuntimeError("Jumlah baris tidak sama dengan config_definition.json")

    if not set(training["labels"]["severity"]).issubset(SEVERITY_ORDER):
        raise RuntimeError("Label memuat kelas severity yang tidak dikenal")

    distribution = training["labels"]["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    return {
        "row_count": len(reference),
        "group_count": int(training["metadata"]["scenario"].nunique()),
        "severity_distribution": {k: int(v) for k, v in distribution.items()},
    }


#11.4 Audit Kebocoran Label
def audit_leakage(training):
    # Tugas: membuktikan tidak ada jalan bagi model untuk "mengintip" jawabannya.
    # Tiga pemeriksaan pada setiap file fitur:
    #   1. Tidak ada kolom terlarang (severity, service, fault_type, dll.)
    #   2. Tidak ada fitur svc__ yang ditolak proxy audit Fase 9.5
    #   3. Tidak ada kolom fitur yang nilainya PERSIS SAMA dengan fcw, sct, atau
    #      severity_score di file label (salinan langsung bahan pembentuk label)
    # Satu pelanggaran saja -> berhenti. Jika bersih, hasil lengkap dikembalikan
    # untuk ditulis ke leakage_audit.json.

    rejected = []
    audit_path = OUTPUT_ROOT / "service_context_proxy_audit.csv"
    if audit_path.exists():
        proxy_audit = pd.read_csv(audit_path)
        rejected = proxy_audit.loc[proxy_audit["decision"].eq("rejected"), "feature_name"].tolist()

    labels = training["labels"].set_index("case_id")
    results = {}
    violations = []

    for config in CONFIG_ORDER:
        frame = training[config].set_index("case_id")

        forbidden_present = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in frame.columns]
        rejected_present = [c for c in rejected if c in frame.columns]

        copies = []
        for ingredient in LABEL_INGREDIENT_COLUMNS:
            target = labels.loc[frame.index, ingredient].astype(float)
            for column in frame.columns:
                if frame[column].astype(float).equals(target):
                    copies.append(f"{column} == {ingredient}")

        results[config] = {
            "forbidden_columns_present": forbidden_present,
            "rejected_service_context_present": rejected_present,
            "exact_copies_of_label_ingredients": copies,
        }
        if forbidden_present or rejected_present or copies:
            violations.append(config)

    if violations:
        raise RuntimeError(f"Kebocoran terdeteksi pada konfigurasi: {violations} -> {results}")

    return {
        "forbidden_columns_checked": list(FORBIDDEN_FEATURE_COLUMNS),
        "rejected_service_context_checked": rejected,
        "label_ingredients_checked": list(LABEL_INGREDIENT_COLUMNS),
        "per_configuration": results,
        "status": "clean",
    }


#11.5 Laporan Kesiapan dan Reproduksibilitas
def _file_md5(path):
    # Tugas: menghitung sidik jari (md5) satu file. Dua file dengan isi persis sama
    # menghasilkan sidik jari sama; beda satu karakter saja sudah berbeda.
    # Dipakai untuk membuktikan file training yang dipakai model = file yang dilaporkan.

    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_excluded_runs():
    # Tugas: mengumpulkan run yang gugur di Fase 7 dan Fase 8 beserta alasannya,
    # supaya jawaban "kenapa 89, bukan 90" ada di satu tempat.

    excluded = {}

    alignment_path = OUTPUT_ROOT / "alignment_excluded_runs.csv"
    if alignment_path.exists():
        frame = pd.read_csv(alignment_path, dtype={"run_number": "string"})
        excluded["phase_7_alignment"] = frame[["scenario", "run_number", "exclusion_reason"]].to_dict("records")


    eligibility_path = OUTPUT_ROOT / "feature_eligibility.csv"
    if eligibility_path.exists():
        frame = pd.read_csv(eligibility_path)
        frame["eligible"] = _parse_required_boolean(frame["eligible"], "eligible")
        excluded["phase_8_cleaning"] = frame.loc[~frame["eligible"], ["case_id", "exclusion_reason"]].to_dict("records")

    return excluded



def build_readiness_report(training, quality, consistency, leakage):
    # Tugas: menyusun satu dokumen yang merangkum SELURUH keputusan preprocessing:
    # versi pustaka, aturan label, window, kebijakan cleaning, aturan fitur,
    # jumlah baris/fitur, hasil audit, run yang gugur, sidik jari file training,
    # dan file di data/processed yang tidak dikenal oleh kode saat ini.
    # Semua nilai diambil langsung dari konstanta konfigurasi - tidak ada yang diketik ulang.


    version = str(LABEL_POLICY["policy_version"])
    suffix =  "" if LABEL_POLICY["is_frozen"] else f"_{version}"

    expected_entries = set(EXPECTED_PROCESSED_ENTRIES) | {
        f"derived_severity_labels{suffix}.csv",
        f"derived_severity_distribution{suffix}.csv",
        f"derived_severity_sensitivity_by_run{suffix}.csv",
        f"derived_severity_sensitivity_summary{suffix}.csv",
        f"label_policy{suffix}.json",
    }

    actual_entries = {path.name for path in OUTPUT_ROOT.iterdir()}

    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "READY",
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "label_policy": {
            "version": version,
            "status": "final" if LABEL_POLICY["is_frozen"] else "candidate",
            "method": LABEL_POLICY["method"],
            "fault_category": LABEL_POLICY["fault_category"],
            "fault_category_weight": LABEL_POLICY["fault_category_weight"],
            "service_tier": LABEL_POLICY["service_tier"],
            "service_tier_weight": LABEL_POLICY["service_tier_weight"],
            "severity_bins": LABEL_POLICY["severity_bins"],
        },
        "observation_window": {
            "before_minutes": OBSERVATION_BEFORE_MINUTES,
            "after_minutes": OBSERVATION_AFTER_MINUTES,
            "interval_seconds": ALIGNMENT_INTERVAL_SECONDS,
            "require_full_timeseries_coverage": REQUIRE_FULL_TIMESERIES_COVERAGE,
        },
        "cleaning_policy": {
            "max_missing_bin_ratio": MAX_MISSING_BIN_RATIO,
            "max_edge_fill_bins": MAX_EDGE_FILL_BINS,
            "min_gauge_column_coverage": MIN_GAUGE_COLUMN_COVERAGE,
            "fill_policy": FILL_POLICY,
            "log_template_categories": {
                name: pattern.pattern for name, pattern in LOG_TEMPLATE_CATEGORIES
            },
        },
        "feature_extraction": {
            "statistics": list(FEATURE_STATISTICS),
            "zscore_cap": ZSCORE_CAP,
            "include_service_context": INCLUDE_SERVICE_CONTEXT,
            "service_anomaly_zscore_threshold": SERVICE_ANOMALY_ZSCORE_THRESHOLD,
            "service_anomaly_ratio_threshold": SERVICE_ANOMALY_RATIO_THRESHOLD,
            "call_volume_drop_threshold": CALL_VOLUME_DROP_THRESHOLD,
            "proxy_audit_max_tier_accuracy": PROXY_AUDIT_MAX_TIER_ACCURACY,
            "proxy_audit_min_within_variance_ratio": PROXY_AUDIT_MIN_WITHIN_VARIANCE_RATIO,
        },
        "dataset": {
            **consistency,
            "configurations": training["definition"]["configurations"],
            "labels_without_features": training["definition"].get("labels_without_features", []),
        },
        "numeric_quality": quality,
        "leakage_audit": leakage,
        "excluded_runs": collect_excluded_runs(),
        "training_files": {
            name: {
                "md5": _file_md5(TRAINING_DATASET_ROOT / name),
                "bytes": (TRAINING_DATASET_ROOT / name).stat().st_size,
            }
            for name in TRAINING_FILES
        },
        "unexpected_files_in_processed": sorted(actual_entries - expected_entries),
    }


#11.6 Phase 11 Orchestrator
def run_phase_11():
    # Tugas: menjalankan empat pemeriksaan berurutan, lalu menulis dua dokumen bukti.
    # Jika satu pemeriksaan gagal, program berhenti dan dokumen tidak ditulis -
    # tidak boleh ada laporan "READY" untuk dataset yang belum lolos.

    print("\n[Pipeline] Memulai Fase 11: Final validation dan classifier handoff...")

    training = load_training_dataset()
    validate_training_schema(training)
    print("[OK] 11.1 Skema dataset training sesuai")

    quality = validate_numeric_quality(training)
    print("[OK] 11.2 Kualitas numerik: tanpa inf, tanpa kolom kosong, tanpa kolom konstan")

    consistency = validate_case_consistency(training)
    print("[OK] 11.3 case_id identik di seluruh file training")

    leakage = audit_leakage(training)
    print("[OK] 11.4 Audit kebocoran: bersih")

    report = build_readiness_report(training, quality, consistency, leakage)

    report_path = OUTPUT_ROOT / "preprocessing_readiness_report.json"
    leakage_path = OUTPUT_ROOT / "leakage_audit.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with leakage_path.open("w", encoding="utf-8") as file:
        json.dump(leakage, file, ensure_ascii=False, indent=2)

    print("\n========== PHASE 11 SUMMARY ==========")
    print("Status                  :", report["status"])
    print("Run (baris)             :", consistency["row_count"], "| group:", consistency["group_count"])
    print("Distribusi severity     :", consistency["severity_distribution"])
    print("Versi label             :", report["label_policy"]["version"],
          f"({report['label_policy']['status']})")
    print("\nFitur dan sel NaN per konfigurasi:")
    for config in CONFIG_ORDER:
        print(f"  {config}: {quality[config]['feature_count']:4d} fitur, NaN = {quality[config]['nan_cell_count']}")
    print("\nRun yang dikeluarkan:")
    for phase, rows in report["excluded_runs"].items():
        names = [r.get("case_id") or f"{r['scenario']}-{r['run_number']}" for r in rows]
        print(f"  {phase}: {len(rows)} {names}")
    if report["unexpected_files_in_processed"]:
        print("\n[PERHATIAN] File di data/processed yang tidak dihasilkan kode saat ini:")
        for name in report["unexpected_files_in_processed"]:
            print("  -", name)
    print("\nReadiness report        :", report_path)
    print("Leakage audit           :", leakage_path)
    print("[OK] Fase 11 selesai - dataset siap diserahkan ke classifier.py")

    return report


#=====================================#
# RUNNER FASE 1-5
#=====================================#
def run_phases_1_to_5():
    
    # ===================================
    # 1. VALIDASI STRUKTUR DATASET
    # ===================================

    print(
        "[Pipeline] Memvalidasi struktur dataset..."
    )

    run_paths = discover_runs()

    print("[OK] Struktur dataset valid")
    print("[OK] Total run:", len(run_paths))

    # ===================================
    # 2. VALIDASI FILE SETIAP RUN
    # ===================================

    print(
        "[Pipeline] Memvalidasi tujuh file "
        "pada setiap run..."
    )

    valid_run_count = 0

    for run_path in run_paths:
        validate_run_files(run_path)
        valid_run_count += 1

    print("[OK] Seluruh file dataset valid")
    print(
        "[OK] Run yang berhasil divalidasi:",
        valid_run_count,
    )

    # Membuat folder output sebelum menyimpan laporan
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ===================================
    # 3. RUN-LEVEL QUALITY AUDIT
    # ===================================

    print(
        "\n[Pipeline] Mengaudit kualitas "
        "seluruh run..."
    )

    quality_report = audit_all_runs(
        run_paths
    )

    quality_report_path = (
        OUTPUT_ROOT
        / "timeseries_quality_report.csv"
    )

    quality_report.to_csv(
        quality_report_path,
        index=False,
    )

    problem_rows = quality_report[
        quality_report["status"] != "valid"
    ]

    print(
        "\n========== QUALITY SUMMARY =========="
    )

    print(
        quality_report[
            "status"
        ].value_counts()
    )

    print(
        "\nJumlah record audit:",
        len(quality_report),
    )

    print(
        "Jumlah record bermasalah:",
        len(problem_rows),
    )

    print(
        "Laporan disimpan di:",
        quality_report_path,
    )

    # ===================================
    # 4. COLUMN-LEVEL MISSING AUDIT
    # ===================================

    print(
        "\n[Pipeline] Mengaudit missing value "
        "per kolom..."
    )

    missing_column_report = (
        audit_all_missing_columns(
            run_paths
        )
    )

    missing_column_report_path = (
        OUTPUT_ROOT
        / "missing_by_column_report.csv"
    )

    missing_column_report.to_csv(
        missing_column_report_path,
        index=False,
    )

    print(
        "\n========== MISSING COLUMN SUMMARY =========="
    )

    if missing_column_report.empty:
        print(
            "Tidak ditemukan missing value"
        )

    else:
        print(
            missing_column_report.groupby(
                "data_name"
            )["missing_count"].sum()
        )

        print(
            "\nJumlah kombinasi kolom bermasalah:",
            len(missing_column_report),
        )

    print(
        "Laporan disimpan di:",
        missing_column_report_path,
    )

    # ===================================
    # 5.2 RAW TRACE STRUCTURAL AUDIT
    # ===================================

    print(
        "\n[Pipeline] Mengaudit struktur raw trace "
        "seluruh run..."
    )

    trace_structure_report = (
        audit_all_trace_structures(
            run_paths
        )
    )

    trace_structure_report_path = (
        OUTPUT_ROOT
        / "raw_trace_structure_report.csv"
    )

    trace_structure_report.to_csv(
        trace_structure_report_path,
        index=False,
    )

    print(
        "\n========== RAW TRACE AUDIT SUMMARY =========="
    )

    print(
        trace_structure_report["status"]
        .value_counts(dropna=False)
    )

    print(
        "Jumlah run yang diaudit:",
        len(trace_structure_report),
    )

    print(
        "Laporan disimpan di:",
        trace_structure_report_path,
    )

    # ===================================
    # 5.3 ROOT SPAN AND REQUEST IDENTITY
    # ===================================

    print(
        "\n[Pipeline] Mengaudit root span "
        "dan identitas request seluruh run..."
    )

    root_identity_report = (
        audit_all_root_span_identities(
            run_paths
        )
    )

    root_identity_report_path = (
        OUTPUT_ROOT
        / "root_span_identity_report.csv"
    )

    root_identity_report.to_csv(
        root_identity_report_path,
        index=False,
    )

    print(
        "\n========== ROOT IDENTITY SUMMARY =========="
    )

    print(
        root_identity_report["status"]
        .value_counts(dropna=False)
    )

    print(
        "Jumlah run yang diaudit:",
        len(root_identity_report),
    )

    print(
        "Laporan disimpan di:",
        root_identity_report_path,
    )

    if len(root_identity_report) != len(
        run_paths
    ):
        raise RuntimeError(
            "Jumlah hasil root identity audit "
            "tidak sama dengan jumlah run"
        )

    audit_error_mask = (
        root_identity_report["status"]
        .astype(str)
        .str.startswith("audit_error")
    )

    if audit_error_mask.any():
        error_rows = root_identity_report.loc[
            audit_error_mask,
            [
                "scenario",
                "run_number",
                "status",
            ],
        ]

        print(
            "\nRun yang gagal diaudit:"
        )

        print(
            error_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Terdapat run yang gagal menjalankan "
            "root identity audit"
        )

    if not (
        root_identity_report[
            "identity_balance_valid"
        ]
        .fillna(False)
        .all()
    ):
        raise RuntimeError(
            "Terdapat perhitungan identitas "
            "trace yang tidak seimbang"
        )

    print(
        "[OK] Root identity audit "
        "berhasil untuk seluruh run"
    )

    # ===================================
    # 5.4 SERVICE AND OPERATION AUDIT
    # ===================================

    print(
        "\n[Pipeline] Mengaudit service dan "
        "operation pada root span..."
    )

    service_operation_report = (
        audit_all_root_service_operations(
            run_paths
        )
    )

    service_operation_report_path = (
        OUTPUT_ROOT
        / "root_service_operation_report.csv"
    )

    service_operation_report.to_csv(
        service_operation_report_path,
        index=False,
    )

    print(
        "\n========== SERVICE-OPERATION SUMMARY =========="
    )

    print(
        "Jumlah baris laporan:",
        len(service_operation_report),
    )

    audited_run_count = (
        service_operation_report[
            [
                "scenario",
                "run_number",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Jumlah run yang diaudit:",
        audited_run_count,
    )

    print(
        "Laporan disimpan di:",
        service_operation_report_path,
    )

    service_operation_error_mask = (
        service_operation_report["status"]
        .astype(str)
        .str.startswith("audit_error")
    )

    if service_operation_error_mask.any():
        error_rows = (
            service_operation_report.loc[
                service_operation_error_mask,
                [
                    "scenario",
                    "run_number",
                    "error_message",
                ],
            ]
        )

        print(
            "\nRun yang gagal diaudit:"
        )

        print(
            error_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Terdapat run yang gagal menjalankan "
            "service-operation audit"
        )

    if audited_run_count != len(
        run_paths
    ):
        raise RuntimeError(
            "Jumlah run pada service-operation "
            "report tidak sama dengan jumlah run"
        )

    valid_service_operation_report = (
        service_operation_report.loc[
            ~service_operation_error_mask
        ]
    )

    service_operation_root_totals = (
        valid_service_operation_report
        .groupby(
            [
                "scenario",
                "run_number",
            ],
            as_index=False,
        )["root_count"]
        .sum()
        .rename(
            columns={
                "root_count": (
                    "service_operation_root_count"
                )
            }
        )
    )

    root_count_comparison = (
        root_identity_report[
            [
                "scenario",
                "run_number",
                "unique_root_span_count",
            ]
        ]
        .merge(
            service_operation_root_totals,
            on=[
                "scenario",
                "run_number",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    root_count_mismatch = (
        root_count_comparison[
            root_count_comparison[
                "unique_root_span_count"
            ]
            != root_count_comparison[
                "service_operation_root_count"
            ]
        ]
    )

    if not root_count_mismatch.empty:
        print(
            "\nPerbedaan jumlah root:"
        )

        print(
            root_count_mismatch.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Jumlah root Fase 5.4 tidak "
            "konsisten dengan Fase 5.3"
        )

    root_share_totals = (
        valid_service_operation_report
        .groupby(
            [
                "scenario",
                "run_number",
            ]
        )["root_share"]
        .sum()
    )

    invalid_root_share = (
        (root_share_totals - 1.0)
        .abs()
        > 1e-9
    )

    if invalid_root_share.any():
        raise RuntimeError(
            "Jumlah root_share per run "
            "tidak sama dengan 1.0"
        )

    print(
        "[OK] Service-operation audit "
        "berhasil untuk seluruh run"
    )

    print(
        "[OK] Jumlah root Fase 5.4 "
        "konsisten dengan Fase 5.3"
    )

    # ===================================
    # 5.5 STATUS CODE AUDIT
    # ===================================

    print(
        "\n[Pipeline] Mengaudit status code "
        "seluruh span..."
    )

    span_status_report = (
        audit_all_span_status_codes(
            run_paths
        )
    )

    span_status_report_path = (
        OUTPUT_ROOT
        / "span_status_code_report.csv"
    )

    span_status_report.to_csv(
        span_status_report_path,
        index=False,
    )

    print(
        "\n========== STATUS CODE SUMMARY =========="
    )

    status_code_summary = (
        span_status_report
        .groupby(
            [
                "status_code",
                "status_category",
            ],
            as_index=False,
            dropna=False,
        )["span_count"]
        .sum()
        .sort_values(
            "span_count",
            ascending=False,
        )
    )

    print(
        status_code_summary.to_string(
            index=False
        )
    )

    status_audited_run_count = (
        span_status_report[
            [
                "scenario",
                "run_number",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Jumlah baris laporan:",
        len(span_status_report),
    )

    print(
        "Jumlah run yang diaudit:",
        status_audited_run_count,
    )

    print(
        "Laporan disimpan di:",
        span_status_report_path,
    )

    status_audit_error_mask = (
        span_status_report["audit_status"]
        .astype(str)
        .str.startswith("audit_error")
    )

    if status_audit_error_mask.any():
        error_rows = span_status_report.loc[
            status_audit_error_mask,
            [
                "scenario",
                "run_number",
                "error_message",
            ],
        ]

        print(
            "\nRun yang gagal diaudit:"
        )

        print(
            error_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Terdapat run yang gagal menjalankan "
            "status code audit"
        )

    if status_audited_run_count != len(
        run_paths
    ):
        raise RuntimeError(
            "Jumlah run pada status code report "
            "tidak sama dengan jumlah run"
        )

    valid_span_status_report = (
        span_status_report.loc[
            ~status_audit_error_mask
        ]
    )

    status_span_totals = (
        valid_span_status_report
        .groupby(
            [
                "scenario",
                "run_number",
            ],
            as_index=False,
        )["span_count"]
        .sum()
        .rename(
            columns={
                "span_count": (
                    "status_audit_span_count"
                )
            }
        )
    )

    span_count_comparison = (
        trace_structure_report[
            [
                "scenario",
                "run_number",
                "unique_span_count",
            ]
        ]
        .merge(
            status_span_totals,
            on=[
                "scenario",
                "run_number",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    span_count_mismatch = (
        span_count_comparison[
            span_count_comparison[
                "unique_span_count"
            ]
            != span_count_comparison[
                "status_audit_span_count"
            ]
        ]
    )

    if not span_count_mismatch.empty:
        print(
            "\nPerbedaan jumlah span:"
        )

        print(
            span_count_mismatch.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Jumlah span Fase 5.5 tidak "
            "konsisten dengan Fase 5.2"
        )

    status_span_share_totals = (
        valid_span_status_report
        .groupby(
            [
                "scenario",
                "run_number",
            ]
        )["span_share"]
        .sum()
    )

    invalid_status_span_share = (
        (status_span_share_totals - 1.0)
        .abs()
        > 1e-9
    )

    if invalid_status_span_share.any():
        raise RuntimeError(
            "Jumlah span_share per run "
            "tidak sama dengan 1.0"
        )

    print(
        "[OK] Status code audit "
        "berhasil untuk seluruh run"
    )

    print(
        "[OK] Jumlah span Fase 5.5 "
        "konsisten dengan Fase 5.2"
    )

    

#=====================================#
# PREPROCESSING ORCHESTRATOR
#=====================================#
def main():
    """
    Menjalankan seluruh fase preprocessing yang
    sudah diimplementasikan secara berurutan.

    Saat ini pipeline lengkap mencakup Fase 1-11.
    """

    print(
        "\n========== PREPROCESSING PIPELINE "
        "FASE 1-11 =========="
    )

    run_phases_1_to_5()
    run_phase_6()
    run_phase_7()
    run_phase_8()
    run_phase_9()
    run_phase_10()
    run_phase_11()

    print(
        "\n[OK] Seluruh preprocessing Fase 1-11 "
        "selesai dijalankan"
    )


def parse_execution_arguments():
    """
    --phase all : menjalankan seluruh pipeline Fase 1-11
    --phase 1-5 : hanya menjalankan validasi dan audit
    --phase 6   : hanya menjalankan Derived Severity
                  dari output Fase 5 yang sudah tersimpan
    --phase 7   : hanya menjalankan observation-window
                  extraction dan time-series alignment
    --phase 8   : hanya menjalankan feature cleaning
                  dan run eligibility dari output Fase 7
    --phase 9   : hanya menjalankan feature extraction
                  dari output Fase 8
    --phase 10  : hanya menjalankan perakitan dataset
                  training dari output Fase 9
    --phase 11  : hanya menjalankan pemeriksaan akhir
                  dan readiness report dari output Fase 10
    """

    parser = argparse.ArgumentParser(
        description=(
            "RE2-OB preprocessing pipeline"
        )
    )

    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "1-5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
        ],
        default="all",
        help=(
            "Gunakan 'all' untuk Fase 1-11, "
            "'1-5' untuk validasi dan audit, "
            "'6' untuk Derived Severity, "
            "'7' untuk alignment, "
            "'8' untuk feature cleaning, "
            "'9' untuk feature extraction, "
            "'10' untuk perakitan dataset training, atau "
            "'11' untuk pemeriksaan akhir."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = (
        parse_execution_arguments()
    )

    if arguments.phase == "1-5":
        run_phases_1_to_5()
    elif arguments.phase == "6":
        run_phase_6()
    elif arguments.phase == "7":
        run_phase_7()
    elif arguments.phase == "8":
        run_phase_8()
    elif arguments.phase == "9":
        run_phase_9()
    elif arguments.phase == "10":
        run_phase_10()
    elif arguments.phase == "11":
        run_phase_11()
    else:
        main()
