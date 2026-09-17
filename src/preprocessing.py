from pathlib import Path
import json
import pandas as pd
from collections import defaultdict, Counter



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

USER_FACING_ROOT_SERVICE = "frontendservice"
USER_FACING_ROOT_OPERATION = "frontend"


TRACE_CHUNK_SIZE = 100_000

#=========== DERIVED SEVERITY CONFIGS ===========#
LABEL_AGGREGATION_CHUNK_SIZE = 200_000

LABEL_POLICY = {
    #Konfigurasi Policy (jika sudah fix bisa di ganti nama policy version dan is frozen bisa di ganti ke true)
    "policy_version": "candidate_v1",
    "is_frozen": False,

    #Kandidat target SLO dan Latency Treshold (bisa di ganti setelah konfirmasi nanti)
    "slo_target": 0.99, #99% SLO
    "latency_threshold": 750.0, #Treshold dari latency (menentukan bahwa request itu buruk atau tidak)


    #LOW : burn rate < 1
    #MEDIUM : 1 <= burn rate < 6
    #HIGH : burn rate >= 6
    "medium_burn_rate": 1.0,
    "high_burn_rate": 6.0,


    "observation_window": "incident",
    "bad_request_rule": (
        "nonzero_status_or_latency_exceeded"
    ),
}

LABEL_SENSITIVITY = {
    "slo_target": [
        0.99,
        0.995,
        0.999,
    ],
    "latenct_treshold_ms" : [
        500.0,
        750.0,
        1000.0,
    ],
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


# Shared Trace Context for Fase 5.6-5.8

LABEL_SOURCE_REQUEST_COLUMNS = [
    "scenario",
    "service",
    "fault_type",
    "run_number",
    "trace_id",
    "root_span_id",
    "request_timestamp",
    "inject_time",
    "window_type",
    "duration_raw_us",
    "duration_ms",
    "has_nonzero_status",
    "nonzero_status_codes",
]


def _collect_trace_label_context(run_data):
    """Membaca raw trace sekali untuk dipakai bersama Fase 5.6-5.8."""

    seen_span_ids = set()
    all_trace_ids = set()
    root_span_ids_by_trace = defaultdict(set)
    frontend_roots_by_trace = defaultdict(dict)
    nonzero_codes_by_trace = defaultdict(set)

    duration_values = []
    frontend_duration_values = []
    clock_pair_count = 0
    clock_match_count = 0
    max_clock_difference_ms = 0
    trace_start_time = None
    trace_end_time = None

    for _, chunk in iterate_trace_chunks(run_data["traces_path"]):
        rows = pd.DataFrame({
            "trace_id": chunk["traceID"].astype("string").str.strip(),
            "span_id": chunk["spanID"].astype("string").str.strip(),
            "parent_span_id": chunk["parentSpanID"].astype("string").str.strip(),
            "service_name": chunk["serviceName"].astype("string").str.strip(),
            "operation_name": chunk["operationName"].astype("string").str.strip(),
            "start_time": chunk["startTime"],
            "start_time_millis": chunk["startTimeMillis"],
            "timestamp": chunk["timestamp"],
            "duration": chunk["duration"],
            "status_code": chunk["statusCode"],
        })

        valid_span = rows["span_id"].notna() & rows["span_id"].ne("")
        rows = rows.loc[valid_span]
        rows = rows.loc[~rows["span_id"].isin(seen_span_ids)]
        rows = rows.drop_duplicates("span_id", keep="first")

        if rows.empty:
            continue

        seen_span_ids.update(rows["span_id"].tolist())

        valid_timestamps = rows["timestamp"].dropna()
        if not valid_timestamps.empty:
            chunk_start = valid_timestamps.min()
            chunk_end = valid_timestamps.max()
            trace_start_time = (
                chunk_start
                if trace_start_time is None
                else min(trace_start_time, chunk_start)
            )
            trace_end_time = (
                chunk_end
                if trace_end_time is None
                else max(trace_end_time, chunk_end)
            )

        valid_clock = (
            rows["start_time"].notna()
            & rows["start_time_millis"].notna()
        )
        if valid_clock.any():
            derived_ms = (
                rows.loc[valid_clock, "start_time"].astype("int64") // 1000
            )
            recorded_ms = rows.loc[
                valid_clock,
                "start_time_millis",
            ].astype("int64")
            clock_difference = (derived_ms - recorded_ms).abs()
            clock_pair_count += int(valid_clock.sum())
            clock_match_count += int((clock_difference == 0).sum())
            max_clock_difference_ms = max(
                max_clock_difference_ms,
                int(clock_difference.max()),
            )

        valid_duration = rows["duration"].notna() & rows["duration"].ge(0)
        duration_values.extend(
            rows.loc[valid_duration, "duration"].astype(float).tolist()
        )

        valid_trace = rows["trace_id"].notna() & rows["trace_id"].ne("")
        all_trace_ids.update(rows.loc[valid_trace, "trace_id"].tolist())

        nonzero_status = (
            valid_trace
            & rows["status_code"].notna()
            & rows["status_code"].ne(0)
        )
        for trace_id, codes in rows.loc[nonzero_status].groupby("trace_id"):
            nonzero_codes_by_trace[trace_id].update(
                float(code) for code in codes["status_code"].unique()
            )

        has_parent = (
            rows["parent_span_id"].notna()
            & rows["parent_span_id"].ne("")
        )
        root_rows = rows.loc[valid_trace & ~has_parent]

        for trace_id, spans in root_rows.groupby("trace_id"):
            root_span_ids_by_trace[trace_id].update(
                spans["span_id"].tolist()
            )

        frontend_mask = (
            root_rows["service_name"].eq(USER_FACING_ROOT_SERVICE).fillna(False)
            & root_rows["operation_name"].eq(USER_FACING_ROOT_OPERATION).fillna(False)
        )
        frontend_rows = root_rows.loc[frontend_mask]

        frontend_duration_values.extend(
            frontend_rows.loc[
                frontend_rows["duration"].notna()
                & frontend_rows["duration"].ge(0),
                "duration",
            ].astype(float).tolist()
        )

        for row in frontend_rows.itertuples(index=False):
            frontend_roots_by_trace[row.trace_id][row.span_id] = {
                "root_span_id": row.span_id,
                "timestamp": row.timestamp,
                "duration": row.duration,
            }

    return {
        "seen_span_ids": seen_span_ids,
        "all_trace_ids": all_trace_ids,
        "root_span_ids_by_trace": root_span_ids_by_trace,
        "frontend_roots_by_trace": frontend_roots_by_trace,
        "nonzero_codes_by_trace": nonzero_codes_by_trace,
        "duration_series": pd.Series(duration_values, dtype="float64"),
        "frontend_duration_series": pd.Series(
            frontend_duration_values,
            dtype="float64",
        ),
        "clock_pair_count": clock_pair_count,
        "clock_match_count": clock_match_count,
        "max_clock_difference_ms": max_clock_difference_ms,
        "trace_start_time": trace_start_time,
        "trace_end_time": trace_end_time,
    }


def _resolve_trace_label_context(run_data, trace_context):
    """Memakai context yang sudah ada atau membaca raw trace satu kali."""

    return (
        trace_context
        if trace_context is not None
        else _collect_trace_label_context(run_data)
    )


# 5.6 Duration Unit Audit
def audit_duration_unit(run_data, trace_context=None):
    """Fase 5.6: memvalidasi bahwa kolom duration memakai mikrodetik."""

    context = _resolve_trace_label_context(run_data, trace_context)
    duration_ms = context["duration_series"] / 1000.0
    frontend_duration_ms = context["frontend_duration_series"] / 1000.0

    def quantile(series, value):
        return None if series.empty else float(series.quantile(value))

    clock_pair_count = context["clock_pair_count"]
    clock_match_count = context["clock_match_count"]
    clock_mismatch_count = clock_pair_count - clock_match_count

    return {
        "unique_span_count": len(context["seen_span_ids"]),
        "clock_pair_count": clock_pair_count,
        "clock_match_count": clock_match_count,
        "clock_mismatch_count": clock_mismatch_count,
        "clock_match_ratio": (
            clock_match_count / clock_pair_count
            if clock_pair_count
            else 0.0
        ),
        "max_clock_difference_ms": context["max_clock_difference_ms"],
        "duration_count": len(context["duration_series"]),
        "duration_ms_p50": quantile(duration_ms, 0.50),
        "duration_ms_p95": quantile(duration_ms, 0.95),
        "duration_ms_p99": quantile(duration_ms, 0.99),
        "duration_ms_max": (
            None if duration_ms.empty else float(duration_ms.max())
        ),
        "frontend_duration_ms_p50": quantile(frontend_duration_ms, 0.50),
        "frontend_duration_ms_p95": quantile(frontend_duration_ms, 0.95),
        "frontend_duration_ms_p99": quantile(frontend_duration_ms, 0.99),
        "frontend_duration_ms_max": (
            None
            if frontend_duration_ms.empty
            else float(frontend_duration_ms.max())
        ),
        "duration_unit_evidence": (
            "supports_microseconds"
            if clock_pair_count and clock_mismatch_count == 0
            else "needs_review"
        ),
    }


# 5.7 Incident Window Coverage Audit
def audit_incident_window_coverage(run_data, trace_context=None):
    """Fase 5.7: memeriksa cakupan baseline dan incident window."""

    context = _resolve_trace_label_context(run_data, trace_context)
    inject_time = run_data["inject_time"]
    baseline_start = inject_time - pd.Timedelta(
        minutes=OBSERVATION_BEFORE_MINUTES
    )
    incident_end = inject_time + pd.Timedelta(
        minutes=OBSERVATION_AFTER_MINUTES
    )
    trace_start_time = context["trace_start_time"]
    trace_end_time = context["trace_end_time"]

    return {
        "trace_start_time": trace_start_time,
        "trace_end_time": trace_end_time,
        "baseline_start": baseline_start,
        "inject_time": inject_time,
        "incident_end": incident_end,
        "window_covered": bool(
            trace_start_time is not None
            and trace_end_time is not None
            and trace_start_time <= baseline_start
            and trace_end_time >= incident_end
        ),
    }


# 5.8 Label Source Eligibility
def audit_label_source_eligibility(
    run_data,
    trace_context=None,
    duration_audit=None,
    window_audit=None,
):
    """Fase 5.8: memilih request frontend yang layak sebagai sumber label."""

    context = _resolve_trace_label_context(run_data, trace_context)
    duration_audit = duration_audit or audit_duration_unit(run_data, context)
    window_audit = window_audit or audit_incident_window_coverage(
        run_data,
        context,
    )

    root_counts = {
        trace_id: len(span_ids)
        for trace_id, span_ids in context["root_span_ids_by_trace"].items()
    }
    traces_with_root = set(root_counts)
    single_root_ids = {
        trace_id for trace_id, count in root_counts.items() if count == 1
    }
    multiple_root_ids = {
        trace_id for trace_id, count in root_counts.items() if count > 1
    }
    frontend_single_root_ids = {
        trace_id
        for trace_id in single_root_ids
        if len(context["frontend_roots_by_trace"].get(trace_id, {})) == 1
    }

    request_records = []
    invalid_frontend_request_count = 0
    outside_window_request_count = 0
    inject_time = window_audit["inject_time"]
    baseline_start = window_audit["baseline_start"]
    incident_end = window_audit["incident_end"]

    for trace_id in frontend_single_root_ids:
        root = next(
            iter(context["frontend_roots_by_trace"][trace_id].values())
        )
        timestamp = root["timestamp"]
        duration = root["duration"]

        if pd.isna(timestamp) or pd.isna(duration) or float(duration) < 0:
            invalid_frontend_request_count += 1
            continue

        if baseline_start <= timestamp < inject_time:
            window_type = "baseline"
        elif inject_time <= timestamp < incident_end:
            window_type = "incident"
        else:
            outside_window_request_count += 1
            continue

        nonzero_codes = sorted(
            context["nonzero_codes_by_trace"].get(trace_id, set())
        )
        formatted_codes = "|".join(
            str(int(code)) if code.is_integer() else str(code)
            for code in nonzero_codes
        )
        request_records.append({
            "scenario": run_data["scenario"],
            "service": run_data["service"],
            "fault_type": run_data["fault_type"],
            "run_number": run_data["run_number"],
            "trace_id": trace_id,
            "root_span_id": root["root_span_id"],
            "request_timestamp": timestamp,
            "inject_time": inject_time,
            "window_type": window_type,
            "duration_raw_us": float(duration),
            "duration_ms": float(duration) / 1000.0,
            "has_nonzero_status": bool(nonzero_codes),
            "nonzero_status_codes": formatted_codes,
        })

    request_report = pd.DataFrame(
        request_records,
        columns=LABEL_SOURCE_REQUEST_COLUMNS,
    )
    window_counts = request_report["window_type"].value_counts()
    baseline_request_count = int(window_counts.get("baseline", 0))
    incident_request_count = int(window_counts.get("incident", 0))
    incident_error_candidate_count = int(
        (
            request_report["window_type"].eq("incident")
            & request_report["has_nonzero_status"].eq(True)
        ).sum()
    )

    eligibility_audit = {
        "unique_trace_count": len(context["all_trace_ids"]),
        "zero_root_trace_count": len(
            context["all_trace_ids"] - traces_with_root
        ),
        "multiple_root_trace_count": len(multiple_root_ids),
        "single_root_trace_count": len(single_root_ids),
        "frontend_single_root_trace_count": len(frontend_single_root_ids),
        "non_frontend_single_root_trace_count": len(
            single_root_ids - frontend_single_root_ids
        ),
        "invalid_frontend_request_count": invalid_frontend_request_count,
        "outside_window_request_count": outside_window_request_count,
        "baseline_request_count": baseline_request_count,
        "incident_request_count": incident_request_count,
        "incident_error_candidate_count": incident_error_candidate_count,
        "label_source_eligible": bool(
            window_audit["window_covered"]
            and duration_audit["duration_unit_evidence"]
            == "supports_microseconds"
            and incident_request_count > 0
        ),
    }
    return eligibility_audit, request_report


def audit_label_source_run(run_data):
    """Mengorkestrasi Fase 5.6-5.8 dengan satu pembacaan raw trace."""

    context = _collect_trace_label_context(run_data)
    duration_audit = audit_duration_unit(run_data, context)
    window_audit = audit_incident_window_coverage(run_data, context)
    eligibility_audit, request_report = audit_label_source_eligibility(
        run_data,
        context,
        duration_audit,
        window_audit,
    )

    quality_issues = []
    if duration_audit["duration_count"] != duration_audit["unique_span_count"]:
        quality_issues.append("incomplete_duration")
    if duration_audit["clock_pair_count"] != duration_audit["unique_span_count"]:
        quality_issues.append("incomplete_clock_pair")
    if duration_audit["clock_mismatch_count"]:
        quality_issues.append("clock_unit_mismatch")
    if not window_audit["window_covered"]:
        quality_issues.append("incomplete_observation_window")
    if eligibility_audit["incident_request_count"] == 0:
        quality_issues.append("no_incident_request")

    audit_record = {
        "scenario": run_data["scenario"],
        "service": run_data["service"],
        "fault_type": run_data["fault_type"],
        "run_number": run_data["run_number"],
        **duration_audit,
        **window_audit,
        **eligibility_audit,
        "quality_status": (
            "valid" if not quality_issues else "|".join(quality_issues)
        ),
    }
    return audit_record, request_report


def audit_all_label_sources(run_paths):
    """Menjalankan gabungan Fase 5.6-5.8 untuk seluruh run."""

    audit_records = []
    request_reports = []
    total_runs = len(run_paths)

    for index, run_path in enumerate(run_paths, start=1):
        try:
            run_data = load_validated_run(run_path)
            audit_record, request_report = audit_label_source_run(run_data)
            audit_record["execution_status"] = "valid"
            audit_record["error_message"] = ""
            audit_records.append(audit_record)
            if not request_report.empty:
                request_reports.append(request_report)
            print(
                f"[OK] Label-source audit {index}/{total_runs}: "
                f"{run_data['scenario']}/run-{run_data['run_number']}"
            )
        except Exception as error:
            scenario = run_path.parent.name
            service_name, fault_type = scenario.rsplit("_", maxsplit=1)
            audit_records.append({
                "scenario": scenario,
                "service": service_name,
                "fault_type": fault_type,
                "run_number": run_path.name,
                "execution_status": "audit_error",
                "error_message": str(error),
            })
            print(
                f"[ERROR] Label-source audit {index}/{total_runs}: "
                f"{run_path}: {error}"
            )

    request_report = (
        pd.concat(request_reports, ignore_index=True)
        if request_reports
        else pd.DataFrame(columns=LABEL_SOURCE_REQUEST_COLUMNS)
    )
    return pd.DataFrame(audit_records), request_report


def validate_label_source_results(
    audit_report,
    request_report,
    trace_structure_report,
    expected_run_count,
):
    """Memvalidasi output gabungan Fase 5.6-5.8."""

    if len(audit_report) != expected_run_count:
        raise RuntimeError("Jumlah hasil label-source audit tidak sesuai")

    execution_errors = audit_report["execution_status"].ne("valid")
    if execution_errors.any():
        print(
            audit_report.loc[
                execution_errors,
                ["scenario", "run_number", "error_message"],
            ].to_string(index=False)
        )
        raise RuntimeError("Terdapat run yang gagal pada Fase 5.6-5.8")

    span_comparison = trace_structure_report[
        ["scenario", "run_number", "unique_span_count"]
    ].merge(
        audit_report[
            ["scenario", "run_number", "unique_span_count"]
        ],
        on=["scenario", "run_number"],
        suffixes=("_phase_5_2", "_phase_5_6_8"),
        validate="one_to_one",
    )
    span_mismatch = span_comparison[
        span_comparison["unique_span_count_phase_5_2"]
        != span_comparison["unique_span_count_phase_5_6_8"]
    ]
    if not span_mismatch.empty:
        raise RuntimeError("Jumlah span Fase 5.6-5.8 tidak konsisten")

    if not audit_report["duration_unit_evidence"].eq(
        "supports_microseconds"
    ).all():
        raise RuntimeError("Satuan duration belum konsisten pada seluruh run")

    expected_requests = int(
        audit_report[
            ["baseline_request_count", "incident_request_count"]
        ].sum().sum()
    )
    if len(request_report) != expected_requests:
        raise RuntimeError("Jumlah request candidate tidak konsisten")

    return True


# #===================================#
# # 6. DERIVED SEVERITY PROXY
# #===================================#

# # 6.1 Label Policy Validation
# def validate_label_policy(policy, sensitivity):
#     #untuk memastikan konfigurasi slo, treshold latency dll valid dengan konfig di tahap 1

#     required_keys = {
#         "policy_version",
#         "is_frozen",
#         "slo_target",
#         "latency_treshold_ms",
#         "medium_burn_rate",
#         "high_burn_rate",
#         "observation_window",
#         "bad_request_rule",
#     }

#     missing_keys = required_keys - set(policy)

#     if missing_keys:
#         raise ValueError(
#             "Konfigurasi label belum lengkap: "
#             f"{sorted(missing_keys)}"
#         )

#     slo_target = float(policy["slo_target"])
#     latency_treshold = float(policy["latency_treshold_ms"])
#     medium_treshold = float(policy["medium_burn_rate"])
#     high_treshold = float(policy["high_burn_rate"])

    
    


#===================================#
# 8. FEATURE DATA CLEANING 
#    AND RUN ELIGIBILITY
#===================================#

#8.1 Missing Value Classification
def classify_feature_family(data_name, column_name):
    #Mengelompokkan kolom berdasarkan arti datanya

    normalized_name = column_name.lower()

    if data_name == "trace_latency":
        return "trace_latency"

    if data_name == "trace_error":
        return "trace_error"

    if data_name == "logs":
        return "log_count"

    if data_name == "metrics":
        if normalized_name.endswith(RESOURCE_METRIC_SUFFIXES):
            return "continuous_resource"

        if "latency" in normalized_name:
            return "latency_signal"

        if normalized_name.endswith("_error"):
            return "error_signal"

        if normalized_name.endswith("_workload"):
            return "workload_signal"

    return "other_metric"

#8.2 Run Eligibility

#8.3 Data Cleaning

#8.4 Cleaning Log







#=====================================#    
def main():
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

    # ===================================
    # 5.6-5.8 DURATION, WINDOW, ELIGIBILITY
    # ===================================

    print(
        "\n[Pipeline] Mengaudit duration, observation "
        "window, dan label-source eligibility..."
    )

    (
        label_source_audit_report,
        label_source_request_report,
    ) = audit_all_label_sources(run_paths)

    label_source_audit_path = (
        OUTPUT_ROOT / "trace_label_source_audit_report.csv"
    )
    label_source_request_path = (
        OUTPUT_ROOT / "label_source_request_candidates.csv"
    )

    label_source_audit_report.to_csv(
        label_source_audit_path,
        index=False,
    )
    label_source_request_report.to_csv(
        label_source_request_path,
        index=False,
    )

    validate_label_source_results(
        label_source_audit_report,
        label_source_request_report,
        trace_structure_report,
        len(run_paths),
    )

    print(
        "\n========== LABEL-SOURCE AUDIT SUMMARY =========="
    )
    print(
        "Duration unit evidence:\n",
        label_source_audit_report[
            "duration_unit_evidence"
        ].value_counts(dropna=False),
    )
    print(
        "Window covered:\n",
        label_source_audit_report[
            "window_covered"
        ].value_counts(dropna=False),
    )
    print(
        "Label-source eligible:\n",
        label_source_audit_report[
            "label_source_eligible"
        ].value_counts(dropna=False),
    )
    print(
        "Request candidate per window:\n",
        label_source_request_report[
            "window_type"
        ].value_counts(dropna=False),
    )
    print(
        "Laporan audit disimpan di:",
        label_source_audit_path,
    )
    print(
        "Kandidat request disimpan di:",
        label_source_request_path,
    )
    print(
        "[OK] Fase 5.6-5.8 selesai dan "
        "konsisten dengan Fase 5.2"
    )


if __name__ == "__main__":
    main()
