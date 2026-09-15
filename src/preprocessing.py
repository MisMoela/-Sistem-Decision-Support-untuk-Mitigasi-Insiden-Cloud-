from pathlib import Path
import json
import pandas as pd


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


#===================================#
# 3. LOADING RUN DATA
#===================================#

def load_timeseries(file_path):

    #Berfungsi untuk membaca satu file timeseries dan mengubah kolom time dari unix epoch seconds menjadi datetime UTC 
    #Fungsi ini digunakan oleh file (simple_metrics.csv, logts.csv, tracets_lat.csv, tracets_err.csv)

    dataframe = pd.read_csv(file_path)

    #Memastikan kolom waktu berupa angka
    dataframe["time"] = pd.to_numeric(
        dataframe["time"], 
        errors="raise",
    )

    #Memastikan seluruh kolom nilai berupa numerik
    value_columns = [
        column
        for column in dataframe.columns
        if column != "time"
    ]

    dataframe[value_columns] = dataframe[value_columns].apply(
        pd.to_numeric,
        errors = "raise",
    )

    #Jangan mengubah kolom time asli
    #Menambahkan kolom timestamp agar data asli tetap tersedia
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["time"],
        unit = "s",
        utc = True,
    )

    return dataframe


def load_run(run_path):

    #Berfungsi untuk memuat seluruh informasi yang dibutuhkan dari satu run
    #traces.csv tidak dimuat seluruhnya karena ukuran file nya sangat besar, fungsi ini hanya menyimpan pathnya

    run_path = Path(run_path)

    #validasi file sebelum dibaca
    validate_run_files(run_path)

    scenario_name = run_path.parent.name
    run_number = run_path.name

    service_name, fault_type = scenario_name.rsplit("_", maxsplit = 1, )

    #Memuat empat file timeseries
    timeseries_data = {}

    for data_name, file_name in TIMESERIES_FILES.items():
        file_path = run_path / file_name

        timeseries_data[data_name] = load_timeseries(file_path)


    #Memuat medatada semantic log
    cluster_path = run_path / "cluster_info.json"

    with open(cluster_path, encoding="utf-8") as file:
        cluster_info = json.load(file)


    #Memuat waktu inject
    inject_path = run_path / "inject_time.txt"

    inject_epoch = int(inject_path.read_text(encoding = "utf-8").strip())

    inject_time = pd.to_datetime(
        inject_epoch, unit = "s", utc = True
    )

    return {
        "scenario": scenario_name,
        "service": service_name,
        "fault_type": fault_type,
        "run_number": run_number,
        "run_path": run_path,
        "inject_epoch": inject_epoch,
        "inject_time": inject_time,
        "timeseries": timeseries_data,
        "traces_path": run_path / "traces.csv",
        "cluster_info": cluster_info,
    }



#===================================#
# 4. DATA QUALITY AUDIT
#===================================#

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
            run_data = load_run(run_path)
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



#===================================#
# 5. STRICT DATA VALIDATION
#===================================#


#VALIDASI DATA SETELAH LOAD 1 RUN 
def validate_loaded_run(run_data):
    #Memastikan data yang sudah dimuat memiliki isi dan rentang waktunya mencakup waktu injeksi

    inject_time = run_data["inject_time"]

    for data_name, dataframe in (
        run_data["timeseries"].items()
    ):
        if dataframe.empty:
            raise ValueError(
                f"Data {data_name} kosong pada "
                f"{run_data['run_path']}"
            )

        if dataframe["timestamp"].isna().any():
            raise ValueError(
                f"Timestamp tidak valid pada "
                f"{data_name}: {run_data['run_path']}"
            )

        if not dataframe["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Timestamp tidak berurutan pada"
                f"{data_name}: {run_data['run_path']}"
            )

        start_time = dataframe["timestamp"].min()
        end_time = dataframe["timestamp"].max()


        if not start_time <= inject_time <= end_time:
            raise ValueError(
                f"Waktu inject berada di luar rentang"
                f"{data_name}"
                f"Rentang={start_time} sampai {end_time}"
                f"Injeksi={inject_time}"
            )

        duplicate_count = int(
            dataframe["timestamp"].duplicated().sum()
        )

        missing_count = int(
            dataframe.isna().sum().sum()
        )

        if duplicate_count > 0:
            print(
                f"[WARNING!!] {data_name} memiliki "
                f"{duplicate_count} timestamp duplikasi"
            )

        if missing_count > 0:
            print(
                f"[WARNING!!] {data_name} memiliki "
                f"{missing_count} missing values"
            )

    return True




def load_all_runs(run_paths):

    #Memuat dan memvalidasi seluruh run secara berurutan/
    #Fungsi menggunakan yield agar hanya satu run berada di memori pada satu waktu

    total_runs = len(run_paths)

    for index, run_path in enumerate(
        run_paths, 
        start=1,
    ):
        try:
            #Load Satu run
            run_data = load_run(
                run_path
            )

            #Memvalidasi isi run yang sdah dimuat
            validate_loaded_run(
                run_data
            )

        except Exception as error:
            raise RuntimeError(
                f"Gagal Memuat atau memvalidasi "
                f"run {index}/{total_runs}: "
                f"{run_path}"
            ) from error

        print(
            f"[OK] {index}/{total_runs}: {run_path} "
            f"{run_data['scenario']}/"
            f"run-{run_data['run_number']}"
        )

        yield run_data



def show_run_summary(run_data):
    """
    Menampilkan ringkasan satu run tanpa mencetak
    seluruh dataset.
    """

    print("\n========== RUN SUMMARY ==========")
    print("Scenario    :", run_data["scenario"])
    print("Service     :", run_data["service"])
    print("Fault type  :", run_data["fault_type"])
    print("Run number  :", run_data["run_number"])
    print("Inject time :", run_data["inject_time"])

    for data_name, dataframe in (
        run_data["timeseries"].items()
    ):
        print(f"\n--- {data_name} ---")
        print("Shape :", dataframe.shape)
        print(
            "Range :",
            dataframe["timestamp"].min(),
            "sampai",
            dataframe["timestamp"].max(),
        )

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {"time", "timestamp"}
        ]

        preview_columns = [
            "timestamp",
            *feature_columns[:4],
        ]

        print(
            dataframe[
                preview_columns
            ].head(3).to_string(index=False)
        )

    trace_columns = [
        "traceID",
        "spanID",
        "parentSpanID",
        "serviceName",
        "operationName",
        "startTimeMillis",
        "duration",
        "statusCode",
    ]

    trace_preview = pd.read_csv(
        run_data["traces_path"],
        usecols=trace_columns,
        nrows=5,
    )

    print("\n--- traces preview ---")
    print(
        trace_preview.to_string(index=False)
    )








#=====================================#    
def main():
    print(
        "[Pipeline] Memvalidasi struktur dataset..."
    )

    run_paths = discover_runs()

    print(
        "[OK] Struktur dataset valid"
    )

    print(
        "[OK] Total run:",
        len(run_paths),
    )

    print(
        "[Pipeline] Memvalidasi Tujuh file pada setiap run..."
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

    print(
    "\n[Pipeline] Mengaudit kualitas "
    "seluruh run..."
    )

    quality_report = audit_all_runs(
        run_paths
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
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


if __name__ == "__main__":
    main()
