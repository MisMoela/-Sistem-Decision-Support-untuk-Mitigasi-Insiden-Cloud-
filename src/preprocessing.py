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

    print("[INFO] Contoh lima run:")

    for run_path in run_paths[:5]:
        print("-", run_path)


if __name__ == "__main__":
    main()