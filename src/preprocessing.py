from pathlib import Path
import json
import pandas as pd


#===================================#
# 1. DATASET CONFIGURATION 
#===================================#   
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = {
    PROJECT_ROOT
    / "data"
    / "RE2-OB"
    / "RE2-OB"
}


OUTPUT_ROOT = {
    PROJECT_ROOT
    / "data"
    / "processed"
}


EXPETED_SERVICES = {
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

EXPECTED_RUNS = ("1", "2", "3")

TIMESERIES_FILES = {
    "metrics": "simple_metrics.csv",
    "trace_latency": "tracets_lat.csv",
    "trace_error": "tracets_err.csv",
}


REQUIRED_FILES = {
    "simple_metrics.csv",
    "logts.csv",
    "tracets_lat.csv",
    "tracets_err.csv",
    "traces.csv"
    "cluster_info.json",
    "inject_time.txt"
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
