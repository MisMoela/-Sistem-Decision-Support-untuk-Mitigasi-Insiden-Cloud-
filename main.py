import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
from google import genai
from google.genai import types
import json
import time

#==================================#
#LOAD & PREPROCESS DATA (Simulation)
#==================================#

print("[Pipeline] Langkah 1: Memuat dataset RCAEval (Online Boutique)...")

X_dummy = pd.DataFrame

