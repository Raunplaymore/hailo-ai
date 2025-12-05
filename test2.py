
import pickle
from pathlib import Path
import pandas as pd
p = Path("data/golf_db/train_split_1.pkl")
print("pandas version:", pd.__version__)
with open(p, "rb") as f:
    try:
        obj = pickle.load(f)
        print("OK:", type(obj), getattr(obj, "shape", None))
    except Exception as e:
        print("ERROR:", type(e).__name__, e)

