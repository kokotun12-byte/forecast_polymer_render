import joblib

files = [
    "models/hybrid_artifacts.pkl",
    "models/feature_history.pkl",
    "models/history_y.pkl",
    "models/history_X.pkl",
    "models/scaler.pkl",
]

for path in files:
    print(f"\n--- {path} ---")

    try:
        obj = joblib.load(path)
    except Exception as e:
        print("❌ Error loading:", e)
        continue

    print("type:", type(obj))

    if isinstance(obj, dict):
        print("keys:", list(obj.keys()))

    if hasattr(obj, "shape"):
        print("shape:", obj.shape)

    if hasattr(obj, "head"):
        try:
            print(obj.head())
        except:
            pass