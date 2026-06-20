import joblib as jb

try:
    model_data = jb.load("lr_heart.pkl")
    print(f"lr_heart.pkl type: {type(model_data)}")
    print(f"lr_heart.pkl content: {model_data}")
    if isinstance(model_data, list):
        print(f"  List length: {len(model_data)}")
        for i, item in enumerate(model_data):
            print(f"  Item {i} type: {type(item)}, value: {item}")
    elif isinstance(model_data, dict):
        print(f"  Dict keys: {model_data.keys()}")
        for key, value in model_data.items():
            print(f"  Key '{key}' type: {type(value)}")
except Exception as e:
    print(f"Error loading lr_heart.pkl: {e}")

print("\n" + "="*40 + "\n")

# Check scaler.pkl
try:
    scaler_data = jb.load("scaler.pkl")
    print(f"scaler.pkl type: {type(scaler_data)}")
    print(f"Has 'transform' method: {hasattr(scaler_data, 'transform')}")
except Exception as e:
    print(f"Error loading scaler.pkl: {e}")

print("\n" + "="*40 + "\n")

# Check columns.pkl
try:
    columns_data = jb.load("columns.pkl")
    print(f"columns.pkl type: {type(columns_data)}")
    print(f"columns.pkl content: {columns_data}")
except Exception as e:
    print(f"Error loading columns.pkl: {e}")
