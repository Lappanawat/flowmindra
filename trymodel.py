import joblib

obj = joblib.load("scaler.pkl")
print(type(obj))
print(obj)