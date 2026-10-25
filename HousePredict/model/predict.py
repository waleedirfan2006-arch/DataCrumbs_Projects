import json, sys, joblib, pandas as pd

# Load the pre‑trained pipeline (path relative to this script)
model_path = "../model/antigravity_pipeline.joblib"
model = joblib.load(model_path)

# Read JSON from stdin
input_data = json.load(sys.stdin)

# Convert to DataFrame (single record)
df = pd.DataFrame([input_data])

# Predict
prediction = model.predict(df)[0]

# Output prediction as JSON
print(json.dumps({"prediction": prediction}))
