import joblib

# Ruta a tu modelo .joblib
MODEL_PATH = "../../../models/best_rf_model.joblib"

# Carga el modelo
model = joblib.load(MODEL_PATH)

# Accede al ColumnTransformer dentro del pipeline
# El nombre puede variar, pero suele ser 'columntransformer'
ct = model.named_steps['columntransformer']

# Get the feature names expected by the model
fields = ct.get_feature_names_out()


# Print all fields
for field in fields:
    print(field)
