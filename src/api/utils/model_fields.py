import joblib

# Ruta a tu modelo .joblib
MODEL_PATH = "../../../models/best_rf_model.joblib"

# Carga el modelo
model = joblib.load(MODEL_PATH)

# Accede al ColumnTransformer dentro del pipeline
# El nombre puede variar, pero suele ser 'columntransformer'
ct = model.named_steps['columntransformer']

# Obtén los nombres de los features que espera el modelo
fields = ct.get_feature_names_out()

# Imprime todos los fields
for field in fields:
    print(field)