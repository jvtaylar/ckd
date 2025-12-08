import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential

tf.get_logger().setLevel('ERROR')
st.set_page_config(layout="wide")

# ===============================================================
# 1. CHECK REQUIRED FILES
# ===============================================================
REQUIRED_FILES = [
    "scaler.joblib",
    "encoder.joblib",
    "final_stacked_model.h5",
    "base_classifier_knn.joblib",
    "base_classifier_randomforest.joblib",
    "base_classifier_ann_base.h5",
    "base_classifier_decisiontree.joblib",
    "base_classifier_svm.joblib"
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_files:
    st.error("❌ Missing model files!")
    st.write(missing_files)
    st.stop()

# ===============================================================
# 2. LOAD SCALER, ENCODER, META-CLASSIFIER (CACHED)
# ===============================================================
@st.cache_resource(show_spinner=True)
def load_core_models():
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("encoder.joblib")
    meta_classifier = load_model("final_stacked_model.h5")
    return scaler, encoder, meta_classifier

scaler, encoder, meta_classifier = load_core_models()

# ===============================================================
# 3. FEATURES
# ===============================================================
numerical_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                      'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                        'appet', 'pe', 'ane']

encoder_feature_names = encoder.get_feature_names_out(categorical_features)
trained_feature_columns = numerical_features + list(encoder_feature_names)

# Defaults
numerical_defaults = {
    'age': 46.3, 'bp': 76.2, 'sg': 1.0174, 'al': 1.0183, 'su': 0.46,
    'bgr': 148.03, 'bu': 57.43, 'sc': 3.07, 'sod': 137.5, 'pot': 4.63,
    'hemo': 12.52, 'pcv': 38.88, 'wbcc': 8406, 'rbcc': 4.70
}

categorical_defaults = {
    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
}

categorical_choices = {
    feature: list(encoder.categories_[i])
    for i, feature in enumerate(categorical_features)
}

# ===============================================================
# 4. UI
# ===============================================================
st.title("⚕️ Chronic Kidney Disease Prediction")
st.write("Enter patient information to get a prediction.")
st.markdown("---")

st.sidebar.header("Patient Data Input")
input_data = {}

for feature in numerical_features:
    input_data[feature] = st.sidebar.number_input(
        feature.upper(),
        value=float(numerical_defaults.get(feature, 0)),
        step=0.1
    )

for feature in categorical_features:
    choices = categorical_choices[feature]
    default = categorical_defaults.get(feature, choices[0])
    input_data[feature] = st.sidebar.selectbox(
        feature.upper(),
        choices,
        index=choices.index(default)
    )

# ===============================================================
# 5. PREPROCESS INPUT
# ===============================================================
def preprocess_input(data):
    df = pd.DataFrame([data])
    scaled = scaler.transform(df[numerical_features])
    scaled_df = pd.DataFrame(scaled, columns=numerical_features)

    encoded = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded, columns=encoder_feature_names)

    final_df = pd.DataFrame(columns=trained_feature_columns)
    for col in final_df.columns:
        if col in scaled_df:
            final_df[col] = scaled_df[col]
        elif col in encoded_df:
            final_df[col] = encoded_df[col]
        else:
            final_df[col] = 0
    return final_df

# ===============================================================
# 6. LAZY-LOAD BASE MODELS
# ===============================================================
def load_base_model(name):
    path_map = {
        "KNN": "base_classifier_knn.joblib",
        "RandomForest": "base_classifier_randomforest.joblib",
        "ANN_base": "base_classifier_ann_base.h5",
        "DecisionTree": "base_classifier_decisiontree.joblib",
        "SVM": "base_classifier_svm.joblib"
    }
    path = path_map[name]
    if path.endswith(".h5"):
        return load_model(path)
    else:
        return joblib.load(path)

# ===============================================================
# 7. GENERATE META FEATURES
# ===============================================================
def generate_meta_features(df):
    meta = pd.DataFrame()
    order = ["KNN", "RandomForest", "ANN_base", "DecisionTree", "SVM"]

    for name in order:
        model = load_base_model(name)
        if isinstance(model, Sequential):
            pred = model.predict(df, verbose=0).flatten()
        else:
            pred = model.predict_proba(df)[:, 1]
        meta[name] = pred
    return meta

# ===============================================================
# 8. PREDICT
# ===============================================================
if st.sidebar.button("Predict"):

    processed = preprocess_input(input_data)
    meta = generate_meta_features(processed)

    final_prob = meta_classifier.predict(meta, verbose=0)[0][0]
    label = int(final_prob > 0.5)

    st.subheader("Prediction Result")
    if label == 1:
        st.error(f"CKD DETECTED — Probability: {final_prob:.4f}")
    else:
        st.success(f"No CKD Detected — Probability: {final_prob:.4f}")

    # Debug info
    st.markdown("---")
    with st.expander("🔍 Debug Information"):
        st.write("Raw Input:", pd.DataFrame([input_data]))
        st.write("Processed Input:", processed)
        st.write("Meta-features:", meta)
