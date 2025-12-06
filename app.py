import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential

# Suppress TensorFlow logging output for predictions
tf.get_logger().setLevel('ERROR')

st.set_page_config(layout="wide")

# --- 1. Load Artifacts ---
@st.cache_resource # Cache the loading of models and transformers for efficiency
def load_all_artifacts():
    # Load StandardScaler and OneHotEncoder
    scaler = joblib.load('scaler.joblib')
    encoder = joblib.load('encoder.joblib')
    
    # Load the final meta-classifier
    final_meta_classifier = load_model('final_stacked_model.h5')
    
    # Load base classifiers
    base_classifiers_app = {}
    base_classifiers_info = {
        'KNN': 'base_classifier_knn.joblib',
        'RandomForest': 'base_classifier_randomforest.joblib',
        'ANN_base': 'base_classifier_ann_base.h5',
        'DecisionTree': 'base_classifier_decisiontree.joblib',
        'SVM': 'base_classifier_svm.joblib'
    }

    for name, path in base_classifiers_info.items():
        if '.h5' in path:
            # Keras models need a specific loading function
            base_classifiers_app[name] = load_model(path)
        else:
            # Scikit-learn models use joblib
            base_classifiers_app[name] = joblib.load(path)
    
    return scaler, encoder, final_meta_classifier, base_classifiers_app

scaler, encoder, final_meta_classifier, base_classifiers_app = load_all_artifacts()

# --- 2. Define Feature Lists and Defaults ---
# These lists must match the order and names used during model training
numerical_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
original_categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Get the exact column order from the trained X_final for consistent preprocessing
# This list should ideally be loaded from a saved file, but for this exercise, we reconstruct it
# based on the known preprocessing steps (numerical features first, then one-hot encoded categories)
trained_feature_columns = numerical_features + list(encoder.get_feature_names_out(original_categorical_features))

# Hardcoded default values for Streamlit widgets based on the training data's means/modes
# In a production environment, these would be saved and loaded from model artifacts.
numerical_defaults = {
    'age': 46.347826, 'bp': 76.242424, 'sg': 1.017408, 'al': 1.018305, 'su': 0.457831,
    'bgr': 148.036517, 'bu': 57.433299, 'sc': 3.072454, 'sod': 137.528754, 'pot': 4.627244,
    'hemo': 12.526909, 'pcv': 38.884499, 'wbcc': 8406.122449, 'rbcc': 4.707435
}

categorical_defaults = {
    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
}

# Map encoder categories to original categorical feature names for selectbox options
categorical_choices = {
    original_categorical_features[i]: list(encoder.categories_[i])
    for i in range(len(original_categorical_features))
}


# --- Streamlit App Title and Description ---
st.title("Chronic Kidney Disease Prediction App")
st.markdown("---")
st.write("Please enter the patient's details below to get a prediction on the likelihood of Chronic Kidney Disease.")

# --- Input fields for user data ---
st.sidebar.header("Patient Data Input")
input_data = {}

# Numerical Inputs
for feature in numerical_features:
    default_value = numerical_defaults.get(feature, 0.0)
    input_data[feature] = st.sidebar.number_input(
        f"{feature.replace('_', ' ').title()}",
        value=float(default_value),
        step=0.1 if feature not in ['age', 'bp', 'wbcc', 'rbcc'] else 1.0 # Adjust step for integer-like features
    )

# Categorical Inputs
for feature in original_categorical_features:
    default_value = categorical_defaults.get(feature, categorical_choices[feature][0])
    input_data[feature] = st.sidebar.selectbox(
        f"{feature.replace('_', ' ').title()}",
        options=categorical_choices[feature],
        index=categorical_choices[feature].index(default_value)
    )

# --- Preprocessing function ---
def preprocess_input(raw_input_data):
    # Create DataFrame from raw input
    input_df = pd.DataFrame([raw_input_data])

    # Separate numerical and categorical for preprocessing
    numerical_part_df = input_df[numerical_features]
    categorical_part_df = input_df[original_categorical_features]

    # Scale numerical features using the loaded scaler
    scaled_numerical_features = scaler.transform(numerical_part_df)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features, index=input_df.index)

    # One-hot encode categorical features using the loaded encoder
    encoded_categorical_features = encoder.transform(categorical_part_df)
    encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(original_categorical_features), index=input_df.index)

    # Combine preprocessed numerical and one-hot encoded categorical features
    combined_processed_df = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
    
    # Ensure all columns are present and in the correct order as per training data
    # Features not present in the input (e.g., if a category was never selected) will be set to 0.0
    final_processed_df = pd.DataFrame(0.0, columns=trained_feature_columns, index=input_df.index)
    for col in combined_processed_df.columns:
        if col in final_processed_df.columns:
            final_processed_df[col] = combined_processed_df[col]
    
    return final_processed_df

# --- Meta-feature generation function ---
def generate_meta_features(processed_input_df, base_models):
    meta_features = pd.DataFrame(index=processed_input_df.index)
    
    # Define the order of base classifiers to match how the meta-classifier was trained
    meta_feature_names_order = ['KNN', 'RandomForest', 'ANN_base', 'DecisionTree', 'SVM']
    
    for name in meta_feature_names_order:
        model = base_models[name]
        if isinstance(model, Sequential): # Keras model (ANN_base)
            pred_proba = model.predict(processed_input_df, verbose=0).flatten()
        else: # Scikit-learn models (KNN, RandomForest, DecisionTree, SVM)
            pred_proba = model.predict_proba(processed_input_df)[:, 1]
        meta_features[name] = pred_proba
    return meta_features


# --- Prediction button and result display ---
if st.sidebar.button("Predict"): # Button to trigger prediction
    # Preprocess user input
    processed_input = preprocess_input(input_data)

    # Generate meta-features from base classifiers' predictions
    meta_features_input = generate_meta_features(processed_input, base_classifiers_app)

    # Make final prediction using the deep learning meta-classifier
    final_prediction_proba = final_meta_classifier.predict(meta_features_input, verbose=0).flatten()[0]
    final_prediction = (final_prediction_proba > 0.5).astype(int)

    st.subheader("Prediction Result")
    if final_prediction == 1:
        st.error(f"**Prediction: Chronic Kidney Disease Detected!** (Probability: {final_prediction_proba:.4f})")
    else:
        st.success(f"**Prediction: No Chronic Kidney Disease Detected.** (Probability: {final_prediction_proba:.4f})")

    st.markdown("---")
    st.subheader("Debug Information")
    with st.expander("Click to see raw input, processed input, and meta-features"): # Expander for debug info
        st.write("Raw Input:")
        st.write(pd.DataFrame([input_data]))
        st.write("Processed Input (Scaled and One-Hot Encoded):")
        st.write(processed_input)
        st.write("Meta-Features (Base Classifier Probabilities):")
        st.write(meta_features_input)

# --- Instructions to run the app ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to run this Streamlit app:
1.  **Save this code** as `app.py`.
2.  **Ensure all saved model artifacts** (`scaler.joblib`, `encoder.joblib`, `final_stacked_model.h5`, and all `base_classifier_*.joblib`/`.h5` files) are in the **same directory** as `app.py`.
3.  Open your **terminal or command prompt**.
4.  Navigate to the directory where you saved `app.py` and the model artifacts.
5.  Run the command: `streamlit run app.py`
6.  Your web browser will automatically open to the Streamlit app.
""")