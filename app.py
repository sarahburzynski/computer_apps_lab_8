import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Hamilton County Housing Value Predictor")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
    "deploy_artifacts/house_value_model.h5",
    compile=False
    )
    preprocess = joblib.load("deploy_artifacts/preprocess.joblib")
    return model, preprocess

model, preprocess = load_artifacts()

st.title("Hamilton County Housing Value Predictor")

st.write("Predicts APPRAISED_VALUE using a neural network.")

st.header("Input features")

land_value = st.number_input("LAND_VALUE", min_value=0.0, value=50000.0, step=1000.0)
build_value = st.number_input("BUILD_VALUE", min_value=0.0, value=150000.0, step=1000.0)

neighborhood = st.text_input("NEIGHBORHOOD_CODE_DESC", value="")
zoning = st.text_input("ZONING_DESC", value="")

X_input = pd.DataFrame([{
    "LAND_VALUE": land_value,
    "BUILD_VALUE": build_value,
    "NEIGHBORHOOD_CODE_DESC": neighborhood,
    "ZONING_DESC": zoning,
}])

if st.button("Predict property value"):
    Xp = preprocess.transform(X_input)
    pred_log = model.predict(Xp, verbose=0).ravel()[0]
    pred_dollars = np.expm1(pred_log)
    st.success(f"Estimated property value: ${pred_dollars:,.0f}")


