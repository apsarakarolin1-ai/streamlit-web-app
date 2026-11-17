
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #6a0dad;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model(format_type="joblib"):
    try:
        if format_type == "joblib":
            return joblib.load("models/iris_model.joblib")

        elif format_type == "pickle":
            with open("models/iris_model.pickle", "rb") as f:
                return pickle.load(f)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model information
@st.cache_data
def load_model_info():
    try:
        with open("models/model_info.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading model information: {e}")
        return None


# Load feature ranges
@st.cache_data
def load_feature_ranges():
    default_ranges = {
        "sepal_length": {
            "min": 4.0, "max": 8.0, "default": 5.8
        },
        "sepal_width": {
            "min": 2.0, "max": 4.5, "default": 3.0
        },
        "petal_length": {
            "min": 1.0, "max": 7.0, "default": 4.0
        },
        "petal_width": {
            "min": 0.1, "max": 2.5, "default": 1.2
        }
    }

    try:
        with open("models/feature_ranges.json", "r") as f:
            saved_ranges = json.load(f)

        for feature in default_ranges:
            if feature not in saved_ranges:
                saved_ranges[feature] = default_ranges[feature]
            else:
                for key in default_ranges[feature]:
                    saved_ranges[feature].setdefault(
                        key, default_ranges[feature][key]
                    )

        return saved_ranges

    except Exception:
        return default_ranges


# Load information
model_info = load_model_info()
feature_ranges = load_feature_ranges()


# Sidebar settings
with st.sidebar:
    st.title("⚙ Settings")

    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"]
    )

    if st.button("🔄 Reload Model"):
        load_model.clear()
        st.rerun()

    st.divider()
    st.subheader("📊 Model Information")

    if model_info:
        st.write(f"**Type:** {model_info.get('model_type', 'RandomForest')}")
        st.write(f"**Accuracy:** {model_info.get('accuracy', 0.96):.1%}")
        st.write(f"**Features:** {len(model_info.get('feature_names', []))}")
        st.write(f"**Classes:** {len(model_info.get('target_names', []))}")


# Load selected model
model = load_model(model_format)


# Main heading
st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

st.write(
    "This application predicts the species of an Iris flower "
    "based on its measurements using Machine Learning."
)


# Input columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges["sepal_length"]["min"]),
        max_value=float(feature_ranges["sepal_length"]["max"]),
        value=float(feature_ranges["sepal_length"]["default"]),
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges["sepal_width"]["min"]),
        max_value=float(feature_ranges["sepal_width"]["max"]),
        value=float(feature_ranges["sepal_width"]["default"]),
        step=0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges["petal_length"]["min"]),
        max_value=float(feature_ranges["petal_length"]["max"]),
        value=float(feature_ranges["petal_length"]["default"]),
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges["petal_width"]["min"]),
        max_value=float(feature_ranges["petal_width"]["max"]),
        value=float(feature_ranges["petal_width"]["default"]),
        step=0.1
    )


# Current values
with col2:
    st.header("📊 Current Values")

    features_df = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value (cm)": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(
        features_df,
        hide_index=True,
        use_container_width=True
    )


# Prediction
if st.button("🎯 Predict Species", type="primary"):

    input_features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    if model is not None and model_info is not None:
        try:
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)

            predicted_class = model_info["target_names"][
                int(prediction[0])
            ]

            st.success(
                f"🌸 Predicted Species: {predicted_class}"
            )

            st.subheader("📈 Confidence Scores")

            for i, prob in enumerate(prediction_proba[0]):
                species = model_info["target_names"][i]
                percentage = float(prob) * 100

                st.write(
                    f"**{species}: {percentage:.1f}%**"
                )

                st.progress(float(prob))

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    else:
        st.error(
            "Model could not be loaded. Check your model files."
        )


# Dataset information
with st.expander("📚 About the Iris Dataset"):

    st.write("""
    The Iris dataset was introduced by Ronald Fisher in 1936.

    **Dataset Characteristics:**
    - 150 samples
    - 4 features
    - 3 classes

    **Species:**
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica

    **Features:**
    1. Sepal length
    2. Sepal width
    3. Petal length
    4. Petal width

    Model: Random Forest Classifier.
    """)


# Footer
st.markdown("---")

st.markdown(
    "<div style='text-align:center'>"
    "<p>Built with Streamlit and Scikit-learn</p>"
    "</div>",
    unsafe_allow_html=True
)



