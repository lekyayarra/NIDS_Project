import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/nids_random_forest.pkl")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🛡️ NIDS")

st.sidebar.write(
    "Network Intrusion Detection System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Traffic Detection",
        "📊 Model Evaluation",
        "ℹ️ About Project"
    ]
)

# =====================================================
# HOME
# =====================================================

if page == "🏠 Home":

    st.title("🛡️ Network Intrusion Detection System")

    st.subheader(
        "Machine Learning Based Network Traffic Detection"
    )

    st.write(
        "A Random Forest based system for detecting "
        "potential Bot traffic in network data."
    )

    st.divider()

    # Project statistics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "🎯 Accuracy",
            "99.95%"
        )

    with col3:
        st.metric(
            "🔢 Features",
            "78"
        )

    with col4:
        st.metric(
            "📊 Dataset",
            "CIC-IDS2017"
        )

    st.divider()

    st.header("How the NIDS Works")

    st.write(
        """
        1. Upload network traffic data.

        2. The system preprocesses the traffic features.

        3. The trained Random Forest model analyzes the data.

        4. Each traffic record is classified as BENIGN or Bot.

        5. Detection results can be viewed and downloaded.
        """
    )

    st.info(
        "Use the sidebar to navigate through the NIDS."
    )


# =====================================================
# TRAFFIC DETECTION
# =====================================================

elif page == "🔍 Traffic Detection":

    st.title("🔍 Network Traffic Detection")

    st.write(
        "Upload a CSV file containing network traffic features."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.success(
                "CSV uploaded successfully!"
            )

            # Dataset information

            st.subheader("📊 Dataset Information")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Traffic Records",
                    len(data)
                )

            with col2:
                st.metric(
                    "Columns",
                    len(data.columns)
                )

            # Prepare features

            features = data.copy()

            if "Label" in features.columns:
                features = features.drop(
                    "Label",
                    axis=1
                )

            features.columns = (
                features.columns.str.strip()
            )

            # Model features

            model_features = (
                model.feature_names_in_
            )

            missing_features = [
                feature
                for feature in model_features
                if feature not in features.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded file is missing "
                    "required model features."
                )

                st.write(
                    missing_features
                )

            else:

                # Correct feature order

                features = features[
                    model_features
                ]

                # Convert to numeric

                features = features.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Clean invalid values

                features = features.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                features = features.fillna(0)

                # Prediction

                predictions = model.predict(
                    features
                )

                prediction_labels = np.where(
                    predictions == 0,
                    "BENIGN",
                    "Bot"
                )

                # Add prediction

                results = data.copy()

                results["Prediction"] = (
                    prediction_labels
                )

                # Statistics

                benign_count = int(
                    np.sum(predictions == 0)
                )

                bot_count = int(
                    np.sum(predictions == 1)
                )

                total_count = len(
                    predictions
                )

                st.divider()

                st.subheader(
                    "🚨 Detection Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Traffic",
                        total_count
                    )

                with col2:
                    st.metric(
                        "🟢 BENIGN",
                        benign_count
                    )

                with col3:
                    st.metric(
                        "🔴 Bot",
                        bot_count
                    )

                # Alert

                if bot_count > 0:

                    st.error(
                        f"⚠️ Potential intrusion detected "
                        f"in {bot_count} record(s)."
                    )

                else:

                    st.success(
                        "✅ No Bot traffic detected."
                    )

                # Chart

                st.subheader(
                    "📈 Traffic Distribution"
                )

                chart_data = pd.DataFrame(
                    {
                        "Traffic Type": [
                            "BENIGN",
                            "Bot"
                        ],
                        "Count": [
                            benign_count,
                            bot_count
                        ]
                    }
                )

                st.bar_chart(
                    chart_data.set_index(
                        "Traffic Type"
                    )
                )

                # Results

                st.subheader(
                    "📋 Prediction Results"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # Download

                csv = results.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download Results",
                    data=csv,
                    file_name=(
                        "nids_prediction_results.csv"
                    ),
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )

    else:

        st.info(
            "Upload a CSV file to start detection."
        )


# =====================================================
# MODEL EVALUATION
# =====================================================

elif page == "📊 Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        "Evaluate the Random Forest model using "
        "a labeled test dataset."
    )

    test_file = st.file_uploader(
        "Upload labeled test CSV",
        type=["csv"],
        key="evaluation_file"
    )

    if test_file is not None:

        try:

            test_data = pd.read_csv(
                test_file
            )

            if "Label" not in test_data.columns:

                st.error(
                    "The test CSV must contain "
                    "a Label column."
                )

            else:

                actual_labels = (
                    test_data["Label"]
                )

                features = test_data.drop(
                    "Label",
                    axis=1
                )

                features.columns = (
                    features.columns.str.strip()
                )

                model_features = (
                    model.feature_names_in_
                )

                missing_features = [
                    feature
                    for feature in model_features
                    if feature not in features.columns
                ]

                if missing_features:

                    st.error(
                        "Required features are missing."
                    )

                    st.write(
                        missing_features
                    )

                else:

                    features = features[
                        model_features
                    ]

                    features = features.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    features = features.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    features = features.fillna(0)

                    predictions = model.predict(
                        features
                    )

                    actual = actual_labels.map(
                        {
                            "BENIGN": 0,
                            "Bot": 1
                        }
                    )

                    # Metrics

                    accuracy = accuracy_score(
                        actual,
                        predictions
                    )

                    precision = precision_score(
                        actual,
                        predictions,
                        zero_division=0
                    )

                    recall = recall_score(
                        actual,
                        predictions,
                        zero_division=0
                    )

                    f1 = f1_score(
                        actual,
                        predictions,
                        zero_division=0
                    )

                    st.divider()

                    st.subheader(
                        "📈 Performance Metrics"
                    )

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    with col1:
                        st.metric(
                            "Accuracy",
                            f"{accuracy * 100:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Precision",
                            f"{precision * 100:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Recall",
                            f"{recall * 100:.2f}%"
                        )

                    with col4:
                        st.metric(
                            "F1 Score",
                            f"{f1 * 100:.2f}%"
                        )

                    # Confusion matrix

                    st.subheader(
                        "Confusion Matrix"
                    )

                    cm = confusion_matrix(
                        actual,
                        predictions
                    )

                    cm_data = pd.DataFrame(
                        cm,
                        index=[
                            "Actual BENIGN",
                            "Actual Bot"
                        ],
                        columns=[
                            "Predicted BENIGN",
                            "Predicted Bot"
                        ]
                    )

                    st.dataframe(
                        cm_data,
                        use_container_width=True
                    )

                    st.success(
                        "✅ Model evaluation completed!"
                    )

        except Exception as e:

            st.error(
                f"Error during evaluation: {e}"
            )

    else:

        st.info(
            "Upload nids_test_data.csv "
            "to evaluate the model."
        )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.header(
        "Network Intrusion Detection System"
    )

    st.write(
        """
        This project uses machine learning to detect
        potentially malicious network traffic.
        """
    )

    st.subheader("Technology Used")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "Random Forest",
                "Streamlit",
                "CIC-IDS2017"
            ],
            "Purpose": [
                "Programming",
                "Data Processing",
                "Numerical Processing",
                "Machine Learning",
                "Classification",
                "Dashboard",
                "Network Traffic Dataset"
            ]
        }
    )

    st.table(
        technologies
    )

    st.subheader("Project Workflow")

    st.write(
        """
        Network Traffic
              ↓
        Data Cleaning
              ↓
        Preprocessing
              ↓
        Random Forest Model
              ↓
        Traffic Classification
              ↓
        NIDS Dashboard
        """
    )

    st.subheader("Model Result")

    st.success(
        "Random Forest achieved 99.95% accuracy "
        "on the held-out test set."
    )

    st.caption(
        "NIDS Project | Cybersecurity + Machine Learning"
    )