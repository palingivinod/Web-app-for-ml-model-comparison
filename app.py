import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    accuracy_score
)
import matplotlib.pyplot as plt

def main():
    st.title("🚢 Titanic Survival Prediction App")
    st.sidebar.title("🚢 Titanic Survival Prediction App")
    st.sidebar.markdown("Predict which passengers survived the Titanic disaster")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("archive/Titanic-Dataset.csv")
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
        label = LabelEncoder()
        data['Sex'] = label.fit_transform(data['Sex'])
        data['Embarked'] = label.fit_transform(data['Embarked'])
        data['Title'] = label.fit_transform(data['Title'])
        data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        return data

    @st.cache_data(persist=True)
    def split(df):
        y = df.Survived
        X = df.drop(columns=["Survived"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    def plot_metrics(metrics_list, model, X_test, y_test, class_names):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test, display_labels=class_names, ax=ax
            )
            st.pyplot(fig)

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ["Did Not Survive", "Survived"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Support Vector Machine", "Logistic Regression", "Random Forest")
    )

    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C"
        )
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify_svm"):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))
            st.write("🎯 Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("📌 Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, step=10, key="max_iter"
        )
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify_lr"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))
            st.write("🎯 Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("📌 Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "Number of trees in the forest", 100, 5000, step=10, key="n_estimators"
        )
        max_depth = st.sidebar.number_input(
            "Max depth of the tree", 1, 20, step=1, key="max_depth"
        )
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees?", ("True", "False"), key="bootstrap"
        )
        bootstrap = True if bootstrap == "True" else False
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        if st.sidebar.button("Classify", key="classify_rf"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))
            st.write("🎯 Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("📌 Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Titanic Dataset")
        st.write(df)
        

if __name__ == "__main__":
    main()