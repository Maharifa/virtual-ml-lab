import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Experiment", page_icon="📊", layout="wide")
st.title("📊 Experiment: k-Nearest Neighbors")
st.page_link("app.py", label="Back to experiments", icon="🏠")
st.markdown("""
**Goal:** See how the number of neighbors changes a classifier's decision boundary.

The two colored groups form a synthetic `two moons` dataset. Adjust **k**, train the model,
and compare the test accuracy and boundary.
""")

k = st.slider("Number of neighbors (k)", min_value=1, max_value=31, value=5, step=2)
test_fraction = st.slider("Fraction of data used for testing", 0.1, 0.5, 0.25, 0.05)

if st.button("Train model", type="primary"):
    X, y = make_moons(n_samples=500, noise=0.28, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=7, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    col1, col2 = st.columns([1, 2])
    col1.metric("Test accuracy", f"{accuracy:.1%}")
    col1.write(f"Training examples: **{len(X_train)}**")
    col1.write(f"Testing examples: **{len(X_test)}**")

    padding = 0.5
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    predictions = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, predictions, alpha=0.22, cmap="coolwarm")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="white", label="Training data")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", marker="x", s=55, linewidths=1.5, label="Test data")
    ax.set_title(f"k-nearest neighbors decision regions (k={k})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    col2.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Think about it")
    st.markdown("- What changes when you increase **k**?\n- Does the train/test split change the score? Why might it?\n- Can you find a value of **k** that performs poorly?")
else:
    st.info("Choose the settings, then select **Train model** to run the experiment.")
