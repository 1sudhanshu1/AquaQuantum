"""
AquaQuantum - Quantum-Based Water Quality Prediction System
-----------------------------------------------------------
A beginner-friendly Streamlit app that trains a Quantum Support Vector
Machine (QSVC) on the Water Potability dataset using a Qiskit quantum
kernel, run on a classical simulator (no real quantum computer needed).

Run with:  streamlit run app.py
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---- Qiskit imports (shown as a friendly error if not installed) ----------
try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    try:
        from qiskit.circuit.library import zz_feature_map  # Qiskit 2.x
    except ImportError:  # older Qiskit versions
        from qiskit.circuit.library import ZZFeatureMap as zz_feature_map

    QISKIT_OK = True
except ImportError as err:
    QISKIT_OK = False
    QISKIT_ERROR = str(err)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "water_potability.csv")

# (column name in CSV, label, unit, help text)
FEATURES = [
    ("ph", "pH", "", "Acidity of water (0-14). Drinking water is usually about 6.5 - 8.5."),
    ("Hardness", "Hardness", "mg/L", "Calcium and magnesium dissolved in the water."),
    ("Solids", "Solids (TDS)", "ppm", "Total dissolved solids."),
    ("Chloramines", "Chloramines", "ppm", "Disinfectant used in water treatment."),
    ("Sulfate", "Sulfate", "mg/L", "Naturally occurring sulfate salts."),
    ("Conductivity", "Conductivity", "µS/cm", "How well the water conducts electricity."),
    ("Organic_carbon", "Organic Carbon", "ppm", "Total organic carbon in the water."),
    ("Trihalomethanes", "Trihalomethanes", "µg/L", "By-products of chlorine disinfection."),
    ("Turbidity", "Turbidity", "NTU", "Cloudiness of the water."),
]
FEATURE_COLS = [f[0] for f in FEATURES]
N_QUBITS = len(FEATURE_COLS)  # one qubit per water-quality parameter

st.set_page_config(page_title="AquaQuantum", page_icon="💧", layout="wide")

# ---------------------------------------------------------------------------
# Water-inspired styling (blue / cyan / white)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #0B3954 0%, #0891B2 55%, #67E8F9 100%);
        padding: 2.2rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.4rem; }
    .hero p  { color: #E0F7FF; margin: 0.4rem 0 0 0; font-size: 1.05rem; }
    .card {
        background: #E8F7FC; border-left: 5px solid #0891B2;
        padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 0.8rem;
    }
    .result-good { background:#D1FAE5; border-left:6px solid #059669; color:#064E3B;
                   padding:1.1rem 1.3rem; border-radius:12px; font-size:1.25rem; }
    .result-bad  { background:#FFE4E6; border-left:6px solid #E11D48; color:#881337;
                   padding:1.1rem 1.3rem; border-radius:12px; font-size:1.25rem; }
    div.stButton > button {
        background: linear-gradient(90deg, #0891B2, #0E7490); color: white;
        border: none; border-radius: 10px; font-weight: 600; padding: 0.55rem 1.4rem;
    }
    div.stButton > button:hover { background: #0B3954; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Read the Water Potability CSV. Returns None if the file is missing."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def preprocess(raw_values, imputer, scaler):
    """Fill missing values, then scale every feature to the range 0 - pi.
    (Quantum feature maps use the numbers as rotation angles.)"""
    filled = imputer.transform(raw_values)
    scaled = scaler.transform(filled)
    return np.clip(scaled, 0, np.pi)


def train_quantum_model(df, n_train, seed=42):
    """Train a Quantum SVM (QSVC) on a small, balanced sample of the data."""
    X = df[FEATURE_COLS].values
    y = df["Potability"].values

    # Keep 20% of the data aside for testing.
    X_pool, X_test_pool, y_pool, y_test_pool = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    rng = np.random.RandomState(seed)

    def balanced_sample(labels, n_per_class):
        idx = [rng.choice(np.where(labels == c)[0], n_per_class, replace=False) for c in (0, 1)]
        return np.concatenate(idx)

    train_idx = balanced_sample(y_pool, n_train // 2)  # small sample = fast quantum kernel
    test_idx = balanced_sample(y_test_pool, 50)        # 100 balanced test samples
    X_train, y_train = X_pool[train_idx], y_pool[train_idx]
    X_test, y_test = X_test_pool[test_idx], y_test_pool[test_idx]

    # Preprocessing (fitted on the training sample only)
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(imputer.transform(X_train))
    X_train_q = preprocess(X_train, imputer, scaler)
    X_test_q = preprocess(X_test, imputer, scaler)

    # Quantum feature map + quantum kernel + SVM
    feature_map = zz_feature_map(feature_dimension=N_QUBITS, reps=1)
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)

    start = time.time()
    qsvc.fit(X_train_q, y_train)
    train_time = time.time() - start
    q_acc = accuracy_score(y_test, qsvc.predict(X_test_q))

    # Classical SVM on the same data, only as an honest reference point
    classical = SVC(kernel="rbf").fit(X_train_q, y_train)
    c_acc = accuracy_score(y_test, classical.predict(X_test_q))

    # A small piece of the kernel matrix (30 samples per class) for the heatmap
    show = np.concatenate([np.where(y_train == 0)[0][:30], np.where(y_train == 1)[0][:30]])
    kernel_block = kernel.evaluate(X_train_q[show])

    return {
        "model": qsvc,
        "imputer": imputer,
        "scaler": scaler,
        "q_acc": q_acc,
        "c_acc": c_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_time": train_time,
        "kernel_block": kernel_block,
        "n_each": len(show) // 2,
    }


# ---------------------------------------------------------------------------
# Load data and check requirements
# ---------------------------------------------------------------------------
df = load_data()

if not QISKIT_OK:
    st.error(
        "Qiskit is not installed correctly. Run `pip install -r requirements.txt` "
        f"and restart the app.\n\nDetails: {QISKIT_ERROR}"
    )
    st.stop()

if df is None:
    st.error(
        "Dataset not found. Download **water_potability.csv** from Kaggle "
        "(search for 'Water Potability') and save it as `data/water_potability.csv`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: model training
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💧 AquaQuantum")
    st.caption("Quantum ML water quality predictor")
    st.divider()
    st.subheader("⚙️ Model Setup")
    n_train = st.slider(
        "Training samples", min_value=60, max_value=300, value=200, step=20,
        help="Quantum kernels are expensive, so only a small sample is used.",
    )
    if st.button("🚀 Train Quantum Model"):
        with st.spinner("Computing the quantum kernel on the simulator..."):
            st.session_state["trained"] = train_quantum_model(df, n_train)
        st.success("Model trained!")

    if "trained" in st.session_state:
        st.info(f"✅ Model ready ({st.session_state['trained']['n_train']} training samples)")
    else:
        st.warning("Model not trained yet. Click the button above.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💧 AquaQuantum</h1>
        <p>Quantum-Based Water Quality Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_predict, tab_quantum, tab_apps = st.tabs(
    ["🏠 Home", "🔬 Predict", "⚛️ How It Works", "🌍 Real-Life Use"]
)

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
with tab_home:
    st.subheader("About the Project")
    st.write(
        "Clean drinking water is essential for health, but testing it in a lab takes time and money. "
        "**AquaQuantum** takes nine water-quality measurements and predicts whether the water is "
        "**potentially potable (safe to drink)** or **non-potable**."
    )
    st.write(
        "The prediction is made by a **Quantum Machine Learning** model: a Support Vector Machine "
        "(SVM) that uses a **quantum kernel** built with Qiskit and run on a quantum simulator."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records in dataset", f"{len(df):,}")
    c2.metric("Water parameters", len(FEATURE_COLS))
    c3.metric("Potable samples", f"{(df['Potability'] == 1).mean() * 100:.0f}%")
    c4.metric("Qubits used", N_QUBITS)

    st.markdown("### How to use")
    st.markdown(
        """
        1. Click **Train Quantum Model** in the sidebar.
        2. Open the **Predict** tab and enter the water parameters.
        3. Click **Predict** to see the result.
        """
    )
    st.markdown(
        '<div class="card"><b>Note:</b> This is a college project that demonstrates quantum machine '
        "learning. It is <b>not</b> a replacement for real laboratory water testing.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Water Quality Prediction")
    medians = df[FEATURE_COLS].median()

    values = {}
    cols = st.columns(3)
    for i, (col, label, unit, hint) in enumerate(FEATURES):
        name = f"{label} ({unit})" if unit else label
        values[col] = cols[i % 3].number_input(
            name, min_value=0.0, value=float(round(medians[col], 2)), step=0.1, help=hint
        )

    if st.button("💧 Predict Water Quality"):
        if "trained" not in st.session_state:
            st.warning("⚠️ The quantum model is not trained yet. Click **Train Quantum Model** in the sidebar first.")
        else:
            t = st.session_state["trained"]
            sample = pd.DataFrame([values])[FEATURE_COLS].values
            sample_q = preprocess(sample, t["imputer"], t["scaler"])
            with st.spinner("Running the quantum kernel..."):
                prediction = int(t["model"].predict(sample_q)[0])

            if prediction == 1:
                st.markdown('<div class="result-good">✅ <b>Potentially Potable</b> - the model predicts this water may be safe to drink.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-bad">⚠️ <b>Non-Potable</b> - the model predicts this water may NOT be safe to drink.</div>', unsafe_allow_html=True)
            st.caption("This is a model prediction for learning purposes, not a certified water safety test.")

    if "trained" in st.session_state:
        t = st.session_state["trained"]
        st.divider()
        st.subheader("Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Quantum SVM accuracy", f"{t['q_acc'] * 100:.1f}%")
        m2.metric("Classical SVM (reference)", f"{t['c_acc'] * 100:.1f}%")
        m3.metric("Training time", f"{t['train_time']:.1f} s")
        st.caption(
            f"Measured on {t['n_test']} balanced test samples the model never saw during training "
            f"(chance level is 50%). The model was trained on only {t['n_train']} samples. "
            "The Water Potability data is noisy, so accuracy stays modest for both models; "
            "this project does not claim that quantum is better than classical."
        )

        st.markdown("**Quantum kernel matrix (training samples)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        img = ax.imshow(t["kernel_block"], cmap="Blues", vmin=0, vmax=1)
        n = t["n_each"]
        ax.axhline(n - 0.5, color="white", lw=2)
        ax.axvline(n - 0.5, color="white", lw=2)
        ax.set_xlabel("Training samples (non-potable first, then potable)")
        ax.set_ylabel("Training samples")
        fig.colorbar(img, label="Quantum similarity")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Each cell shows how similar two water samples are, as measured by the quantum circuit. Brighter blue = more similar.")

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
with tab_quantum:
    st.subheader("How Quantum Machine Learning Predicts Water Quality")
    st.markdown(
        """
        <div class="card"><b>1. Prepare the data.</b> Missing values are filled with the median and
        every parameter is scaled to an angle between 0 and π.</div>
        <div class="card"><b>2. Quantum feature map (ZZFeatureMap).</b> Each of the 9 parameters
        is assigned to one qubit. The feature map rotates each qubit by the parameter's angle and
        entangles neighbouring qubits, so a water sample becomes a quantum state.</div>
        <div class="card"><b>3. Quantum kernel.</b> For two water samples we build their quantum
        states and measure how much they overlap (their <i>fidelity</i>). A high overlap means the
        samples are similar. Doing this for all pairs of training samples gives the kernel matrix.</div>
        <div class="card"><b>4. SVM classifier.</b> A classical Support Vector Machine uses this
        quantum kernel matrix to find the boundary between potable and non-potable water.</div>
        <div class="card"><b>5. Prediction.</b> For new input values, the kernel compares the new
        sample with the training samples and the SVM outputs the category.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Simulator:** the quantum circuits are simulated exactly on your computer with Qiskit, so no real quantum hardware is needed. A small training sample is used because the number of kernel calculations grows quickly with the sample size.")
    st.code(
        "feature_map = zz_feature_map(feature_dimension=9, reps=1)\n"
        "kernel = FidelityStatevectorKernel(feature_map=feature_map)\n"
        "model = QSVC(quantum_kernel=kernel)\n"
        "model.fit(X_train, y_train)",
        language="python",
    )

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
with tab_apps:
    st.subheader("Real-Life Applications")
    st.markdown(
        """
        - 🚰 **Quick screening** of drinking water in villages and remote areas before lab testing.
        - 🏙️ **Municipal water monitoring** to flag unusual readings early.
        - 🏭 **Industrial and environmental monitoring** of rivers, lakes and wastewater.
        - 🎓 **Research and education** on how quantum computing can support AI for environmental problems.
        """
    )
    st.markdown(
        '<div class="card">Quantum machine learning is still an early research area. Today this project '
        "runs on a simulator and demonstrates the idea; it does not show a quantum advantage.</div>",
        unsafe_allow_html=True,
    )
