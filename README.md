# 💧 AquaQuantum – Quantum-Based Water Quality Prediction System

A beginner-friendly Streamlit web app that predicts whether water is **potentially potable** or **non-potable** from 9 water-quality measurements, using a **Quantum Support Vector Machine (QSVC)** built with **Qiskit** and run on a **quantum simulator** (no real quantum computer needed).

## Project files

```
AquaQuantum/
├── app.py                    # the whole Streamlit app
├── requirements.txt          # Python packages
├── README.md                 # this file
├── .streamlit/config.toml    # blue/cyan/white theme
└── data/water_potability.csv # Water Potability dataset (3,276 rows)
```

## How it works (short version)

1. Missing values are filled with the median; every parameter is scaled to an angle between 0 and π.
2. A **ZZFeatureMap** (9 qubits, one per parameter) turns each water sample into a quantum state.
3. A **quantum kernel** (`FidelityStatevectorKernel`) measures how similar two samples are (state overlap).
4. A classical **SVM** uses this kernel to separate potable from non-potable water (`QSVC`).
5. Only a small balanced sample (default 200) is used for training because quantum kernels are expensive.

## Run it in VS Code on Windows (step by step)

1. **Install Python 3.10, 3.11 or 3.12** from https://www.python.org/downloads/. On the first installer screen, tick **"Add python.exe to PATH"**. (The app was tested with Python 3.12.)
2. **Install VS Code** from https://code.visualstudio.com/ and install the **Python** extension (Extensions icon on the left, search "Python").
3. **Put the project folder somewhere simple**, e.g. `C:\Projects\AquaQuantum`. In VS Code: **File → Open Folder…** and choose it.
4. **Open a terminal**: **Terminal → New Terminal**.
5. **Create a virtual environment** (keeps packages tidy):
   ```
   python -m venv venv
   ```
6. **Activate it**:
   ```
   venv\Scripts\activate
   ```
   If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope Process Bypass` and try again, or switch the terminal to **Command Prompt** (the dropdown next to the `+` in the terminal panel). You should see `(venv)` at the start of the line.
7. **Install the packages** (takes a few minutes):
   ```
   pip install -r requirements.txt
   ```
8. **Run the app**:
   ```
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.
9. In the app: click **Train Quantum Model** in the sidebar (takes a few seconds), open the **Predict** tab, enter values and click **Predict Water Quality**.
10. To stop the app, press `Ctrl + C` in the terminal.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick "Add python.exe to PATH", then reopen VS Code. |
| `streamlit is not recognized` | Activate the virtual environment first (step 6), or run `python -m streamlit run app.py`. |
| Qiskit install errors | Use Python 3.10–3.12 (very new Python versions may not have packages yet). |
| "Dataset not found" | Make sure the file is exactly `data/water_potability.csv`. |
| Training feels slow | Lower the **Training samples** slider in the sidebar. |

## Dataset

`data/water_potability.csv` is the public **Water Potability** dataset from Kaggle (search "Water Potability" on kaggle.com if you need to download it again). Columns: `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability` (1 = potable, 0 = not potable). It has missing values in `ph`, `Sulfate` and `Trihalomethanes`, which the app fills with the median.

## Honest notes (good for your viva / report)

- The model really trains on the dataset and predicts with the quantum kernel; there are no hard-coded predictions. If the model isn't trained, the app tells you to train it first.
- In our test run (200 training samples, 100 balanced test samples) the quantum SVM scored about **60%** and a classical RBF SVM about **58%**. Chance level is 50%. The dataset is noisy, so both models are modest. **This project does not claim quantum is more accurate than classical ML.**
- The app is a learning demo, **not** a certified water-safety test.
- The simulator computes the kernel exactly (no hardware noise). Real hardware would be slower and noisier.

## Ideas for later versions

Classical-vs-quantum comparison page, different feature maps (ZFeatureMap, PauliFeatureMap), PCA to fewer qubits, noisy simulator, saving/loading the trained model.
