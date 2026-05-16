import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# make sure results folder exists before we try to save anything there
os.makedirs("results", exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────

# the raw file has '?' for missing values — caught me off guard the first time
df = pd.read_csv("data/risk_factors_cervical_cancer.csv", na_values="?")

# these columns had way too many missing values to be useful (>80% missing)
# dropping them was painful but they were just noise
cols_to_drop = [
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# median imputation — tried mean, basically no difference on this dataset
df.fillna(df.median(numeric_only=True), inplace=True)

target_col = "Biopsy"
X = df.drop(columns=[target_col, "Hinselmann", "Schiller", "Citology"], errors="ignore")
y = df[target_col]

# ── Load trained model ────────────────────────────────────────────────────────

# model is saved after training in the notebook — needs to exist before running this
model_path = "models/xgboost_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}. "
        "Run the notebook first to train and save the model."
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

# ── SHAP explainer ────────────────────────────────────────────────────────────

# TreeExplainer is the right one for XGBoost — much faster than KernelExplainer
# and gives exact SHAP values rather than approximations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# shap_values shape is (n_samples, n_features) for binary classification with XGBoost
# positive values push toward class 1 (cancer positive), negative toward class 0
print(f"SHAP values computed. Shape: {shap_values.shape}")

# ── Plot 1: Summary bar chart ─────────────────────────────────────────────────
# this shows mean absolute SHAP value per feature — basically global feature importance
# clinically useful: tells you which factors matter most across all patients

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Mean |SHAP Value| — Global Feature Importance", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("results/shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/shap_summary_bar.png")

# ── Plot 2: Beeswarm plot ─────────────────────────────────────────────────────
# beeswarm is more informative than bar — shows the direction of effect too
# red dots = high feature value, blue = low
# e.g. if red dots for "Num sexual partners" are on the right side, high count → higher risk
# this is the plot I'd show to a clinician to explain what the model is doing

plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Beeswarm — Feature Impact on Prediction", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("results/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/shap_summary_beeswarm.png")

# ── Plot 3: Waterfall plot for a single high-risk patient ─────────────────────
# waterfall shows exactly why the model predicted what it did for ONE patient
# super useful for explainability — you can point to it and say "here's why this flag fired"

# find the positive-class sample with highest predicted probability
# this gives us the most interesting case to explain
proba = model.predict_proba(X)[:, 1]
high_risk_idx = int(np.argmax(proba))
print(f"Showing waterfall for sample index {high_risk_idx} "
      f"(predicted risk: {proba[high_risk_idx]:.3f})")

# Explanation object needed for the newer waterfall API
# tried shap.force_plot first but waterfall is cleaner for static images
explanation = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=X.iloc[high_risk_idx].values,
    feature_names=X.columns.tolist(),
)

plt.figure(figsize=(10, 7))
shap.waterfall_plot(explanation, max_display=12, show=False)
plt.title(
    f"SHAP Waterfall — High-Risk Patient (Predicted Risk: {proba[high_risk_idx]:.1%})",
    fontsize=12,
    pad=10,
)
plt.tight_layout()
plt.savefig("results/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/shap_waterfall.png")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\nAll SHAP plots saved to results/")
print("Top 5 features by mean |SHAP value|:")
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=X.columns
).sort_values(ascending=False)
print(mean_abs_shap.head(5).to_string())
