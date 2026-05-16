import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go

# kept the UI simple, a doctor needs to read this fast
st.set_page_config(
    page_title="Cervical Cancer Risk Estimator",
    page_icon="🔬",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("models/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(model):
    # TreeExplainer is deterministic so caching it is safe
    return shap.TreeExplainer(model)

try:
    model = load_model()
    explainer = load_explainer(model)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Feature definitions ───────────────────────────────────────────────────────

# keeping the exact column names the model was trained on
FEATURE_NAMES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "STDs: Number of diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
]

# ── Sidebar inputs ────────────────────────────────────────────────────────────

st.sidebar.header("Patient Information")
st.sidebar.markdown("Fill in the details below and click **Predict Risk**.")

age = st.sidebar.slider("Age", min_value=13, max_value=84, value=30)
num_partners = st.sidebar.slider("Number of sexual partners", 0, 28, 2)
first_intercourse = st.sidebar.slider("Age at first sexual intercourse", 10, 32, 17)
num_pregnancies = st.sidebar.slider("Number of pregnancies", 0, 11, 1)

smokes = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
smokes_val = 1 if smokes == "Yes" else 0
smokes_years = st.sidebar.slider("Smoking duration (years)", 0.0, 37.0, 0.0, step=0.5,
                                  disabled=(smokes_val == 0))
smokes_packs = st.sidebar.slider("Packs per year", 0.0, 37.0, 0.0, step=0.5,
                                  disabled=(smokes_val == 0))

st.sidebar.markdown("---")

hc = st.sidebar.selectbox("Hormonal contraceptives?", ["No", "Yes"])
hc_val = 1 if hc == "Yes" else 0
hc_years = st.sidebar.slider("Hormonal contraceptive duration (years)", 0.0, 30.0, 0.0,
                               step=0.5, disabled=(hc_val == 0))

iud = st.sidebar.selectbox("IUD use?", ["No", "Yes"])
iud_val = 1 if iud == "Yes" else 0
iud_years = st.sidebar.slider("IUD duration (years)", 0.0, 19.0, 0.0, step=0.5,
                                disabled=(iud_val == 0))

st.sidebar.markdown("---")

stds = st.sidebar.selectbox("History of STDs?", ["No", "Yes"])
stds_val = 1 if stds == "Yes" else 0
stds_count = st.sidebar.slider("Number of STDs", 0, 8, 0, disabled=(stds_val == 0))

st.sidebar.markdown("**Specific STD history** (select all that apply)")
std_types = {
    "STDs:condylomatosis": st.sidebar.checkbox("Condylomatosis"),
    "STDs:cervical condylomatosis": st.sidebar.checkbox("Cervical condylomatosis"),
    "STDs:vaginal condylomatosis": st.sidebar.checkbox("Vaginal condylomatosis"),
    "STDs:vulvo-perineal condylomatosis": st.sidebar.checkbox("Vulvo-perineal condylomatosis"),
    "STDs:syphilis": st.sidebar.checkbox("Syphilis"),
    "STDs:pelvic inflammatory disease": st.sidebar.checkbox("Pelvic inflammatory disease"),
    "STDs:genital herpes": st.sidebar.checkbox("Genital herpes"),
    "STDs:molluscum contagiosum": st.sidebar.checkbox("Molluscum contagiosum"),
    "STDs:AIDS": st.sidebar.checkbox("AIDS"),
    "STDs:HIV": st.sidebar.checkbox("HIV"),
    "STDs:Hepatitis B": st.sidebar.checkbox("Hepatitis B"),
    "STDs:HPV": st.sidebar.checkbox("HPV"),
}
stds_num_diag = st.sidebar.slider("Number of STD diagnoses", 0, 9, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Prior diagnoses**")
dx_cancer = st.sidebar.checkbox("Prior cancer diagnosis")
dx_cin = st.sidebar.checkbox("Prior CIN diagnosis")
dx_hpv = st.sidebar.checkbox("Prior HPV diagnosis")
dx = st.sidebar.checkbox("General Dx flag")

# ── Build input vector ────────────────────────────────────────────────────────

input_data = {
    "Age": age,
    "Number of sexual partners": num_partners,
    "First sexual intercourse": first_intercourse,
    "Num of pregnancies": num_pregnancies,
    "Smokes": smokes_val,
    "Smokes (years)": smokes_years if smokes_val else 0.0,
    "Smokes (packs/year)": smokes_packs if smokes_val else 0.0,
    "Hormonal Contraceptives": hc_val,
    "Hormonal Contraceptives (years)": hc_years if hc_val else 0.0,
    "IUD": iud_val,
    "IUD (years)": iud_years if iud_val else 0.0,
    "STDs": stds_val,
    "STDs (number)": stds_count if stds_val else 0,
    **{k: int(v) for k, v in std_types.items()},
    "STDs: Number of diagnosis": stds_num_diag,
    "Dx:Cancer": int(dx_cancer),
    "Dx:CIN": int(dx_cin),
    "Dx:HPV": int(dx_hpv),
    "Dx": int(dx),
}

input_df = pd.DataFrame([input_data])
# reorder to match training column order
input_df = input_df.reindex(columns=FEATURE_NAMES, fill_value=0)

# ── Main panel ────────────────────────────────────────────────────────────────

st.title("Cervical Cancer Risk Estimator")
st.markdown(
    "> **Disclaimer:** This is an academic project, not medical advice. "
    "Do not use this tool for clinical decisions. Always consult a qualified healthcare provider."
)
st.markdown("---")

predict_btn = st.button("Predict Risk", type="primary", use_container_width=False)

if not model_loaded:
    st.warning(
        "Model file not found at `models/xgboost_model.pkl`. "
        "Run the training notebook first to generate the model."
    )
elif predict_btn:
    risk_prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 1])

    # ── Gauge chart ───────────────────────────────────────────────────────────
    with col1:
        st.subheader("Predicted Risk Probability")

        if risk_prob < 0.3:
            gauge_color = "#2ecc71"
            risk_label = "Low Risk"
        elif risk_prob < 0.6:
            gauge_color = "#f39c12"
            risk_label = "Moderate Risk"
        else:
            gauge_color = "#e74c3c"
            risk_label = "High Risk"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": risk_label, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": risk_prob * 100,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Top contributing features ─────────────────────────────────────────────
    with col2:
        st.subheader("Top 3 Contributing Factors")

        shap_vals = explainer.shap_values(input_df)[0]
        shap_series = pd.Series(shap_vals, index=FEATURE_NAMES)

        # sort by absolute value but keep sign for direction
        top3 = shap_series.abs().nlargest(3).index
        top3_vals = shap_series[top3]

        for feat, val in top3_vals.items():
            direction = "increases" if val > 0 else "decreases"
            direction_icon = "▲" if val > 0 else "▼"
            feat_value = input_df[feat].iloc[0]
            st.markdown(
                f"**{direction_icon} {feat}**  \n"
                f"Value: `{feat_value}` — {direction} risk "
                f"(SHAP: {val:+.3f})"
            )
            st.markdown("---")

    # ── Interpretation note ───────────────────────────────────────────────────
    st.markdown(
        f"**Model output:** {risk_prob:.1%} probability of a positive biopsy result.  \n"
        "SHAP values above show which inputs pushed the model's prediction up or down "
        "relative to the average patient in the training data."
    )

else:
    st.info("Set patient parameters in the sidebar and click **Predict Risk** to see results.")
