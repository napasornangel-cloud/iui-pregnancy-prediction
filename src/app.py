import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# =============================
# Paths / fixed settings
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent  # src/ → project root

BASE_MODEL_PATH = BASE_DIR / "models_test2/final_model/XGBoost_Baseline_calibration_base_model.joblib"
CALIBRATOR_PATH = BASE_DIR / "models_test2/final_model/isotonic_calibrator_final_xgb.joblib"
SHAP_IMG        = BASE_DIR / "reports_test2/figures/shap_final_xgb/SHAP_Beeswarm_Final_XGBoost_Baseline.png"
TEMPORAL_TABLE  = BASE_DIR / "reports_temporal/tables/Temporal_Validation_Results.xlsx"

FINAL_FEATURES = [
    "Uterine_Factors",
    "Total_Female_Pathology",
    "Ovulatory_Factors",
    "Cycle_Day",
    "Post_TPMSC",
    "First_Count",
    "Pre_Count",
    "Gynecological_Surgical_History",
    "Post_Count",
    "Delta_Motile",
    "Age_Female",
    "First_Progressive_Motile",
    "First_Volume",
    "Menstrual_Interval_Days",
    "First_Motile",
    "BMI_InfertilityType_Interaction",
]

LOW_TIER_CUTOFF  = 0.023256
HIGH_TIER_CUTOFF = 0.055556
VERY_LOW_CUTOFF  = LOW_TIER_CUTOFF

DISPLAY_MAP = {
    "Uterine_Factors":                 "Uterine factor",
    "Total_Female_Pathology":          "Total female pathology burden",
    "Ovulatory_Factors":               "Ovulatory factor",
    "Cycle_Day":                       "IUI cycle day",
    "Post_TPMSC":                      "Postwash TPMSC",
    "First_Count":                     "Initial sperm count",
    "Pre_Count":                       "Prewash sperm count",
    "Gynecological_Surgical_History":  "Gynecologic surgery history",
    "Post_Count":                      "Postwash sperm count",
    "Delta_Motile":                    "Change in total motility",
    "Age_Female":                      "Female age",
    "First_Progressive_Motile":        "Initial progressive motility",
    "First_Volume":                    "Initial semen volume",
    "Menstrual_Interval_Days":         "Menstrual cycle interval (days)",
    "First_Motile":                    "Initial total motility",
    "BMI_InfertilityType_Interaction": "BMI × infertility type",
}

REQUIRED_RAW_COLUMNS = [
    "Uterine_Factors", "Tubal_Factors", "Ovarian_Factors",
    "Ovulatory_Factors", "Cervical_Factors", "Endometriosis_Factors",
    "Multisystem_Factors", "Cycle_Day", "Post_TPMSC", "First_Count",
    "Pre_Count", "Gynecological_Surgical_History", "Post_Count",
    "Post_Motile", "Pre_Motile", "Age_Female", "First_Progressive_Motile",
    "First_Volume", "Menstrual_Interval_Days", "First_Motile",
    "Body_Mass_Index", "Infertility_Type",
]

# =============================
# Helpers
# =============================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def assign_probability_tier(p_cal):
    if p_cal < LOW_TIER_CUTOFF:
        return "Tier 1 (Low Probability)"
    if p_cal < HIGH_TIER_CUTOFF:
        return "Tier 2 (Intermediate Probability)"
    return "Tier 3 (High Probability)"

def very_low_flag(p_cal):
    return "Yes" if p_cal < VERY_LOW_CUTOFF else "No"

def tier_color_name(tier):
    if "Tier 1" in tier: return "red"
    if "Tier 2" in tier: return "orange"
    return "green"

def cum3_from_p1(p1):
    return 1.0 - (1.0 - p1) ** 3

def group_display_name(raw_name):
    return DISPLAY_MAP.get(str(raw_name), str(raw_name).replace("_", " "))

@st.cache_resource
def load_base_model():
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")
    return joblib.load(BASE_MODEL_PATH)

@st.cache_resource
def load_calibrator():
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")
    return joblib.load(CALIBRATOR_PATH)

def compute_engineered_features(df_raw):
    df = df_raw.copy()
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required raw columns:\n- " + "\n- ".join(missing))
    for c in REQUIRED_RAW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_rows = df[REQUIRED_RAW_COLUMNS].isna().any(axis=1)
    if bad_rows.any():
        raise ValueError(f"Missing or non-numeric values at row indices: {list(df.index[bad_rows][:10])}")
    df["Total_Female_Pathology"]          = (df["Uterine_Factors"] + df["Tubal_Factors"] +
                                              df["Ovarian_Factors"] + df["Ovulatory_Factors"] +
                                              df["Cervical_Factors"] + df["Endometriosis_Factors"] +
                                              df["Multisystem_Factors"])
    df["Delta_Motile"]                    = df["Post_Motile"] - df["Pre_Motile"]
    df["BMI_InfertilityType_Interaction"] = df["Body_Mass_Index"] * df["Infertility_Type"]
    return df[FINAL_FEATURES].copy()

def predict_raw_and_calibrated(X):
    model      = load_base_model()
    calibrator = load_calibrator()
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = np.clip(calibrator.predict(p_raw), 0, 1)
    return p_raw, p_cal

def local_explain_one_row(X_row, top_k=8):
    model     = load_base_model()
    xgb_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):         shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:                 shap_vals = shap_vals[:, :, 1]
    sv   = shap_vals.reshape(-1)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(np.ravel(base)) >= 2 else float(np.ravel(base)[0])
    base  = float(base)
    order = np.argsort(np.abs(sv))[::-1][:min(top_k, len(sv))]
    z, rows = base, []
    for j in order:
        label    = group_display_name(X_row.columns[j])
        dz       = float(sv[j])
        p_before = sigmoid(z); z += dz; p_after = sigmoid(z)
        delta_pp = (p_after - p_before) * 100.0
        arrow    = "↑" if delta_pp >= 0 else "↓"
        rows.append((label, round(delta_pp, 2),
                     f"{arrow} increases model-estimated probability" if delta_pp >= 0
                     else f"{arrow} decreases model-estimated probability"))
    return pd.DataFrame(rows, columns=["Item", "Approx. change (pp)", "Meaning"])

def plot_shap_waterfall(X_row):
    model     = load_base_model()
    xgb_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb_model)
    exp       = explainer(X_row)
    fig, ax   = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(exp[0], max_display=10, show=False)
    plt.tight_layout()
    return fig

def plot_gauge(p_cal):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"aspect": "equal"})
    theta   = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=18, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - p_cal * np.pi, 300)
    color = "#d73027" if p_cal < LOW_TIER_CUTOFF else ("#fee08b" if p_cal < HIGH_TIER_CUTOFF else "#1a9850")
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=18, solid_capstyle="round")
    ax.text(0, -0.15, f"{p_cal:.1%}", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.45, "Calibrated probability", ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.2); ax.axis("off")
    return fig

def interpretation_text(p_cal):
    tier = assign_probability_tier(p_cal)
    if "Tier 1" in tier:
        return ("This result falls in the **low-probability** group. "
                "In the study cohort, this group had the lowest cumulative pregnancy rates across 1–3 cycles.")
    if "Tier 2" in tier:
        return ("This result falls in the **intermediate-probability** group. "
                "This profile suggests a moderate expected IUI yield.")
    return ("This result falls in the **high-probability** group. "
            "In the study cohort, this group had the highest cumulative pregnancy rates across 1–3 cycles.")

def render_prediction_summary(p_raw, p_cal, show_gauge=True):
    p_cum3 = cum3_from_p1(p_cal)
    tier   = assign_probability_tier(p_cal)
    vlow   = very_low_flag(p_cal)
    color  = tier_color_name(tier)

    if show_gauge:
        col_gauge, col_metrics = st.columns([1, 2])
        with col_gauge:
            st.pyplot(plot_gauge(p_cal), use_container_width=True)
        with col_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw per-cycle prob.", f"{p_raw:.1%}")
            c2.metric("Within 3 cycles",     f"{p_cum3:.1%}")
            c3.metric("Risk tier",            tier)
            st.caption(f"Very low flag: {vlow}")
            st.markdown(f"**Tier:** :{color}[{tier}]")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw per-cycle prob.",        f"{p_raw:.1%}")
        c2.metric("Calibrated per-cycle prob.", f"{p_cal:.1%}")
        c3.metric("Within 3 cycles",            f"{p_cum3:.1%}")
        c4.metric("Risk tier",                  tier)
        st.caption(f"Very low flag: {vlow}")
        st.markdown(f"**Tier:** :{color}[{tier}]")

    st.markdown("### Clinical interpretation")
    st.info(interpretation_text(p_cal))

def build_example_input():
    return pd.DataFrame([{
        "Uterine_Factors": 0, "Tubal_Factors": 0, "Ovarian_Factors": 0,
        "Ovulatory_Factors": 0, "Cervical_Factors": 0, "Endometriosis_Factors": 0,
        "Multisystem_Factors": 0, "Cycle_Day": 14, "Post_TPMSC": 10.0,
        "First_Count": 40.0, "Pre_Count": 35.0, "Gynecological_Surgical_History": 0,
        "Post_Count": 12.0, "Post_Motile": 80.0, "Pre_Motile": 60.0,
        "Age_Female": 32.0, "First_Progressive_Motile": 40.0, "First_Volume": 2.5,
        "Menstrual_Interval_Days": 28.0, "First_Motile": 60.0,
        "Body_Mass_Index": 22.0, "Infertility_Type": 1,
    }])

# =============================
# Page config
# =============================
st.set_page_config(page_title="IUI Pregnancy Probability Tool", layout="wide")
st.title("🔬 IUI Pregnancy Probability Tool")
st.caption("Research prototype · For counseling and risk stratification only · Not a clinical decision system")

with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown(f"""
**What the tool reports**
- **Raw per-cycle probability:** direct XGBoost model output
- **Calibrated per-cycle probability:** post-hoc isotonic-calibrated probability for counseling
- **Within 3 cycles (approx.):** 1 − (1 − p)³
- **Risk tier:** Low / Intermediate / High

**Risk-tier cutoffs (calibrated probability)**
- Tier 1 (Low): p < {LOW_TIER_CUTOFF:.6f}
- Tier 2 (Intermediate): {LOW_TIER_CUTOFF:.6f} ≤ p < {HIGH_TIER_CUTOFF:.6f}
- Tier 3 (High): p ≥ {HIGH_TIER_CUTOFF:.6f}

**Important**
- Outputs are statistical estimates, not guarantees.
- Always use in conjunction with clinical judgment.
- This tool is for research use only.
""")

example_df = build_example_input()
st.download_button("⬇️ Download sample CSV template",
                   example_df.to_csv(index=False).encode("utf-8"),
                   "iui_input_template.csv", "text/csv")

tab1, tab2, tab3, tab4 = st.tabs(["✏️ Manual entry", "📂 Batch CSV", "🔍 Explanation", "ℹ️ Model info"])

# =================================
# TAB 1: Manual entry
# =================================
with tab1:
    st.subheader("Manual entry — one patient-cycle")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Female & cycle factors**")
            age_female      = st.number_input("Female age", 18.0, 55.0, 32.0, 1.0)
            bmi             = st.number_input("BMI", 10.0, 60.0, 22.0, 0.1)
            menstrual_interval_days = st.number_input("Menstrual cycle interval (days)", 15.0, 180.0, 28.0, 1.0)
            cycle_day       = st.number_input("IUI cycle day", 1.0, 40.0, 14.0, 1.0)
            infertility_type = st.selectbox("Infertility type",
                options=[1, 0],
                format_func=lambda x: "Primary infertility" if x == 1 else "Secondary infertility")

            st.markdown("**Female pathology factors** (0 = absent, 1 = present)")
            uterine_factors       = st.selectbox("Uterine factors",       [0, 1], index=0)
            tubal_factors         = st.selectbox("Tubal factors",         [0, 1], index=0)
            ovarian_factors       = st.selectbox("Ovarian factors",       [0, 1], index=0)
            ovulatory_factors     = st.selectbox("Ovulatory factors",     [0, 1], index=0)
            cervical_factors      = st.selectbox("Cervical factors",      [0, 1], index=0)
            endometriosis_factors = st.selectbox("Endometriosis factors", [0, 1], index=0)
            multisystem_factors   = st.selectbox("Multisystem factors",   [0, 1], index=0)
            gyn_surgery           = st.selectbox("Gynecologic surgery history", [0, 1], index=0)

        with col2:
            st.markdown("**Semen parameters — initial sample**")
            first_volume      = st.number_input("Initial semen volume (mL)",          0.0, 20.0, 2.5, 0.1)
            first_count       = st.number_input("Initial sperm count (×10⁶/mL)",      0.0, 500.0, 40.0, 0.1)
            first_motile      = st.number_input("Initial total motility (%)",          0.0, 100.0, 60.0, 0.1)
            first_prog_motile = st.number_input("Initial progressive motility (%)",    0.0, 100.0, 40.0, 0.1)

            st.markdown("**Semen parameters — prewash**")
            pre_count   = st.number_input("Prewash sperm count (×10⁶/mL)",  0.0, 500.0, 35.0, 0.1)
            pre_motile  = st.number_input("Prewash motility (%)",            0.0, 100.0, 60.0, 0.1)

            st.markdown("**Semen parameters — postwash**")
            post_count  = st.number_input("Postwash sperm count (×10⁶/mL)", 0.0, 500.0, 12.0, 0.1)
            post_tpmsc  = st.number_input("Postwash TPMSC (×10⁶)",          0.0, 500.0, 10.0, 0.1)
            post_motile = st.number_input("Postwash motility (%)",           0.0, 100.0, 80.0, 0.1)

        submitted = st.form_submit_button("🚀 Run prediction")

    if submitted:
        try:
            manual_df = pd.DataFrame([{
                "Uterine_Factors": uterine_factors, "Tubal_Factors": tubal_factors,
                "Ovarian_Factors": ovarian_factors, "Ovulatory_Factors": ovulatory_factors,
                "Cervical_Factors": cervical_factors, "Endometriosis_Factors": endometriosis_factors,
                "Multisystem_Factors": multisystem_factors, "Cycle_Day": cycle_day,
                "Post_TPMSC": post_tpmsc, "First_Count": first_count,
                "Pre_Count": pre_count, "Gynecological_Surgical_History": gyn_surgery,
                "Post_Count": post_count, "Post_Motile": post_motile, "Pre_Motile": pre_motile,
                "Age_Female": age_female, "First_Progressive_Motile": first_prog_motile,
                "First_Volume": first_volume, "Menstrual_Interval_Days": menstrual_interval_days,
                "First_Motile": first_motile, "Body_Mass_Index": bmi,
                "Infertility_Type": infertility_type,
            }])
            X_manual = compute_engineered_features(manual_df)
            p_raw, p_cal = predict_raw_and_calibrated(X_manual)
            st.success("✅ Prediction complete")
            render_prediction_summary(float(p_raw[0]), float(p_cal[0]), show_gauge=True)
            st.markdown("### Top factors influencing this prediction")
            expl = local_explain_one_row(X_manual, top_k=8)
            st.dataframe(expl, use_container_width=True)
        except Exception as e:
            st.error(str(e))

# =================================
# TAB 2: Batch CSV
# =================================
with tab2:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV with the required raw input columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upl_calc")

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write("Preview"); st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🚀 Run batch prediction"):
            try:
                X = compute_engineered_features(df_raw)
                p_raw, p_cal = predict_raw_and_calibrated(X)
                out = df_raw.copy()
                out["raw_per_cycle_probability"]                     = p_raw
                out["calibrated_per_cycle_probability"]              = p_cal
                out["approx_cumulative_probability_within_3_cycles"] = [cum3_from_p1(float(x)) for x in p_cal]
                out["risk_tier"]                                     = [assign_probability_tier(float(x)) for x in p_cal]
                out["very_low_flag"]                                 = [very_low_flag(float(x)) for x in p_cal]
                st.success("✅ Prediction complete")
                render_prediction_summary(float(p_raw[0]), float(p_cal[0]), show_gauge=False)
                st.markdown("### Prediction table")
                show_cols = ["raw_per_cycle_probability", "calibrated_per_cycle_probability",
                             "approx_cumulative_probability_within_3_cycles", "risk_tier", "very_low_flag"]
                st.dataframe(out[show_cols], use_container_width=True)
                st.download_button("⬇️ Download results (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "iui_predictions_out.csv", "text/csv")
            except Exception as e:
                st.error(str(e))

# =================================
# TAB 3: Explanation
# =================================
with tab3:
    st.subheader("Explain one row from CSV")
    uploaded2 = st.file_uploader("Upload CSV", type=["csv"], key="upl_exp")

    if uploaded2 is not None:
        df_raw2 = pd.read_csv(uploaded2)
        st.write("Preview"); st.dataframe(df_raw2.head(), use_container_width=True)
        row_idx = st.number_input("Row index", 0, max(0, len(df_raw2)-1), 0, 1)

        if st.button("🔍 Explain this row"):
            try:
                X2    = compute_engineered_features(df_raw2)
                x_row = X2.iloc[[int(row_idx)]].copy()
                p_raw, p_cal = predict_raw_and_calibrated(x_row)
                render_prediction_summary(float(p_raw[0]), float(p_cal[0]), show_gauge=True)

                st.markdown("### SHAP waterfall plot")
                try:
                    fig = plot_shap_waterfall(x_row)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    pass

                st.markdown("### Top factors (table)")
                expl = local_explain_one_row(x_row, top_k=8)
                st.dataframe(expl, use_container_width=True)
                st.caption("Explanations describe model output, not causal effects.")
            except Exception as e:
                st.error(str(e))

# =================================
# TAB 4: Model info
# =================================
with tab4:
    st.subheader("Model information")

    st.markdown("""
**Final model**
- Algorithm: XGBoost with 16 selected predictors
- Imbalance handling: No resampling (scale_pos_weight)
- Probability calibration: Post-hoc isotonic regression
- Feature selection: Gain-based importance with elbow detection

**Validation**
- Primary: Patient-level GroupShuffleSplit (80/20), random seed 42
- Secondary: Temporal holdout — trained 2017–2023, tested 2024–2025
""")

    st.markdown("### Primary vs Temporal validation")
    comparison_df = pd.DataFrame({
        "Metric":    ["PR-AUC", "ROC-AUC", "Brier", "Sensitivity", "Specificity", "NPV"],
        "Primary":   [0.1386, 0.6808, 0.2207, 0.902, 0.430, 0.984],
        "Temporal":  [0.2434, 0.7894, 0.2053, 0.818, 0.461, 0.975],
    })
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("### Final model predictors")
    feature_table = pd.DataFrame({
        "Feature":      FINAL_FEATURES,
        "Display name": [group_display_name(f) for f in FINAL_FEATURES]
    })
    st.dataframe(feature_table, use_container_width=True)

    st.markdown("### Global SHAP explanation")
    if SHAP_IMG.exists():
        st.image(str(SHAP_IMG), use_container_width=True)
    else:
        st.info("SHAP image not found.")

    st.markdown("---")
    st.markdown("""
**⚠️ Disclaimer**

This tool is a research prototype developed for academic purposes.
It is intended to support — not replace — clinical judgment.
Outputs represent statistical estimates based on a single-center retrospective cohort
and may not generalize to all clinical settings.
Do not use this tool as the sole basis for clinical decision-making.
""")