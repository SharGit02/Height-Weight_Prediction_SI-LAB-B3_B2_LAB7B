"""
Height & Weight Predictor -- Streamlit Application
===================================================
Predicts weight from height+gender (Linear Regression) and
height from weight+gender (Polynomial Regression).
Models are loaded from height_weight_models.pkl.
Metrics are computed live from the loaded pipeline on the dataset.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Height & Weight Predictor",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #f0f4f8; }
.block-container { padding: 1.5rem 2.5rem 2rem 2.5rem; max-width: 1300px; }

/* Hero */
.hero-wrapper {
    background: linear-gradient(135deg, #ffffff 60%, #e8f4fd 100%);
    border-radius: 20px; padding: 2rem 2.5rem 1.8rem 2.5rem;
    margin-bottom: 1.5rem; border: 1px solid #dbeafe;
    box-shadow: 0 4px 24px rgba(59,130,246,0.10);
    display: flex; align-items: center; justify-content: space-between;
}
.hero-title  { font-size: 2.4rem; font-weight: 800; color: #1e3a5f; margin: 0 0 0.3rem 0; line-height: 1.1; }
.hero-sub    { font-size: 1.05rem; color: #5b8bc9; font-weight: 400; margin: 0; }
.hero-icons  { font-size: 3rem; }

/* Section headers */
.sec-hd  { font-size: 1.25rem; font-weight: 700; color: #1e3a5f; margin: 0 0 0.1rem 0; }
.sec-sub { font-size: 0.85rem; color: #7a9bbf; margin-bottom: 0.8rem; }

/* Prediction card wrapper */
.pred-card {
    background: #ffffff; border-radius: 18px; padding: 1.5rem 1.8rem;
    border: 1px solid #dbeafe; box-shadow: 0 2px 18px rgba(59,130,246,0.08);
    margin-bottom: 1.2rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
    border: 2px solid #38bdf8; border-radius: 18px; padding: 1.6rem;
    text-align: center; box-shadow: 0 4px 20px rgba(56,189,248,0.18);
    min-height: 190px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.res-label  { font-size: 0.85rem; color: #0369a1; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.res-value  { font-size: 3.4rem; font-weight: 800; color: #0c4a6e; line-height: 1; }
.res-unit   { font-size: 1.3rem; font-weight: 500; color: #0369a1; }
.res-badge  { background: #d1fae5; color: #065f46; border-radius: 20px; padding: 0.2rem 0.9rem; font-size: 0.78rem; font-weight: 600; margin-top: 0.7rem; }
.res-hint   { font-size: 0.95rem; color: #94a3b8; }

/* Chart cards */
.chart-card {
    background: #ffffff; border-radius: 16px; padding: 1.1rem 1.3rem 0.5rem 1.3rem;
    border: 1px solid #e2ecf6; box-shadow: 0 2px 12px rgba(59,130,246,0.07);
    margin-bottom: 0.8rem;
}
.chart-title { font-size: 0.95rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.1rem; }

/* Model Performance table */
.metric-tbl { width: 100%; border-collapse: collapse; font-size: 0.87rem; margin-top: 0.5rem; }
.metric-tbl th { background: #eff6ff; color: #1e3a5f; font-weight: 700; padding: 0.65rem 0.9rem; text-align: left; border-bottom: 2px solid #bfdbfe; }
.metric-tbl td { padding: 0.6rem 0.9rem; border-bottom: 1px solid #e2ecf6; color: #334155; }
.metric-tbl tr:last-child td { border-bottom: none; }
.metric-good { color: #16a34a; font-weight: 700; }
.metric-na   { color: #94a3b8; font-style: italic; }
.info-chip   { background: #eff6ff; color: #3b82f6; border-radius: 8px; padding: 0.25rem 0.7rem; font-size: 0.78rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.3rem; margin-top: 0.6rem; }

/* Model info cards */
.mcard { background: linear-gradient(135deg,#f8faff,#eff6ff); border: 1px solid #bfdbfe; border-radius: 14px; padding: 1.1rem 1.4rem; display: flex; gap: 0.9rem; align-items: flex-start; margin-bottom: 0.7rem; }
.mcard-ico   { font-size: 1.8rem; }
.mcard-title { font-size: 0.98rem; font-weight: 700; color: #1e3a5f; margin: 0; }
.mcard-type  { font-size: 0.8rem; color: #3b82f6; font-weight: 600; }
.mcard-desc  { font-size: 0.8rem; color: #64748b; }

/* Section wrapper cards */
.section-box {
    background: #ffffff; border-radius: 18px; padding: 1.4rem 1.6rem;
    border: 1px solid #e2ecf6; box-shadow: 0 2px 14px rgba(59,130,246,0.07);
    margin-bottom: 1rem;
}

/* Footer */
.footer { background: linear-gradient(90deg,#1e3a5f,#0f6b8a); color: #93c5fd; border-radius: 12px; padding: 0.85rem 2rem; display: flex; align-items: center; justify-content: space-between; margin-top: 1.5rem; font-size: 0.82rem; }
.footer-r { color: #7dd3fc; font-style: italic; }

/* Buttons */
.stButton > button { border-radius: 10px !important; font-weight: 600 !important; transition: all 0.2s ease !important; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_FILE    = "height_weight_models.pkl"
CSV_NAMES     = ["height_weight.csv","height_weight_data.csv","data.csv","dataset.csv"]
MALE_COLOR    = "#3b82f6"
FEMALE_COLOR  = "#f472b6"
CURVE_COLOR   = "#0ea5e9"
LAYOUT_BASE   = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#334155"),
    margin=dict(l=44, r=16, t=36, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2ecf6", borderwidth=1,
                x=0.98, xanchor="right", y=0.98, yanchor="top"),
)


# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(path):
    if not os.path.exists(path):
        return None, "Model file '{}' not found.".format(path)
    try:
        return joblib.load(path), None
    except AttributeError as exc:
        if "_RemainderColsList" in str(exc):
            return None, (
                "scikit-learn version mismatch: the PKL was saved with scikit-learn 1.6.1 "
                "but a different version is active. "
                "Run: pip install scikit-learn==1.6.1  then restart Streamlit."
            )
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_csv():
    for name in CSV_NAMES:
        if os.path.exists(name):
            try:
                return pd.read_csv(name), name
            except Exception:
                pass
    return None, None


def make_synthetic(n=600, seed=42):
    rng  = np.random.default_rng(seed)
    mn   = n // 2
    fn   = n - mn
    mh   = rng.normal(175, 7, mn);  mw = 0.50*mh - 19 + rng.normal(0, 5, mn)
    fh   = rng.normal(162, 6, fn);  fw = 0.42*fh - 10 + rng.normal(0, 4, fn)
    df   = pd.DataFrame({
        "Height_cm": np.clip(np.concatenate([mh,fh]), 140, 210).round(1),
        "Weight_kg": np.clip(np.concatenate([mw,fw]),  40, 145).round(1),
        "Gender":    ["Male"]*mn + ["Female"]*fn,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def detect_cols(df):
    h=w=g=None
    for c in df.columns:
        cl=c.lower()
        if   any(k in cl for k in ["height","ht","_cm"]): h=c
        elif any(k in cl for k in ["weight","wt","_kg"]): w=c
        elif "gender" in cl or "sex" in cl:               g=c
    return h,w,g


# ─── METRIC COMPUTATION (live from model + data) ──────────────────────────────
@st.cache_data(show_spinner=False)
def compute_metrics(_models, _df):
    """Compute R2, MAE, RMSE for both pipelines on the dataset."""
    results = {}
    if _models is None or _df is None:
        return results

    has_h = "Height_cm" in _df.columns
    has_w = "Weight_kg" in _df.columns
    has_g = "Gender"    in _df.columns

    # Weight Prediction: predict Weight_kg from Height_cm + Gender
    if has_h and has_w and has_g:
        try:
            pipe  = _models["Weight Prediction"]["model"]
            X     = _df[["Height_cm","Gender"]].copy()
            y     = _df["Weight_kg"].values
            y_hat = pipe.predict(X)
            results["weight"] = {
                "r2":   round(float(r2_score(y, y_hat)), 4),
                "mae":  round(float(mean_absolute_error(y, y_hat)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y, y_hat))), 4),
            }
        except Exception as e:
            results["weight_err"] = str(e)

    # Height Prediction: predict Height_cm from Weight_kg + Gender
    if has_h and has_w and has_g:
        try:
            pipe  = _models["Height Prediction"]["model"]
            X     = _df[["Weight_kg","Gender"]].copy()
            y     = _df["Height_cm"].values
            y_hat = pipe.predict(X)
            results["height"] = {
                "r2":   round(float(r2_score(y, y_hat)), 4),
                "mae":  round(float(mean_absolute_error(y, y_hat)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y, y_hat))), 4),
            }
        except Exception as e:
            results["height_err"] = str(e)

    return results


# ─── PREDICTION HELPERS ───────────────────────────────────────────────────────
def pred_weight(bundle, height_cm, gender):
    try:
        pipe = bundle["Weight Prediction"]["model"]
        return float(pipe.predict(pd.DataFrame({"Height_cm":[height_cm],"Gender":[gender]}))[0])
    except Exception:
        return None

def pred_height(bundle, weight_kg, gender):
    try:
        pipe = bundle["Height Prediction"]["model"]
        return float(pipe.predict(pd.DataFrame({"Weight_kg":[weight_kg],"Gender":[gender]}))[0])
    except Exception:
        return None

def curve_weight(bundle, gender):
    h = np.linspace(140, 210, 250)
    try:
        pipe = bundle["Weight Prediction"]["model"]
        w = pipe.predict(pd.DataFrame({"Height_cm": h, "Gender": gender}))
        return h, w
    except Exception:
        return None, None


# ─── LOAD EVERYTHING ──────────────────────────────────────────────────────────
models, model_err = load_models(MODEL_FILE)
raw_df, csv_name  = load_csv()

if raw_df is not None:
    h_col, w_col, g_col = detect_cols(raw_df)
    if h_col and w_col:
        rename_map = {h_col: "Height_cm", w_col: "Weight_kg"}
        if g_col: rename_map[g_col] = "Gender"
        df = raw_df.rename(columns=rename_map)
        if "Gender" not in df.columns:
            df["Gender"] = "Unknown"
        src_note = "Loaded from {}".format(csv_name)
    else:
        df = make_synthetic(); src_note = "Using synthetic data (columns unrecognised)"
else:
    df = make_synthetic()
    src_note = "Using synthetic demonstration data (place height_weight.csv here to use real data)"

# Compute metrics live
metrics = compute_metrics(models, df)

gender_colors = {"Male": MALE_COLOR, "Female": FEMALE_COLOR}
has_gender    = "Gender" in df.columns and df["Gender"].nunique() > 1

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("mode","Predict Weight"),("prediction",None),
             ("pred_unit",""),("pred_label",""),("pred_badge","")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
  <div>
    <div class="hero-title">&#128207; Height &amp; Weight Predictor</div>
    <div class="hero-sub">Explore measurements. Predict with machine learning.</div>
  </div>
  <div class="hero-icons">&#129485; &#128208; &#129486;</div>
</div>
""", unsafe_allow_html=True)

if model_err:
    st.error("Could not load model: " + model_err)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="pred-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-hd">&#127919; Make a Prediction</div>'
            '<div class="sec-sub">Choose what you want to predict, enter the value and click the button.</div>',
            unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("&#x2696;&#xFE0F;  Predict Weight", use_container_width=True,
                     type="primary" if st.session_state.mode=="Predict Weight" else "secondary",
                     key="bw"):
            st.session_state.mode = "Predict Weight"
            st.session_state.prediction = None
            st.rerun()
    with b2:
        if st.button("&#x1F4CF;  Predict Height", use_container_width=True,
                     type="primary" if st.session_state.mode=="Predict Height" else "secondary",
                     key="bh"):
            st.session_state.mode = "Predict Height"
            st.session_state.prediction = None
            st.rerun()

    st.markdown("<div style='margin-top:0.7rem'></div>", unsafe_allow_html=True)

    ic, gc = st.columns(2)
    with ic:
        if st.session_state.mode == "Predict Weight":
            user_val = st.number_input("Height (cm)", 100.0, 250.0, 170.0, 0.5, key="iv")
        else:
            user_val = st.number_input("Weight (kg)",  30.0, 200.0,  70.0, 0.5, key="iv")
    with gc:
        gender = st.selectbox("Gender", ["Male","Female"], key="gs")

    st.markdown("<div style='margin-top:0.3rem'></div>", unsafe_allow_html=True)
    if st.button("&#9889; Predict Now", type="primary", key="pn"):
        if models is None:
            st.error("Models not loaded.")
        elif st.session_state.mode == "Predict Weight":
            st.session_state.prediction = pred_weight(models, user_val, gender)
            st.session_state.pred_label = "Predicted Weight"
            st.session_state.pred_unit  = "kg"
            st.session_state.pred_badge = "Linear Regression model"
        else:
            st.session_state.prediction = pred_height(models, user_val, gender)
            st.session_state.pred_label = "Predicted Height"
            st.session_state.pred_unit  = "cm"
            st.session_state.pred_badge = "Polynomial Regression model"

with right_col:
    if st.session_state.prediction is not None:
        v = st.session_state.prediction
        st.markdown(
            '<div class="result-card">'
            '<div class="res-label">{}</div>'
            '<div class="res-value">{:.1f}<span class="res-unit"> {}</span></div>'
            '<div class="res-badge">&#9989; Based on {}</div>'
            '</div>'.format(
                st.session_state.pred_label, v,
                st.session_state.pred_unit, st.session_state.pred_badge),
            unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
          <div style="font-size:2.8rem;margin-bottom:0.5rem">&#128202;</div>
          <div class="res-hint">Enter a value and click<br><strong>Predict Now</strong></div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close pred-card
st.markdown("<hr style='border:none;border-top:1px solid #dbeafe;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXPLORE THE DATA — 4 charts in a 2x2 grid
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hd">&#128202; Explore the Data</div>'
            '<div class="sec-sub">Visualise the dataset and model insights. &nbsp;'
            '<em style="color:#94a3b8;font-size:0.8rem">{}</em></div>'.format(src_note),
            unsafe_allow_html=True)

CHART_H = 320

# ── Row 1: Scatter  |  Height Dist ───────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="medium")

with r1c1:
    st.markdown('<div class="chart-card"><div class="chart-title">&#128200; Height vs Weight Scatter Plot</div></div>', unsafe_allow_html=True)
    fig1 = px.scatter(df, x="Height_cm", y="Weight_kg",
                      color="Gender" if has_gender else None,
                      color_discrete_map=gender_colors,
                      labels={"Height_cm":"Height (cm)","Weight_kg":"Weight (kg)"},
                      opacity=0.6)
    fig1.update_traces(marker=dict(size=5))
    fig1.update_layout(**LAYOUT_BASE, height=CHART_H,
                       xaxis=dict(title="Height (cm)", gridcolor="#eef2f7", zeroline=False),
                       yaxis=dict(title="Weight (kg)", gridcolor="#eef2f7", zeroline=False))
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with r1c2:
    st.markdown('<div class="chart-card"><div class="chart-title">&#128202; Height Distribution</div></div>', unsafe_allow_html=True)
    if has_gender:
        fig2 = go.Figure()
        for gv, col in gender_colors.items():
            fig2.add_trace(go.Histogram(x=df[df["Gender"]==gv]["Height_cm"], name=gv,
                                        marker_color=col, opacity=0.72, xbins=dict(size=2),
                                        hovertemplate="%{y} people<extra>"+gv+"</extra>"))
        fig2.update_layout(barmode="overlay")
    else:
        fig2 = px.histogram(df, x="Height_cm", nbins=30, color_discrete_sequence=[MALE_COLOR])
    fig2.update_layout(**LAYOUT_BASE, height=CHART_H,
                       xaxis=dict(title="Height (cm)", gridcolor="#eef2f7", zeroline=False),
                       yaxis=dict(title="Count",       gridcolor="#eef2f7", zeroline=False))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Row 2: Weight Dist  |  Regression Curve (dynamic gender) ─────────────────
r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    st.markdown('<div class="chart-card"><div class="chart-title">&#9878;&#65039; Weight Distribution</div></div>', unsafe_allow_html=True)
    if has_gender:
        fig3 = go.Figure()
        for gv, col in gender_colors.items():
            fig3.add_trace(go.Histogram(x=df[df["Gender"]==gv]["Weight_kg"], name=gv,
                                        marker_color=col, opacity=0.72, xbins=dict(size=2.5),
                                        hovertemplate="%{y} people<extra>"+gv+"</extra>"))
        fig3.update_layout(barmode="overlay")
    else:
        fig3 = px.histogram(df, x="Weight_kg", nbins=30, color_discrete_sequence=[FEMALE_COLOR])
    fig3.update_layout(**LAYOUT_BASE, height=CHART_H,
                       xaxis=dict(title="Weight (kg)", gridcolor="#eef2f7", zeroline=False),
                       yaxis=dict(title="Count",       gridcolor="#eef2f7", zeroline=False))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with r2c2:
    # Header + gender selector side-by-side inside native Streamlit columns
    hd_col, sel_col = st.columns([3, 1])
    with hd_col:
        st.markdown('<div class="chart-card"><div class="chart-title">&#128201; Regression Curve &nbsp;'
                    '<span style="font-size:0.78rem;font-weight:400;color:#64748b">'
                    '(Predicted Weight from Height)</span></div></div>', unsafe_allow_html=True)
    with sel_col:
        curve_g = st.selectbox("Gender", ["Male","Female"], key="cg", label_visibility="collapsed")

    # Build figure — re-executes every time curve_g changes (fully dynamic)
    fig4 = go.Figure()

    # Scatter: actual data for selected gender
    pts = df[df["Gender"] == curve_g] if has_gender else df
    fig4.add_trace(go.Scatter(
        x=pts["Height_cm"], y=pts["Weight_kg"],
        mode="markers", name="Actual Data",
        marker=dict(color=gender_colors.get(curve_g, MALE_COLOR), size=4, opacity=0.45),
        hovertemplate="H: %{x:.1f} cm  W: %{y:.1f} kg<extra></extra>"))

    # Curve: predicted by the live pipeline
    if models is not None:
        hv, wv = curve_weight(models, curve_g)
        if hv is not None:
            fig4.add_trace(go.Scatter(
                x=hv, y=wv, mode="lines", name="Model Prediction",
                line=dict(color=CURVE_COLOR, width=3),
                hovertemplate="H: %{x:.1f} cm  Pred W: %{y:.1f} kg<extra></extra>"))
    else:
        hv = np.linspace(140, 210, 250)
        fig4.add_trace(go.Scatter(
            x=hv, y=0.42*hv - (8 if curve_g=="Male" else 15),
            mode="lines", name="Illustrative Curve",
            line=dict(color=CURVE_COLOR, width=3, dash="dot")))

    fig4.update_layout(**LAYOUT_BASE, height=CHART_H,
                       xaxis=dict(title="Height (cm)", gridcolor="#eef2f7", zeroline=False),
                       yaxis=dict(title="Weight (kg)", gridcolor="#eef2f7", zeroline=False))
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr style='border:none;border-top:1px solid #dbeafe;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE  +  MODEL INFORMATION (side by side)
# ══════════════════════════════════════════════════════════════════════════════
pc, ic = st.columns(2, gap="large")

with pc:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">&#128201; Model Performance</div>'
                '<div class="sec-sub">Metrics computed live on the current dataset using the trained pipelines.</div>',
                unsafe_allow_html=True)

    wp_name = models["Weight Prediction"]["model_name"] if models else "Linear Regression"
    hp_name = models["Height Prediction"]["model_name"] if models else "Polynomial Regression"

    wm = metrics.get("weight")
    hm = metrics.get("height")

    def cell(val):
        if val is None:
            return '<span class="metric-na">&mdash;</span>'
        return '<span class="metric-good">{:.4f}</span>'.format(val)

    st.markdown("""
    <table class="metric-tbl">
      <thead>
        <tr><th>Model</th><th>R&sup2;</th><th>MAE</th><th>RMSE</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Weight Prediction</strong><br><small style="color:#64748b">{}</small></td>
          <td>{}</td><td>{}</td><td>{}</td>
        </tr>
        <tr>
          <td><strong>Height Prediction</strong><br><small style="color:#64748b">{}</small></td>
          <td>{}</td><td>{}</td><td>{}</td>
        </tr>
      </tbody>
    </table>
    """.format(
        wp_name,
        cell(wm["r2"]   if wm else None),
        cell(wm["mae"]  if wm else None),
        cell(wm["rmse"] if wm else None),
        hp_name,
        cell(hm["r2"]   if hm else None),
        cell(hm["mae"]  if hm else None),
        cell(hm["rmse"] if hm else None),
    ), unsafe_allow_html=True)

    if not wm and not hm:
        st.markdown('<div class="info-chip">&#8505;&#65039; Could not compute metrics (check data columns).</div>',
                    unsafe_allow_html=True)
    else:
        src_label = csv_name if csv_name else "Height-Weight Data from kaggle"
        st.markdown('<div style="font-size:0.76rem;color:#94a3b8;margin-top:0.5rem">Computed on: {}</div>'.format(src_label),
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with ic:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">&#9881;&#65039; Model Information</div>'
                '<div class="sec-sub">Pipelines loaded from <code>{}</code></div>'.format(MODEL_FILE),
                unsafe_allow_html=True)

    if models:
        wp = models["Weight Prediction"]
        hp = models["Height Prediction"]
        wp_n = wp["model_name"]; wp_in = wp["input_col"]; wp_tgt = wp["target_col"]
        hp_n = hp["model_name"]; hp_in = hp["input_col"]; hp_tgt = hp["target_col"]
    else:
        wp_n="Linear Regression";    wp_in="Height_cm"; wp_tgt="Weight_kg"
        hp_n="Polynomial Regression"; hp_in="Weight_kg"; hp_tgt="Height_cm"

    st.markdown("""
    <div class="mcard">
      <div class="mcard-ico">&#9878;&#65039;</div>
      <div>
        <div class="mcard-title">Weight Prediction</div>
        <div class="mcard-type">{}</div>
        <div class="mcard-desc">Predicts <strong>{}</strong> from <strong>{}</strong> + Gender</div>
      </div>
    </div>
    <div class="mcard">
      <div class="mcard-ico">&#9999;&#65039;</div>
      <div>
        <div class="mcard-title">Height Prediction</div>
        <div class="mcard-type">{}</div>
        <div class="mcard-desc">Predicts <strong>{}</strong> from <strong>{}</strong> + Gender</div>
      </div>
    </div>
    """.format(wp_n, wp_tgt, wp_in, hp_n, hp_tgt, hp_in), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
csv_label = csv_name if csv_name else "synthetic data"
st.markdown("""
<div class="footer">
  <div>&#128196; Dataset: <strong>{}</strong> &nbsp;|&nbsp; &#129302; Models: <strong>{}</strong></div>
  <div class="footer-r">&#128161; Better data. Smarter predictions.</div>
</div>
""".format(csv_label, MODEL_FILE), unsafe_allow_html=True)
