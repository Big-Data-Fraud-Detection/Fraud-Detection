#pip install streamlit plotly polars lightgbm shap scikit-learn kagglehub
#python -m streamlit run fraud_dashboard.py
"""
PaySim Fraud Detection — Streamlit Dashboard
Run: streamlit run fraud_dashboard.py
Requires: streamlit polars lightgbm shap scikit-learn kagglehub plotly
"""

import os
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="PaySim Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; }
  .block-container { padding-top: 1.5rem; }
  h1 { font-size: 1.5rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data + Model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading dataset & training model…")
def load_and_train():
    import polars as pl
    import kagglehub
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_recall_curve, roc_auc_score,
        average_precision_score, confusion_matrix,
    )

    path = kagglehub.dataset_download("ealaxi/paysim1")
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pl.read_csv(os.path.join(path, csv_file))

    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    df = df.with_columns(
        pl.col("type").replace_strict(type_map, default=-1).alias("type_enc"),
        pl.col("type").is_in(["TRANSFER", "CASH_OUT"]).cast(pl.Int8).alias("is_risky_type"),
    )
    df = df.with_columns([
        (pl.col("oldbalanceOrg") - pl.col("amount") - pl.col("newbalanceOrig")).alias("orig_balance_error"),
        (pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")).alias("dest_balance_error"),
        (pl.col("newbalanceOrig") == 0).cast(pl.Int8).alias("orig_zeroed"),
        (pl.col("newbalanceDest") == 0).cast(pl.Int8).alias("dest_zeroed"),
        (pl.col("oldbalanceOrg") >= pl.col("amount")).cast(pl.Int8).alias("sufficient_balance"),
        (pl.col("oldbalanceDest") == 0).cast(pl.Int8).alias("dest_was_empty"),
        (pl.col("amount") / (pl.col("oldbalanceOrg") + 1)).alias("amount_ratio_orig"),
        (pl.col("amount") / (pl.col("oldbalanceDest") + 1)).alias("amount_ratio_dest"),
    ])
    df = df.with_columns([
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("oldbalanceOrg").log1p().alias("log_oldbal_orig"),
        pl.col("newbalanceOrig").log1p().alias("log_newbal_orig"),
        pl.col("oldbalanceDest").log1p().alias("log_oldbal_dest"),
        pl.col("newbalanceDest").log1p().alias("log_newbal_dest"),
    ])
    df = df.with_columns([
        (
            (pl.col("oldbalanceOrg") > 0) &
            (pl.col("newbalanceOrig") == 0) &
            (pl.col("amount") == pl.col("oldbalanceOrg"))
        ).cast(pl.Int8).alias("exact_drain"),
        (pl.col("step") % 24).alias("hour_of_day"),
        (pl.col("step") % 168).alias("hour_of_week"),
        (pl.col("step") // 24).alias("day"),
        ((pl.col("step") % 24) < 6).cast(pl.Int8).alias("is_night"),
        pl.col("nameOrig").str.starts_with("C").cast(pl.Int8).alias("orig_is_customer"),
        pl.col("nameDest").str.starts_with("C").cast(pl.Int8).alias("dest_is_customer"),
        pl.col("nameDest").str.starts_with("M").cast(pl.Int8).alias("dest_is_merchant"),
    ])

    feature_cols = [
        "type_enc", "is_risky_type", "log_amount", "step",
        "hour_of_day", "hour_of_week", "day", "is_night",
        "log_oldbal_orig", "log_newbal_orig",
        "log_oldbal_dest", "log_newbal_dest",
        "orig_balance_error", "dest_balance_error",
        "orig_zeroed", "dest_zeroed",
        "sufficient_balance", "dest_was_empty",
        "amount_ratio_orig", "amount_ratio_dest",
        "exact_drain",
        "orig_is_customer", "dest_is_customer", "dest_is_merchant",
    ]

    df_clean = df.drop_nulls(subset=feature_cols + ["isFraud"])
    X = df_clean.select(feature_cols).to_numpy().astype(np.float32)
    y = df_clean["isFraud"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    neg, pos = np.bincount(y_train)
    scale_pos_wt = neg / pos

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data   = lgb.Dataset(X_val,   label=y_val,   reference=train_data)

    params = {
        "objective": "binary", "metric": "average_precision",
        "scale_pos_weight": scale_pos_wt,
        "learning_rate": 0.05, "num_leaves": 127,
        "min_child_samples": 50, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 0.1,
        "verbose": -1, "seed": 42,
    }
    model = lgb.train(
        params, train_data, num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_proba = model.predict(X_val)
    prec, rec, thresholds = precision_recall_curve(y_val, val_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx    = np.argmax(f1s[:-1])
    best_thresh = thresholds[best_idx]

    test_proba = model.predict(X_test)
    test_pred  = (test_proba >= best_thresh).astype(int)

    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc  = average_precision_score(y_test, test_proba)
    cm      = confusion_matrix(y_test, test_pred)

    # SHAP (sampled)
    import shap
    sample_idx = np.random.RandomState(42).choice(len(X_test), min(2000, len(X_test)), replace=False)
    explainer  = shap.TreeExplainer(model)
    raw        = explainer.shap_values(X_test[sample_idx])
    shap_vals  = raw[1] if isinstance(raw, list) else raw

    # fraud-by-type
    type_stats = (
        df_clean.group_by("type")
        .agg([pl.col("isFraud").mean().alias("fraud_rate"), pl.len().alias("count")])
        .sort("fraud_rate", descending=True)
    )

    return dict(
        model=model, feature_cols=feature_cols,
        X_test=X_test, y_test=y_test,
        test_proba=test_proba, test_pred=test_pred,
        prec=prec, rec=rec, thresholds=thresholds, f1s=f1s,
        best_thresh=best_thresh, best_idx=best_idx,
        roc_auc=roc_auc, pr_auc=pr_auc, cm=cm,
        shap_vals=shap_vals, sample_idx=sample_idx,
        type_stats=type_stats,
        n_total=len(df_clean), n_fraud=int(y.sum()),
    )


# ── Load ─────────────────────────────────────────────────────────────────────
data = load_and_train()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    thresh = st.slider(
        "Decision threshold",
        min_value=0.01, max_value=0.99,
        value=float(round(data["best_thresh"], 4)),
        step=0.01,
        help="Adjust to trade precision vs. recall",
    )
    top_n = st.slider("Feature importance: top N", 5, len(data["feature_cols"]), 15)
    shap_n = st.slider("SHAP: top N features", 5, len(data["feature_cols"]), 15)
    show_raw = st.checkbox("Show raw score table", value=False)

    st.markdown("---")
    st.caption(f"**Dataset**: PaySim · {data['n_total']:,} txns")
    st.caption(f"**Fraud**: {data['n_fraud']:,} ({data['n_fraud']/data['n_total']:.3%})")
    st.caption(f"**Model**: LightGBM · best iter {data['model'].best_iteration}")

# ── Derived at chosen threshold ───────────────────────────────────────────────
pred = (data["test_proba"] >= thresh).astype(int)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix as cm_fn

prec_s = precision_score(data["y_test"], pred, zero_division=0)
rec_s  = recall_score(data["y_test"], pred, zero_division=0)
f1_s   = f1_score(data["y_test"], pred, zero_division=0)
cm_live = cm_fn(data["y_test"], pred)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 PaySim Fraud Detection Dashboard")
st.caption("LightGBM classifier · PaySim synthetic payments dataset")

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("ROC-AUC",   f"{data['roc_auc']:.4f}")
k2.metric("PR-AUC",    f"{data['pr_auc']:.4f}")
k3.metric("Precision", f"{prec_s:.4f}", delta=f"thresh={thresh:.2f}")
k4.metric("Recall",    f"{rec_s:.4f}")
k5.metric("F1",        f"{f1_s:.4f}")

st.divider()

# ── Row 1: PR curve + Score dist ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision-Recall Curve")
    # find closest threshold index for interactive marker
    thresh_idx = np.argmin(np.abs(data["thresholds"] - thresh))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["rec"][:-1], y=data["prec"][:-1],
        mode="lines", name=f"PR curve (AUC={data['pr_auc']:.3f})",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[data["rec"][thresh_idx]], y=[data["prec"][thresh_idx]],
        mode="markers", name=f"Threshold={thresh:.2f}",
        marker=dict(color="red", size=10, symbol="circle"),
    ))
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=340, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Score Distribution")
    bins = np.linspace(0, 1, 50)
    legit_hist, _ = np.histogram(data["test_proba"][data["y_test"] == 0], bins=bins)
    fraud_hist, _ = np.histogram(data["test_proba"][data["y_test"] == 1], bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bin_centers, y=legit_hist, name="Legit",
        marker_color="rgba(16,185,129,0.55)", width=0.018,
    ))
    fig2.add_trace(go.Bar(
        x=bin_centers, y=fraud_hist, name="Fraud",
        marker_color="rgba(239,68,68,0.7)", width=0.018,
    ))
    fig2.add_vline(x=thresh, line_dash="dash", line_color="black",
                   annotation_text=f"τ={thresh:.2f}", annotation_position="top right")
    fig2.update_layout(
        xaxis_title="Fraud probability", yaxis_title="Count (log scale)",
        yaxis_type="log", barmode="overlay",
        height=340, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Confusion matrix + Fraud by type ──────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Confusion Matrix")
    tn, fp, fn, tp = cm_live.ravel()
    labels = [["TN", "FP"], ["FN", "TP"]]
    values = [[tn, fp], [fn, tp]]
    text   = [[f"<b>{labels[i][j]}</b><br>{values[i][j]:,}" for j in range(2)] for i in range(2)]

    fig3 = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Legit", "Predicted Fraud"],
        y=["True Legit", "True Fraud"],
        text=text, texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Blues", showscale=True,
    ))
    fig3.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("True Positive Rate", f"{tp/(tp+fn):.3%}")
    c2.metric("False Positive Rate", f"{fp/(fp+tn):.3%}")
    c3.metric("Flagged transactions", f"{tp+fp:,}")

with col4:
    st.subheader("Fraud Rate by Transaction Type")
    import polars as pl
    ts = data["type_stats"].to_pandas()
    fig4 = px.bar(
        ts, x="type", y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale=["#dbeafe", "#1d4ed8"],
        text=ts["fraud_rate"].apply(lambda v: f"{v:.3%}"),
        labels={"fraud_rate": "Fraud rate", "type": "Type"},
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        coloraxis_showscale=False,
        yaxis_tickformat=".2%",
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("💡 Only TRANSFER and CASH_OUT transactions contain fraud in PaySim.")

st.divider()

# ── Row 3: Feature importance + SHAP ─────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader(f"Feature Importance (Gain) — Top {top_n}")
    importance = data["model"].feature_importance(importance_type="gain")
    feat_df = sorted(zip(data["feature_cols"], importance), key=lambda x: x[1], reverse=True)
    names, vals = zip(*feat_df[:top_n])
    fig5 = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#6366f1",
    ))
    fig5.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 28 + 60),
        margin=dict(l=10, r=20, t=20, b=40),
        xaxis_title="Gain",
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader(f"SHAP Summary — Top {shap_n} Features")
    mean_abs_shap = np.abs(data["shap_vals"]).mean(axis=0)
    shap_sorted   = sorted(zip(data["feature_cols"], mean_abs_shap), key=lambda x: x[1], reverse=True)
    snames, svals = zip(*shap_sorted[:shap_n])
    fig6 = go.Figure(go.Bar(
        x=svals, y=snames, orientation="h",
        marker_color="#f59e0b",
    ))
    fig6.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(300, shap_n * 28 + 60),
        margin=dict(l=10, r=20, t=20, b=40),
        xaxis_title="Mean |SHAP value|",
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── SHAP Beeswarm scatter ─────────────────────────────────────────────────────
st.subheader("SHAP Beeswarm (sampled 2,000 test points)")
top_shap_feats = [name for name, _ in shap_sorted[:shap_n]]
top_shap_idx   = [data["feature_cols"].index(f) for f in top_shap_feats]

rows = []
for rank, (fi, fn_) in enumerate(zip(top_shap_idx, top_shap_feats)):
    sv = data["shap_vals"][:, fi]
    fv = data["X_test"][data["sample_idx"], fi]
    fv_norm = (fv - fv.min()) / (np.ptp(fv) + 1e-9)
    jitter = np.random.RandomState(rank).uniform(-0.35, 0.35, size=len(sv))
    for s, fn_norm, j in zip(sv, fv_norm, jitter):
        rows.append({"feature": fn_, "SHAP value": float(s),
                     "Feature value (norm)": float(fn_norm), "y": rank + j})

import pandas as pd
bee_df = pd.DataFrame(rows)

fig7 = px.scatter(
    bee_df, x="SHAP value", y="y",
    color="Feature value (norm)",
    color_continuous_scale=["#3b82f6", "#f97316"],
    hover_data={"feature": True, "SHAP value": ":.3f", "y": False},
    labels={"y": ""},
    height=max(350, shap_n * 32 + 80),
    opacity=0.45,
)
fig7.update_traces(marker=dict(size=4))
fig7.update_layout(
    yaxis=dict(
        tickvals=list(range(shap_n)),
        ticktext=top_shap_feats,
        autorange="reversed",
    ),
    coloraxis_colorbar=dict(title="Feature value", tickvals=[0, 0.5, 1],
                            ticktext=["Low", "Mid", "High"]),
    margin=dict(l=10, r=20, t=10, b=40),
    xaxis_title="SHAP value (impact on model output)",
)
fig7.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
st.plotly_chart(fig7, use_container_width=True)

# ── Optional raw score table ──────────────────────────────────────────────────
if show_raw:
    st.subheader("Sample scored transactions (first 500 test rows)")
    sample_df = pd.DataFrame(
        data["X_test"][:500], columns=data["feature_cols"]
    )
    sample_df["fraud_prob"]  = data["test_proba"][:500].round(4)
    sample_df["predicted"]   = (data["test_proba"][:500] >= thresh).astype(int)
    sample_df["actual"]      = data["y_test"][:500]
    sample_df["correct"]     = (sample_df["predicted"] == sample_df["actual"])
    st.dataframe(
        sample_df[["fraud_prob", "predicted", "actual", "correct"]
                  + data["feature_cols"][:6]],
        use_container_width=True, height=300,
    )
