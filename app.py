"""
Automated Machine Learning (AutoML) Predictive Maintenance Dashboard
======================================================================
A dataset-agnostic Streamlit app that ingests ANY failure/fault-style CSV
(AI4I 2020, Steel Plates Faults, Pump Sensor Data, Device Failure
Classification, etc.), automatically detects identifier/leakage columns
and the target, runs a 4-model competition (XGBoost, Random Forest,
Gradient Boosting, Extra Trees), crowns a Champion Model by Macro F1,
and serves Top-2 risk predictions with confidence scores.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier

st.set_page_config(page_title="AutoML Predictive Maintenance Dashboard", layout="wide")

MAX_ROWS_FOR_TRAINING = 50_000  # guardrail for big Kaggle dumps (e.g. Pump Sensor Data)


# ============================================================
# SECTION 0 — HELPER / DETECTION FUNCTIONS
# (pure logic, no Streamlit calls, so they're easy to unit test)
# ============================================================

def detect_identifier_columns(df: pd.DataFrame) -> list:
    """Flag columns that look like row identifiers, not features."""
    n = len(df)
    candidates = []
    name_patterns = {"id", "index", "udi", "unnamed: 0", "unnamed:0"}
    for col in df.columns:
        name = col.lower().strip().replace(" ", "_").replace("-", "_")
        if name in name_patterns or name.endswith("_id") or name.startswith("id_") or "unnamed" in name:
            candidates.append(col)
            continue
        # high-cardinality object or perfectly sequential integer columns = identifiers
        if df[col].nunique(dropna=True) == n:
            if df[col].dtype == object:
                candidates.append(col)
            elif pd.api.types.is_integer_dtype(df[col]):
                sorted_vals = df[col].dropna().sort_values().reset_index(drop=True)
                if len(sorted_vals) > 1:
                    start = sorted_vals.iloc[0]
                    if (sorted_vals == np.arange(start, start + len(sorted_vals))).all():
                        candidates.append(col)
    return candidates


def detect_binary_columns(df: pd.DataFrame, exclude=()) -> list:
    """Columns whose only non-null values are 0/1 -- candidate failure-mode flags."""
    binary_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            try:
                if set(pd.Series(vals).astype(float)) <= {0.0, 1.0}:
                    binary_cols.append(col)
            except (ValueError, TypeError):
                continue
    return binary_cols


def detect_aggregate_leakage(df: pd.DataFrame, binary_cols: list) -> list:
    """
    A column like 'machine_failure' is often EXACTLY the logical OR of several
    specific failure flags (TWF, HDF, PWF...). That makes it pure leakage if
    kept as a feature -- find and flag it generically, without hardcoding names.
    """
    aggregates = []
    for cand in binary_cols:
        rest = [c for c in binary_cols if c != cand]
        if len(rest) >= 2:
            agg = (df[rest].fillna(0).sum(axis=1) > 0).astype(int)
            cand_vals = df[cand].fillna(0).astype(int).reset_index(drop=True)
            if (cand_vals == agg.reset_index(drop=True)).all():
                aggregates.append(cand)
    return aggregates


def guess_target_column(df: pd.DataFrame, exclude=()) -> str:
    """Best-effort guess at the label column by common naming conventions."""
    priority_keywords = [
        "failure_type", "target", "label", "class", "fault",
        "status", "failure", "outcome", "result", "defect",
    ]
    available = [c for c in df.columns if c not in exclude]
    lower_map = {c.lower(): c for c in available}
    for keyword in priority_keywords:
        for lower_name, original in lower_map.items():
            if keyword == lower_name or keyword in lower_name:
                return original
    return available[-1] if available else None


def sanitize_column_name(name: str) -> str:
    """
    Strip units like '[K]' / '[rpm]' and any non-alphanumeric characters.
    XGBoost specifically REJECTS feature names containing '[', ']', or '<',
    so raw Kaggle headers (e.g. 'Air temperature [K]') must be cleaned before
    they ever reach a model -- not just for cosmetics.
    """
    import re
    name = re.sub(r"\[.*?\]", "", str(name))
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name or "col"


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def build_features(raw_df, cols_to_drop, target_mode, target_col, binary_indicator_cols):
    """
    Deterministic cleaning + encoding pipeline. Cached on (data, UI selections)
    so re-running this on every widget interaction doesn't re-clean the whole
    dataset from scratch.
    """
    df = raw_df.drop(columns=list(cols_to_drop), errors="ignore").copy()

    # Keep the original-cased names for display (e.g. 'TWF') before sanitizing
    # the actual column names for model compatibility.
    original_label_lookup = {col: col for col in binary_indicator_cols}

    # Sanitize ALL column names up front (handles raw Kaggle headers like
    # 'Air temperature [K]' or 'Torque [Nm]' that would otherwise crash XGBoost).
    rename_map, seen = {}, {}
    for col in df.columns:
        clean = sanitize_column_name(col)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        rename_map[col] = clean
    df = df.rename(columns=rename_map)
    original_label_lookup = {rename_map.get(k, k): v for k, v in original_label_lookup.items()}
    binary_indicator_cols = [rename_map.get(c, c) for c in binary_indicator_cols]
    if target_mode == "single":
        target_col = rename_map.get(target_col, target_col)

    if target_mode == "multi_binary":
        # Rarer failure modes are usually more mechanistically specific, so
        # when two flags overlap on one row, the rarer one should win.
        # Sorting most-frequent-first and assigning in that order means the
        # rarest column gets applied LAST, naturally overwriting the rest.
        ordered_cols = sorted(binary_indicator_cols, key=lambda c: df[c].sum(), reverse=True)
        class_names = ["Normal"] + [original_label_lookup.get(c, c) for c in ordered_cols]
        df["target"] = 0
        for i, col in enumerate(ordered_cols, start=1):
            df.loc[df[col] == 1, "target"] = i
        df = df.drop(columns=binary_indicator_cols)
    else:
        df = df.rename(columns={target_col: "target"})
        if df["target"].dtype == object or str(df["target"].dtype) == "category":
            target_le = LabelEncoder()
            df["target"] = target_le.fit_transform(df["target"].astype(str))
            class_names = list(target_le.classes_)
        else:
            uniques = sorted(df["target"].dropna().unique())
            target_map = {v: i for i, v in enumerate(uniques)}
            df["target"] = df["target"].map(target_map)
            class_names = [str(v) for v in uniques]

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    feature_cols = [c for c in df.columns if c != "target"]
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Impute missing values -- real Kaggle exports (e.g. Pump Sensor Data) are rarely clean
    for c in numeric_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in categorical_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    return df, feature_cols, numeric_cols, categorical_cols, class_names


@st.cache_resource(show_spinner=False)
def train_and_compare(X_train, X_test, y_train, y_test):
    """
    The AutoML core. Cached as a RESOURCE (not data) because it returns live
    fitted model objects. Streamlit only re-runs this when X_train/y_train
    actually change -- clicking buttons or tweaking unrelated widgets won't
    silently re-train 4 models on every rerun.
    """
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    n_classes = len(np.unique(y_train))

    candidates = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=42, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
    }

    results, trained_models = [], {}
    for name, model in candidates.items():
        start = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weights)
        elapsed = time.time() - start

        preds = model.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "Macro F1-Score": round(macro_f1, 4),
            "Training Time (s)": round(elapsed, 2),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("Macro F1-Score", ascending=False).reset_index(drop=True)
    champion_name = results_df.iloc[0]["Model"]
    return results_df, trained_models, champion_name


# ============================================================
# SECTION 1 — HEADER & FILE UPLOAD
# ============================================================
st.title("🤖 AutoML Predictive Maintenance Dashboard")
st.write(
    "Upload **any** fault/failure-style dataset (AI4I 2020, Steel Plates Faults, "
    "Pump Sensor Data, Device Failure Classification, ...) and this dashboard will "
    "clean it, profile it, race four classifiers against each other, and surface "
    "the top-2 most likely failure modes for every row -- automatically."
)

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

raw_df = load_data(uploaded_file)
st.success(f"Loaded **{raw_df.shape[0]:,} rows** x **{raw_df.shape[1]} columns**.")
st.dataframe(raw_df.head())


# ============================================================
# SECTION 2 — DYNAMIC INGESTION: IDENTIFIER / LEAKAGE / TARGET DETECTION
# ============================================================
st.divider()
st.header("Step 1: Cleaning & Target Detection")

id_candidates = detect_identifier_columns(raw_df)
binary_cols_all = detect_binary_columns(raw_df)
aggregate_cols = detect_aggregate_leakage(raw_df, binary_cols_all)
binary_indicator_cols = [c for c in binary_cols_all if c not in aggregate_cols]

auto_drop = sorted(set(id_candidates + aggregate_cols))

st.write(
    "Auto-detected identifier and leakage-aggregate columns are pre-selected below. "
    "Add or remove columns if the heuristics missed something."
)
cols_to_drop = st.multiselect(
    "Columns to drop (identifiers / leakage)",
    options=list(raw_df.columns),
    default=auto_drop,
)

remaining_cols = [c for c in raw_df.columns if c not in cols_to_drop]
usable_binary_cols = [c for c in binary_indicator_cols if c not in cols_to_drop]

if len(usable_binary_cols) >= 2:
    target_mode = "multi_binary"
    target_col = None
    st.info(
        f"Detected **{len(usable_binary_cols)} binary failure-indicator columns** "
        f"-> will be combined into one multi-class target: `{usable_binary_cols}`"
    )
else:
    target_mode = "single"
    guessed_target = guess_target_column(raw_df, exclude=set(cols_to_drop))
    default_idx = remaining_cols.index(guessed_target) if guessed_target in remaining_cols else 0
    target_col = st.selectbox(
        "Confirm the target (label) column",
        options=remaining_cols,
        index=default_idx,
    )
    if raw_df[target_col].nunique() > 20 and pd.api.types.is_numeric_dtype(raw_df[target_col]):
        st.warning(
            f"'{target_col}' has {raw_df[target_col].nunique()} unique numeric values -- "
            "this looks more like a continuous variable than a classification target. "
            "Double check your selection."
        )

df, feature_cols, numeric_cols, categorical_cols, class_names = build_features(
    raw_df, tuple(cols_to_drop), target_mode, target_col, tuple(usable_binary_cols)
)

# Guardrail for very large Kaggle dumps (Pump Sensor Data can be 200k+ rows)
if len(df) > MAX_ROWS_FOR_TRAINING:
    st.warning(
        f"Dataset has {len(df):,} rows. Using a stratified sample of "
        f"{MAX_ROWS_FOR_TRAINING:,} rows so the 4-model race stays fast."
    )
    df, _ = train_test_split(
        df, train_size=MAX_ROWS_FOR_TRAINING, stratify=df["target"], random_state=42
    )
    df = df.reset_index(drop=True)

st.write(f"**Final feature set ({len(feature_cols)} columns):**", feature_cols)
st.write(f"**Detected classes ({len(class_names)}):**", class_names)


# ============================================================
# SECTION 3 — AUTOMATED EDA
# ============================================================
st.divider()
st.header("Step 2: Automated Exploratory Data Analysis")

corr_source = df[numeric_cols + ["target"]] if numeric_cols else df[["target"]]
corr = corr_source.corr()

if "target" in corr.columns and len(numeric_cols) >= 1:
    target_corr = corr["target"].drop("target").abs().sort_values(ascending=False)
    top2 = target_corr.index[:2].tolist() if len(target_corr) >= 2 else (numeric_cols * 2)[:2]
else:
    top2 = (numeric_cols * 2)[:2] if numeric_cols else [None, None]

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

with eda_col2:
    st.subheader(f"Class Separation: {top2[0]} vs {top2[1]}")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df, x=top2[0], y=top2[1], hue="target",
        palette="tab10", alpha=0.6, s=30, ax=ax2,
    )
    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles, class_names, title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig2)

eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    st.subheader("Target Class Distribution")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.countplot(x="target", data=df, hue="target", palette="rocket", legend=False, ax=ax3)
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=30, ha="right")
    ax3.set_xlabel("")
    st.pyplot(fig3)

with eda_col4:
    most_informative = top2[0]
    st.subheader(f"{most_informative} by Class")
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="target", y=most_informative, data=df, hue="target", palette="mako", legend=False, ax=ax4)
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=30, ha="right")
    ax4.set_xlabel("")
    st.pyplot(fig4)


# ============================================================
# SECTION 4 — SPLIT, SCALE
# ============================================================
st.divider()
st.header("Step 3: Train / Test Split & Scaling")

X = df[feature_cols]
y = df["target"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    st.write("Used a stratified 80/20 split (class ratios preserved).")
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    st.warning(
        "At least one class had too few rows to stratify -- fell back to a plain "
        "random 80/20 split. Predictions for that class may be unreliable."
    )

scaler = StandardScaler()
X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
if numeric_cols:
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

st.write(f"Train: {X_train_scaled.shape[0]:,} rows | Test: {X_test_scaled.shape[0]:,} rows")


# ============================================================
# SECTION 5 — AUTOML MODEL COMPETITION
# ============================================================
st.divider()
st.header("Step 4: AutoML Model Competition")
st.write(
    "Because failure datasets are almost always imbalanced, models are ranked by "
    "**Macro F1-Score** (treats every class equally) rather than raw accuracy "
    "(which a lazy 'always predict Normal' model could win by default)."
)

if st.button("🚀 Run AutoML Training", type="primary"):
    with st.spinner("Training XGBoost, Random Forest, Gradient Boosting, and Extra Trees..."):
        results_df, trained_models, champion_name = train_and_compare(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
    st.session_state["results_df"] = results_df
    st.session_state["trained_models"] = trained_models
    st.session_state["champion_name"] = champion_name
    st.success(f"Training complete -- Champion model: **{champion_name}**")

if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    trained_models = st.session_state["trained_models"]
    champion_name = st.session_state["champion_name"]

    st.dataframe(results_df, use_container_width=True)

    metric_cols = st.columns(len(results_df))
    for i, row in results_df.iterrows():
        delta_label = "🏆 Champion" if row["Model"] == champion_name else None
        metric_cols[i].metric(row["Model"], f"{row['Macro F1-Score']:.3f}", delta_label)

    st.markdown(f"### 🏆 Champion Model: **{champion_name}**")

    with st.expander("Champion model -- detailed classification report"):
        champion_model = trained_models[champion_name]
        champion_preds = champion_model.predict(X_test_scaled)
        st.text(classification_report(
            y_test, champion_preds, labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0,
        ))

        cm = confusion_matrix(y_test, champion_preds, labels=list(range(len(class_names))))
        fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="rocket_r",
            xticklabels=class_names, yticklabels=class_names, ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    # ============================================================
    # SECTION 6 — TOP-K (TOP-2) RISK INFERENCE
    # ===========================================================
    st.divider()
    st.header("Step 5: Top-2 Risk Predictions")
    st.write(
        "Rather than a single hard label, the Champion Model's full probability "
        "distribution is sorted with `numpy.argsort` to surface the **two most "
        "likely** failure modes per row, with confidence scores -- closer to how "
        "a maintenance engineer actually triages ambiguous readings."
    )

    probs = champion_model.predict_proba(X_test_scaled)
    ranked_idx = np.argsort(probs, axis=1)[:, ::-1]  # descending confidence
    top2_idx = ranked_idx[:, :2]

    primary_class = [class_names[i] for i in top2_idx[:, 0]]
    primary_conf = [probs[r, top2_idx[r, 0]] * 100 for r in range(len(probs))]
    secondary_class = [class_names[i] for i in top2_idx[:, 1]]
    secondary_conf = [probs[r, top2_idx[r, 1]] * 100 for r in range(len(probs))]

    risk_table = X_test.reset_index(drop=True).copy()
    risk_table["Actual"] = [class_names[i] for i in y_test.reset_index(drop=True)]
    risk_table["Primary Risk (Confidence %)"] = [
        f"{c} ({p:.1f}%)" for c, p in zip(primary_class, primary_conf)
    ]
    risk_table["Secondary Risk (Confidence %)"] = [
        f"{c} ({p:.1f}%)" for c, p in zip(secondary_class, secondary_conf)
    ]

    st.dataframe(risk_table, use_container_width=True)

    csv_out = risk_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Top-2 Risk Predictions as CSV",
        data=csv_out, file_name="top2_risk_predictions.csv", mime="text/csv",
    )
else:
    st.info("Click **Run AutoML Training** above to start the model competition.")
    # ==========================================
# MAINTENANCE ASSISTANT CHATBOT (GEMINI API)
import google.generativeai as genai

st.markdown("---")
st.subheader("🤖 Maintenance Assistant")

# 1. Pull the API key securely from Streamlit Cloud
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Set the instructions for your bot
    system_prompt = """
    You are the AI Assistant for a Predictive Maintenance and Warranty Claim Dashboard. 
    Answer ONLY questions related to predictive maintenance, machine learning, or this dashboard.
    Keep your responses short, professional, and use bullet points.
    """
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=system_prompt
    )

    # 3. Setup Chat History so the bot remembers the conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. The Chat Input Box
    if prompt := st.chat_input("E.g., What does a high TWF risk mean?"):
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and show bot response
        with st.chat_message("assistant"):
            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"🚨 API Error: {e}")

else:
    # If the app can't find the key, it shows this warning instead of crashing
    st.warning("⚠️ Chatbot is offline. API Key is missing from Streamlit Secrets.")
