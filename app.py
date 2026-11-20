import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler


st.set_page_config(page_title="Student Risk Intelligence Lab", layout="wide")


def create_sample_data():
    return pd.DataFrame(
        [
            ["yes", "no", "no", "no", "yes", "yes", "no", "no", 4, 3, 4, 1, 1, 3, 6, 5, 6, 6],
            ["yes", "yes", "no", "no", "no", "yes", "yes", "no", 5, 3, 3, 1, 1, 3, 4, 5, 5, 6],
            ["no", "yes", "yes", "no", "yes", "yes", "yes", "no", 4, 3, 2, 2, 3, 3, 10, 7, 8, 10],
            ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", 3, 2, 2, 1, 1, 5, 2, 15, 14, 15],
            ["yes", "yes", "no", "yes", "yes", "no", "no", "no", 4, 3, 2, 1, 2, 5, 4, 6, 10, 10],
            ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", 5, 4, 2, 1, 2, 5, 10, 15, 15, 15],
            ["no", "no", "no", "no", "yes", "yes", "yes", "no", 4, 4, 4, 1, 1, 3, 0, 12, 12, 11],
            ["yes", "yes", "no", "no", "yes", "yes", "no", "no", 4, 1, 4, 1, 1, 1, 6, 6, 5, 6],
            ["yes", "yes", "yes", "no", "yes", "yes", "yes", "no", 4, 2, 2, 1, 1, 1, 0, 16, 18, 19],
            ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", 5, 5, 1, 1, 1, 5, 0, 14, 15, 15],
            ["yes", "yes", "yes", "yes", "yes", "yes", "no", "no", 5, 1, 1, 1, 2, 6, 2, 15, 17, 18],
            ["no", "no", "no", "yes", "yes", "yes", "yes", "no", 2, 1, 3, 1, 1, 4, 8, 9, 10, 11],
        ],
        columns=[
            "school_support",
            "family_support",
            "paid_classes",
            "activities",
            "nursery",
            "higher_education",
            "internet",
            "romantic",
            "study_hours",
            "free_time",
            "go_out",
            "health",
            "absences",
            "risk_flag",
            "g1",
            "g2",
            "g3",
            "final_score",
        ],
    )


def get_scaler(name):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    return RobustScaler()


def detect_target_column(df):
    candidates = ["target", "label", "class", "result", "status", "final_score", "pass", "y"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].nunique() <= min(10, len(df)):
            return col
    return df.columns[-1]


def make_target_distribution(df, target_col, max_categories=12):
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > max_categories:
        values = target.dropna()
        bin_count = min(10, values.nunique())
        counts = pd.cut(values, bins=bin_count, duplicates="drop").value_counts(sort=False)
        distribution = pd.DataFrame(
            {
                "Value": [str(interval) for interval in counts.index],
                "Count": counts.to_numpy(),
            }
        )
        missing_count = int(target.isna().sum())
        if missing_count:
            distribution.loc[len(distribution)] = ["(Missing)", missing_count]
        return distribution

    counts = (
        target.astype("string")
        .fillna("(Missing)")
        .value_counts()
    )
    distribution = counts.head(max_categories).rename_axis("Value").reset_index()
    distribution.columns = ["Value", "Count"]
    other_count = int(counts.iloc[max_categories:].sum())
    if other_count:
        distribution.loc[len(distribution)] = ["Other values", other_count]
    return distribution


def build_correlation_chart(df, height=400):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    correlation_matrix = df[numeric_cols].corr()
    correlation_matrix.index.name = "Feature Y"
    correlation_matrix.columns.name = "Feature X"
    correlation = correlation_matrix.stack().rename("Correlation").reset_index()
    return (
        alt.Chart(correlation)
        .mark_rect()
        .encode(
            x=alt.X("Feature X:N", title=None, sort=numeric_cols),
            y=alt.Y("Feature Y:N", title=None, sort=numeric_cols),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
            tooltip=[
                alt.Tooltip("Feature X:N"),
                alt.Tooltip("Feature Y:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(height=min(600, max(280, height)))
    )


def normalize_target(series):
    s = series.copy()
    if s.nunique() > 2:
        s = (s >= s.median()).astype(int)
    return s


def build_preprocessor(X, scaler_name):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", get_scaler(scaler_name)),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_models(df, target_col, scaler_name, leakage_mode, test_size, folds):
    data = df.copy()

    if leakage_mode == "A: Early warning (drop G1 & G2)":
        for col in ["g1", "g2"]:
            if col in data.columns:
                data = data.drop(columns=[col])

    X = data.drop(columns=[target_col])
    y = normalize_target(data[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=min(folds, min(len(np.unique(y_train)), 5)), shuffle=True, random_state=42)

    results = []
    best_score = -1
    best_name = ""

    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scaler_name)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(
            y_test,
            pred,
            average="binary" if len(np.unique(y_test)) == 2 else "weighted",
        )

        cv_score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy"))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1": f1,
            "CV Accuracy": cv_score,
        })

        if acc > best_score:
            best_score = acc
            best_name = name

    return results, best_name, X_train, X_test, y_train, y_test


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #101a2d 0%, #0b1220 30%, #070d18 100%);
        color: #e5edf9;
    }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fbff;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .sidebar-label {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9eb3d2;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.06em;
        color: #edf4ff;
        margin-bottom: 0.4rem;
    }
    .section-subtitle {
        font-size: 1rem;
        color: #a9b9d1;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 12px 30px rgba(2, 6, 23, 0.28);
    }
    .metric-label {
        color: #8ea5c7;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #f8fbff;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .pipeline-step {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 12px;
        padding: 0.8rem;
        min-height: 92px;
    }
    .pipeline-step-title {
        color: #f8fbff;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .pipeline-step-detail {
        color: #a9b9d1;
        font-size: 0.82rem;
    }
    .bar-track {
        width: 100%;
        background: rgba(148, 163, 184, 0.16);
        border-radius: 999px;
        overflow: hidden;
        height: 14px;
        margin: 6px 0 14px;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #8b5cf6);
        border-radius: 999px;
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #dfeaf9;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: rgba(59, 130, 246, 0.16);
        border-color: rgba(96, 165, 250, 0.5);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Model<br>Control</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload student.csv (optional)", type=["csv"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">Pass threshold</div>', unsafe_allow_html=True)
    threshold_value = st.slider("", min_value=0, max_value=100, value=10, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top: 1.2rem;">Leakage scenario</div>', unsafe_allow_html=True)
    leakage_mode = st.radio(
        "",
        ["A: Early warning (drop G1 & G2)", "B: Progressive tracking (keep G1 & G2)"],
        index=0,
        horizontal=False,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label" style="margin-top: 1.2rem;">Feature scaler</div>', unsafe_allow_html=True)
    scaler_name = st.selectbox("", ["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top: 1.2rem;">Test size (holdout)</div>', unsafe_allow_html=True)
    test_size = st.slider("", min_value=0.1, max_value=0.4, value=0.2, step=0.05, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top: 1.2rem;">Cross-validation folds (K)</div>', unsafe_allow_html=True)
    folds = st.slider("", min_value=2, max_value=10, value=5, label_visibility="collapsed")

sample_df = create_sample_data()
if uploaded is not None:
    sample_df = pd.read_csv(uploaded)

st.markdown('<div class="section-header">Student Risk Intelligence Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">A research-oriented learning analytics environment for exploring patterns, preprocessing signals, and comparing predictive models.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "EDA", "Preprocessing", "Modeling", "Evaluation"])

target_col = detect_target_column(sample_df)
if "final_score" in sample_df.columns:
    target_col = "final_score"

with tabs[0]:
    st.markdown("### Dataset overview")
    st.markdown("#### Complete dataset overview pipeline")
    pipeline_steps = [
        ("1. Source", "Uploaded CSV" if uploaded is not None else "Built-in sample dataset"),
        ("2. Structure", f"{len(sample_df):,} rows · {len(sample_df.columns):,} columns"),
        ("3. Data quality", f"{int(sample_df.isna().sum().sum()):,} missing · {int(sample_df.duplicated().sum()):,} duplicate rows"),
        ("4. Target", f"{target_col} · {sample_df[target_col].nunique(dropna=True):,} unique values"),
        ("5. Full data", "All rows and columns available below"),
    ]
    pipeline_cols = st.columns(len(pipeline_steps))
    for col, (title, detail) in zip(pipeline_cols, pipeline_steps):
        with col:
            st.markdown(
                f'<div class="pipeline-step"><div class="pipeline-step-title">{title}</div><div class="pipeline-step-detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    metrics = [
        ("Rows", len(sample_df)),
        ("Columns", len(sample_df.columns)),
        ("Missing values", int(sample_df.isna().sum().sum())),
        ("Target", target_col),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Target distribution")
    target_distribution = make_target_distribution(sample_df, target_col)
    bar_pie_cols = st.columns(2)
    with bar_pie_cols[0]:
        st.caption("Bar chart")
        target_bar = (
            alt.Chart(target_distribution)
            .mark_bar()
            .encode(
                x=alt.X("Value:N", sort=target_distribution["Value"].tolist(), axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("Count:Q", title="Rows"),
                tooltip=["Value:N", "Count:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(target_bar, use_container_width=True)
    with bar_pie_cols[1]:
        st.caption("Pie chart")
        target_pie = (
            alt.Chart(target_distribution)
            .mark_arc()
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Value:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["Value:N", "Count:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(target_pie, use_container_width=True)

    st.markdown("#### Correlation matrix")
    overview_correlation = build_correlation_chart(sample_df, height=440)
    if overview_correlation is not None:
        st.altair_chart(overview_correlation, use_container_width=True)
    else:
        st.info("At least two numeric columns are required to show a correlation matrix.")

    st.markdown("#### Column profile")
    profile = pd.DataFrame(
        {
            "Column": sample_df.columns,
            "Data type": [str(dtype) for dtype in sample_df.dtypes],
            "Non-null": sample_df.notna().sum().to_numpy(),
            "Missing": sample_df.isna().sum().to_numpy(),
            "Missing %": (sample_df.isna().mean().mul(100).round(1)).to_numpy(),
            "Unique values": sample_df.nunique(dropna=True).to_numpy(),
            "Example": [
                sample_df[col].dropna().iloc[0] if sample_df[col].notna().any() else "—"
                for col in sample_df.columns
            ],
        }
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown("#### Complete dataset")
    st.caption(f"Showing all {len(sample_df):,} rows and {len(sample_df.columns):,} columns.")
    st.dataframe(sample_df, use_container_width=True, hide_index=False)

with tabs[1]:
    st.subheader("EDA Studio")
    numeric_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in sample_df.columns if c not in numeric_cols]
    eda_steps = [
        ("1. Data quality", "Missing values and schema"),
        ("2. Univariate", "Feature distributions"),
        ("3. Bivariate", "Feature relationships"),
        ("4. Correlation", "Numeric associations"),
        ("5. Target", "Class or value balance"),
    ]
    eda_step_cols = st.columns(len(eda_steps))
    for col, (title, detail) in zip(eda_step_cols, eda_steps):
        with col:
            st.markdown(
                f'<div class="pipeline-step"><div class="pipeline-step-title">{title}</div><div class="pipeline-step-detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    eda_metric_cols = st.columns(4)
    eda_metrics = [
        ("Observations", len(sample_df)),
        ("Numeric features", len(numeric_cols)),
        ("Categorical features", len(categorical_cols)),
        ("Missing cells", int(sample_df.isna().sum().sum())),
    ]
    for col, (label, value) in zip(eda_metric_cols, eda_metrics):
        with col:
            st.metric(label, value)

    if numeric_cols:
        st.markdown("#### Numeric feature distribution")
        selected_numeric = st.selectbox(
            "Choose a numeric feature",
            numeric_cols,
            key="eda_numeric_feature",
        )
        values = sample_df[selected_numeric].dropna().to_numpy()
        if len(values):
            bin_count = min(20, max(5, int(np.sqrt(len(values)))))
            counts, edges = np.histogram(values, bins=bin_count)
            histogram = pd.DataFrame(
                {
                    "Value range": [
                        f"{edges[i]:.3g} – {edges[i + 1]:.3g}"
                        for i in range(len(counts))
                    ],
                    "Count": counts,
                }
            )
            st.bar_chart(histogram, x="Value range", y="Count")
        else:
            st.info(f"No non-missing values are available for {selected_numeric}.")

    if categorical_cols:
        st.markdown("#### Categorical feature frequency")
        selected_category = st.selectbox(
            "Choose a categorical feature",
            categorical_cols,
            key="eda_categorical_feature",
        )
        category_counts = (
            sample_df[selected_category]
            .astype("string")
            .fillna("(Missing)")
            .value_counts()
            .head(20)
            .rename("Count")
        )
        st.bar_chart(category_counts)

    st.markdown("#### Data quality: missing values")
    missing_profile = (
        sample_df.isna()
        .mean()
        .mul(100)
        .rename("Missing %")
        .reset_index()
    )
    missing_profile.columns = ["Column", "Missing %"]
    if missing_profile["Missing %"].gt(0).any():
        missing_chart = (
            alt.Chart(missing_profile)
            .mark_bar()
            .encode(
                x=alt.X("Column:N", sort="-y", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Missing %:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Column:N", alt.Tooltip("Missing %:Q", format=".1f")],
            )
            .properties(height=280)
        )
        st.altair_chart(missing_chart, use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")

    if len(numeric_cols) >= 2:
        st.markdown("#### Numeric correlation matrix")
        heatmap = build_correlation_chart(
            sample_df,
            height=24 * len(numeric_cols),
        )
        st.altair_chart(heatmap, use_container_width=True)

    scatter_features = [col for col in numeric_cols if col != target_col]
    if len(scatter_features) < 2:
        scatter_features = numeric_cols
    if len(scatter_features) >= 2:
        st.markdown("#### Feature relationship")
        scatter_cols = st.columns(2)
        with scatter_cols[0]:
            x_feature = st.selectbox(
                "X-axis",
                scatter_features,
                key="eda_scatter_x",
            )
        y_options = [col for col in scatter_features if col != x_feature]
        with scatter_cols[1]:
            y_feature = st.selectbox(
                "Y-axis",
                y_options,
                key="eda_scatter_y",
            )

        plot_data = sample_df[[x_feature, y_feature, target_col]].dropna(
            subset=[x_feature, y_feature]
        )
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(5000, random_state=42)
        scatter = (
            alt.Chart(plot_data)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X(field=x_feature, type="quantitative"),
                y=alt.Y(field=y_feature, type="quantitative"),
                color=alt.Color(
                    field=target_col,
                    type="quantitative" if pd.api.types.is_numeric_dtype(sample_df[target_col]) else "nominal",
                ),
                tooltip=[
                    alt.Tooltip(field=x_feature, type="quantitative"),
                    alt.Tooltip(field=y_feature, type="quantitative"),
                    alt.Tooltip(field=target_col, type="nominal"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(scatter, use_container_width=True)

    if categorical_cols:
        st.markdown("#### Categorical summary")
        st.dataframe(sample_df[categorical_cols].astype(str).describe().T, use_container_width=True)

    if numeric_cols:
        st.markdown("#### Numeric summary")
        st.dataframe(sample_df[numeric_cols].describe().T, use_container_width=True)

    if target_col in sample_df.columns:
        st.markdown("#### Target distribution")
        target_distribution = make_target_distribution(sample_df, target_col)
        target_distribution = target_distribution.set_index("Value")["Count"]
        st.bar_chart(target_distribution)

with tabs[2]:
    st.subheader("Preprocessing pipeline")
    st.code(
        """
        # Example cleaning flow
        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ]
        )
        """,
        language="python",
    )

with tabs[3]:
    st.subheader("Model comparison")
    metric_choice = st.selectbox("Metric to compare", ["Accuracy", "F1 score"])

    results, best_name, X_train, X_test, y_train, y_test = evaluate_models(
        sample_df, target_col, scaler_name, leakage_mode, test_size, folds
    )
    comparison_df = pd.DataFrame(results).sort_values(
        "Accuracy" if metric_choice == "Accuracy" else "F1",
        ascending=False,
    )

    st.markdown(f"<h4>Best model: {best_name}</h4>", unsafe_allow_html=True)
    for _, row in comparison_df.iterrows():
        score_col = "Accuracy" if metric_choice == "Accuracy" else "F1"
        width = float(row[score_col]) * 100
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:180px;'>{row['Model']}</div>
                <div style='flex:1;'>
                    <div class='bar-track'>
                        <div class='bar-fill' style='width:{width:.1f}%'></div>
                    </div>
                </div>
                <div style='width:60px; text-align:right;'>{row[score_col]:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[4]:
    st.subheader("Evaluation summary")
    results, best_name, X_train, X_test, y_train, y_test = evaluate_models(
        sample_df, target_col, scaler_name, leakage_mode, test_size, folds
    )

    best_model_name = max(results, key=lambda x: x["Accuracy"])["Model"]
    best_model = build_models()[best_model_name]

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, scaler_name)),
            ("model", best_model),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Validation accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        st.metric("F1 score", f"{f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'):.3f}")
    with col2:
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test)), use_container_width=True)

    st.write("Classification report")
    st.text(classification_report(y_test, y_pred))

st.markdown("<hr style='margin-top: 1.5rem; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
