from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sentiment_analysis import analyze_sentiment, train_sentiment_models

from recommendation import (
    SENTIMENT_VALUES,
    get_recommendation,
    get_personalized_recommendation,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ola EV Insight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

BASE_DIRECTORY = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIRECTORY
    / "Ola_Scooter_Customer_Experience_Dataset_60000.xlsx"
)
ML_TRAINING_FILE = BASE_DIRECTORY / "Ola_60000_Labelled.xlsx"

SHEET_NAME = "All_Reviews_60000"

MODELS = [
    "Ola S1 X",
    "Ola S1 X+",
    "Ola S1 Pro",
    "Ola S1 Pro+",
]

EXPERIENCE_AREAS = [
    "Battery & Range",
    "Charging Experience",
    "Handling",
    "Performance",
    "Ride Comfort",
    "Service Experience",
    "Software & Display",
    "Value for Money",
]

REQUIRED_COLUMNS = [
    "Review_ID",
    "Model",
    "Review_Date",
    "City",
    "State",
    "Rating_1_to_5",
    "Experience_Area",
    "Customer_Opinion_Experience",
    "Daily_Usage",
    "Ownership_Duration",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --ink: #10213d;
        --muted: #475569;
        --primary: #3949ab;
        --primary-dark: #283593;
        --teal: #00897b;
        --surface: #ffffff;
        --border: #e2e8f0;
        --canvas: #f4f6fb;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1440px;
        padding: 30px 48px 56px;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 246, 251, 0.88);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: 16px;
    }

    h1, h2, h3, h4 {
        color: var(--ink) !important;
        letter-spacing: -0.025em;
    }

    h2 {
        margin-top: 1.8rem !important;
        font-size: 1.45rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 1.08rem !important;
        font-weight: 700 !important;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        color: #1e293b;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: 0.92rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.5rem;
        padding: 30px 34px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 22px;
        background: linear-gradient(118deg, #283593 0%, #3949ab 54%, #00897b 125%);
        box-shadow: 0 14px 34px rgba(40, 53, 147, 0.18);
    }

    .hero::after {
        position: absolute;
        top: -115px;
        right: -50px;
        width: 310px;
        height: 310px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 50%;
        box-shadow: 0 0 0 36px rgba(255, 255, 255, 0.035),
                    0 0 0 74px rgba(255, 255, 255, 0.025);
        content: "";
        pointer-events: none;
    }

    .hero-eyebrow {
        margin-bottom: 10px;
        color: #c5cae9;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        letter-spacing: -0.045em;
        line-height: 1.1;
    }

    .hero-subtitle {
        margin-top: 10px;
        color: #e8eaf6;
        font-size: 1rem;
        line-height: 1.55;
    }

    .hero-mark {
        display: inline-flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.15);
        vertical-align: 4px;
    }

    div[data-testid="stMetric"] {
        min-height: 122px;
        padding: 20px 22px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 5px 16px rgba(30, 41, 59, 0.045);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(30, 41, 59, 0.09);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.86rem;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-weight: 800;
    }

    div[data-baseweb="select"] > div {
        border-color: #cbd5e1 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    div[data-baseweb="select"] span {
        color: #1e293b !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #10213d !important;
        font-weight: 650 !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        color: #10213d !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] span {
        color: #10213d !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
        color: #10213d !important;
        -webkit-text-fill-color: #10213d !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] input::placeholder {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1;
    }

    div[data-baseweb="popover"] {
        background: #ffffff !important;
    }

    div[data-baseweb="popover"] [role="option"] {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] [role="option"] * {
        color: #1e293b !important;
    }

    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
        color: #10213d !important;
        background-color: #e0f2f1 !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        border-color: #cbd5e1 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 44px;
        padding: 0 20px;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        font-size: 14px;
        color: #ffffff;
        background: linear-gradient(135deg, #00897b, #00695c);
        box-shadow: 0 5px 14px rgba(0, 137, 123, 0.3);
        transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 8px 18px rgba(0, 105, 92, 0.34);
    }

    .stButton > button:focus,
    div[data-testid="stFormSubmitButton"] > button:focus {
        box-shadow: 0 0 0 3px rgba(0, 137, 123, 0.26);
    }

    .stDownloadButton > button {
        min-height: 44px;
        border-radius: 10px;
        padding: 0 18px;
        font-size: 14px;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: var(--ink);
    }

    div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 3px 12px rgba(30, 41, 59, 0.035);
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #ffffff;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stPlotlyChart"] {
        padding: 10px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(30, 41, 59, 0.04);
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 720px) {
        .block-container {
            padding: 20px 18px 36px;
        }

        .hero {
            padding: 24px 22px;
            border-radius: 18px;
        }

        .hero::after {
            right: -190px;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Customer intelligence dashboard</div>
        <h1 class="hero-title">
            <span class="hero-mark">⚡</span>Ola EV Insight
        </h1>
        <div class="hero-subtitle">
            Understand the voice of Ola Electric customers with review sentiment,
            product comparisons, and personalized recommendations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# CHECK DATA FILE
# ============================================================

if not DATA_FILE.exists():
    st.error("❌ Dataset file not found.")

    st.info(
        f"""
Make sure these files are in the same folder as `app.py`:

- `app.py`
- `sentiment.py`
- `recommendation.py`
- `Ola_Scooter_Customer_Experience_Dataset_60000.xlsx`

Expected dataset path:

`{DATA_FILE}`
"""
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(file_path):
    data = pd.read_excel(
        file_path,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
    )

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + ", ".join(missing_columns)
        )

    text_columns = [
        "Model",
        "Experience_Area",
        "Customer_Opinion_Experience",
        "City",
        "State",
        "Daily_Usage",
        "Ownership_Duration",
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data["Rating_1_to_5"] = pd.to_numeric(
        data["Rating_1_to_5"],
        errors="coerce",
    )

    data["Rating_1_to_5"] = (
        data["Rating_1_to_5"]
        .clip(1, 5)
    )

    data = data.dropna(
        subset=[
            "Model",
            "Rating_1_to_5",
        ]
    ).copy()

    data = data[
        data["Model"].isin(MODELS)
    ].copy()

    data.reset_index(drop=True, inplace=True)

    return data


try:
    df = load_data(str(DATA_FILE))

except ValueError as error:
    st.error(
        f"❌ Dataset structure error:\n\n{error}"
    )
    st.stop()

except Exception as error:
    st.error(
        f"❌ Dataset loading error:\n\n{error}"
    )
    st.stop()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

@st.cache_data
def perform_sentiment_analysis(texts):
    return analyze_sentiment(texts)


@st.cache_resource
def load_sentiment_models(training_file, source_modified_time):
    training_data = pd.read_excel(training_file)
    return train_sentiment_models(training_data)


if not {
    "Sentiment_Score",
    "Sentiment",
}.issubset(df.columns):

    scores, labels = perform_sentiment_analysis(
        tuple(
            df["Customer_Opinion_Experience"]
            .tolist()
        )
    )

    df["Sentiment_Score"] = scores
    df["Sentiment"] = labels


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(data):
    if data.empty:
        return {
            "reviews": 0,
            "rating": 0.0,
            "sentiment": 0.0,
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0,
            "score": 0.0,
            "recommendation": "Neutral",
        }

    average_rating = data["Rating_1_to_5"].mean()
    average_sentiment = data["Sentiment_Score"].mean()

    positive_percentage = (
        (data["Sentiment"] == "Positive").mean() * 100
    )

    neutral_percentage = (
        (data["Sentiment"] == "Neutral").mean() * 100
    )

    negative_percentage = (
        (data["Sentiment"] == "Negative").mean() * 100
    )

    # recommendation.py is the single source of truth.
    recommendation_label = get_recommendation(
        average_sentiment
    )

    return {
        "reviews": len(data),
        "rating": average_rating,
        "sentiment": average_sentiment,
        "positive": positive_percentage,
        "neutral": neutral_percentage,
        "negative": negative_percentage,
        "score": average_sentiment,
        "recommendation": recommendation_label,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "selected_areas" not in st.session_state:
    st.session_state.selected_areas = []

if "comparison_requested" not in st.session_state:
    st.session_state.comparison_requested = False

if "review_search_query" not in st.session_state:
    st.session_state.review_search_query = ""


# ============================================================
# 1. PRODUCT ANALYSIS
# ============================================================

st.header("1. Product Analysis")

st.caption(
    "Select a scooter model and experience areas to analyze customer feedback."
)

product_col1, product_col2 = st.columns(
    [1, 2],
    gap="large",
)

with product_col1:
    selected_model = st.selectbox(
        "Select Scooter Model",
        MODELS,
        index=None,
        placeholder="Choose an Ola model",
        key="product_model_empty",
    )

with product_col2:
    selected_areas = st.multiselect(
        "Select Experience Areas",
        EXPERIENCE_AREAS,
        default=[],
        key="experience_filter_empty",
    )

analyze_clicked = st.button(
    "Analyze Reviews",
    type="primary",
    use_container_width=True,
    key="analyze_product_button",
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if analyze_clicked:
    if not selected_model:
        st.warning("Please select an Ola model.")

    elif not selected_areas:
        st.warning(
            "Please select at least one experience area."
        )

    else:
        filtered_data = df[
            (df["Model"] == selected_model)
            &
            (
                df["Experience_Area"]
                .isin(selected_areas)
            )
        ].copy()

        st.session_state.analysis_df = filtered_data
        st.session_state.selected_model = selected_model
        st.session_state.selected_areas = selected_areas
        st.session_state.comparison_requested = False


# ============================================================
# CURRENT ANALYSIS
# ============================================================

analysis_df = st.session_state.analysis_df
current_model = st.session_state.selected_model
current_areas = st.session_state.selected_areas


# ============================================================
# RESULTS
# ============================================================

if analysis_df is not None:

    if analysis_df.empty:
        st.error(
            "No reviews found for the selected model and experience areas."
        )
        st.stop()

    metrics = calculate_metrics(analysis_df)


    # ========================================================
    # 2. SENTIMENT ANALYSIS
    # ========================================================

    st.header("2. Sentiment Analysis")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Average Rating",
            f"{metrics['rating']:.2f} / 5",
        )

    with metric2:
        st.metric(
            "Average Sentiment",
            f"{metrics['sentiment']:.3f}",
        )

    with metric3:
        st.metric(
            "Positive Reviews",
            f"{metrics['positive']:.1f}%",
        )

    st.write("")

    chart_col, summary_col = st.columns(
        [1.2, 0.8],
        gap="large",
    )

    with chart_col:

        sentiment_fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Positive",
                        "Neutral",
                        "Negative",
                    ],
                    values=[
                        metrics["positive"],
                        metrics["neutral"],
                        metrics["negative"],
                    ],
                    hole=0.60,
                    textinfo="percent",
                    textfont=dict(
                        size=14,
                        color="white",
                    ),
                    marker=dict(
                        colors=[
                            "#00897b",
                            "#90a4ae",
                            "#e53935",
                        ],
                        line=dict(
                            color="#ffffff",
                            width=3,
                        ),
                    ),
                )
            ]
        )

        sentiment_fig.update_layout(
            title="Sentiment Distribution",
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#10213d", size=14),
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
            legend=dict(
                font=dict(color="#10213d", size=14)
            ),
        )

        st.plotly_chart(
            sentiment_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with summary_col:

        st.subheader("Sentiment Summary")

        st.success(
            f"Positive: {metrics['positive']:.1f}%"
        )

        st.info(
            f"Neutral: {metrics['neutral']:.1f}%"
        )

        st.error(
            f"Negative: {metrics['negative']:.1f}%"
        )

        st.write("")

        st.write(
            f"**Analyzed Model:** {current_model}"
        )

        st.write(
            f"**Reviews Analyzed:** "
            f"{metrics['reviews']:,}"
        )

    st.subheader("VADER and Machine Learning Comparison")
    st.caption(
        "The classifiers are trained on VADER-generated labels; agreement is not independent accuracy."
    )

    if ML_TRAINING_FILE.exists():
        try:
            sentiment_models, _, holdout_reviews = load_sentiment_models(
                str(ML_TRAINING_FILE),
                ML_TRAINING_FILE.stat().st_mtime_ns,
            )
            holdout_texts = set(
                holdout_reviews["Customer_Opinion_Experience"]
            )
            comparison_data = analysis_df[
                analysis_df["Customer_Opinion_Experience"].isin(holdout_texts)
            ]

            if comparison_data.empty:
                st.info(
                    "No held-out review texts match these filters. "
                    "Choose other filters to compare VADER and ML models."
                )
            else:
                holdout_metrics = calculate_metrics(comparison_data)
                unique_reviews = comparison_data[
                    "Customer_Opinion_Experience"
                ].drop_duplicates().tolist()
                comparison_rows = [{
                    "Method": "VADER",
                    "Positive (%)": holdout_metrics["positive"],
                    "Neutral (%)": holdout_metrics["neutral"],
                    "Negative (%)": holdout_metrics["negative"],
                    "Agreement with VADER (%)": None,
                }]

                for model_name, model in sentiment_models.items():
                    predictions = model.predict(unique_reviews)
                    prediction_lookup = dict(zip(unique_reviews, predictions))
                    predicted_sentiments = comparison_data[
                        "Customer_Opinion_Experience"
                    ].map(prediction_lookup)
                    percentages = (
                        predicted_sentiments
                        .value_counts(normalize=True)
                        .mul(100)
                    )
                    comparison_rows.append({
                        "Method": model_name,
                        "Positive (%)": percentages.get("Positive", 0.0),
                        "Neutral (%)": percentages.get("Neutral", 0.0),
                        "Negative (%)": percentages.get("Negative", 0.0),
                        "Agreement with VADER (%)": (
                            predicted_sentiments
                            .eq(comparison_data["Sentiment"])
                            .mean()
                            * 100
                        ),
                    })

                st.caption(
                    f"Comparison uses {len(comparison_data):,} held-out reviews."
                )
                st.dataframe(
                    pd.DataFrame(comparison_rows).round(1),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as error:
            st.warning(f"Machine-learning sentiment comparison unavailable: {error}")
    else:
        st.warning(
            "Machine-learning comparison requires Ola_60000_Labelled.xlsx beside app.py."
        )


    # ========================================================
    # 3. EXPERIENCE AREA ANALYSIS
    # ========================================================

    st.header("3. Experience Area Analysis")

    st.caption(
        "Compare customer ratings and sentiment across the selected experience areas."
    )

    experience_data = (
        analysis_df
        .groupby("Experience_Area")
        .agg(
            Reviews=("Review_ID", "count"),
            Average_Rating=(
                "Rating_1_to_5",
                "mean",
            ),
        )
        .reset_index()
    )

    # Use the sentiment-to-number mapping directly from
    # recommendation.py. No duplicate mapping is defined here.
    sentiment_numeric = (
        analysis_df["Sentiment"]
        .map(SENTIMENT_VALUES)
    )

    area_sentiment = (
        analysis_df.assign(
            _Sentiment_Value=sentiment_numeric
        )
        .groupby("Experience_Area")["_Sentiment_Value"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "_Sentiment_Value":
                    "Recommendation_Score"
            }
        )
    )

    experience_data = (
        experience_data
        .merge(
            area_sentiment,
            on="Experience_Area",
            how="left",
        )
    )

    experience_data["Area_Order"] = (
        experience_data["Experience_Area"]
        .apply(
            lambda x:
            EXPERIENCE_AREAS.index(x)
            if x in EXPERIENCE_AREAS
            else 999
        )
    )

    experience_data = (
        experience_data
        .sort_values("Area_Order")
        .drop(columns=["Area_Order"])
        .reset_index(drop=True)
    )

    experience_fig = go.Figure()

    experience_fig.add_trace(
        go.Bar(
            x=experience_data["Experience_Area"],
            y=experience_data["Recommendation_Score"],
            text=[
                f"{value:.3f}"
                for value in
                experience_data[
                    "Recommendation_Score"
                ]
            ],
            textposition="outside",
            marker_color="#3949ab",
        )
    )

    experience_fig.update_layout(
        title="Experience Area Sentiment Score",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#10213d", size=14),
        yaxis=dict(
            title="Sentiment Score (-1 to +1)",
            range=[-1, 1],
            gridcolor="#e2e8f0",
            zeroline=True,
            zerolinecolor="#94a3b8",
        ),
        xaxis=dict(tickangle=-35),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=100,
        ),
    )

    st.plotly_chart(
        experience_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    display_experience = (
        experience_data
        .rename(
            columns={"Experience_Area":"Experience Area",
                "Average_Rating":"Average Rating",
                "Recommendation_Score":"Sentiment Score",
            }
        )
    )

    display_experience["Average Rating"] = (
        display_experience["Average Rating"].round(2)
    )

    display_experience["Sentiment Score"] = (
        display_experience["Sentiment Score"].round(3)
    )

    st.dataframe(
        display_experience,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # 4. RECOMMENDATION
    # ========================================================

    st.header("4. Recommendation")

    recommendation = get_recommendation(
        metrics["sentiment"]
    )

    recommendation_col1, recommendation_col2 = st.columns(2)

    with recommendation_col1:
        st.metric(
            "Sentiment Recommendation Score",
            f"{metrics['score']:.3f}",
        )

    with recommendation_col2:
        st.metric(
            "Reviews Analyzed",
            f"{metrics['reviews']:,}",
        )

    if recommendation == "Highly Recommended":
        st.success(f"★ {recommendation}")

    elif recommendation == "Recommended":
        st.info(f"★ {recommendation}")

    elif recommendation == "Neutral":
        st.warning(f"• {recommendation}")

    elif recommendation == "Not Recommended":
        st.error(f"⚠ {recommendation}")

    else:
        st.error(f"⚠ {recommendation}")

    st.write("")

    st.write("**Analyzed Experience Areas:**")
    st.write(" • ".join(current_areas))


    # ========================================================
    # 5. PERSONALIZED RECOMMENDATION
    # ========================================================

    st.header("5. Personalized Recommendation")

    personal_col1, personal_col2 = st.columns(2)

    usage_values = sorted(
        [
            str(value)
            for value in df["Daily_Usage"].unique()
            if str(value).strip()
        ]
    )

    ownership_values = sorted(
        [
            str(value)
            for value in df["Ownership_Duration"].unique()
            if str(value).strip()
        ]
    )

    with personal_col1:
        selected_usage = st.selectbox(
            "Daily Usage",
            usage_values,
            key="personal_usage",
        )

    with personal_col2:
        selected_ownership = st.selectbox(
            "Ownership Duration",
            ownership_values,
            key="personal_ownership",
        )

    personal_df = df[
        (
            df["Daily_Usage"].astype(str)
            == selected_usage
        )
        &
        (
            df["Ownership_Duration"].astype(str)
            == selected_ownership
        )
    ].copy()

    best_model, recommendation_df = (
        get_personalized_recommendation(
            personal_df,
            current_areas,
        )
    )

    if best_model is not None:

        personalized_df = (
            recommendation_df
            .rename(
                columns={
                    "Score":
                        "Sentiment Score",
                    "Reviews":
                        "Reviews Analyzed",
                }
            )
        )

        personalized_df["Sentiment Score"] = (
            personalized_df["Sentiment Score"].round(4)
        )

        st.dataframe(
            personalized_df,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            f"Best match for these priorities: "
            f"**{best_model['Model']}** "
            f"({best_model['Recommendation']})"
        )

    else:
        st.warning(
            "No reviews match this usage, ownership duration, "
            "and selected experience areas. Try different filters "
            "or select more areas."
        )


    # ========================================================
    # 6. MODEL COMPARISON
    # ========================================================

    st.header("6. Model Comparison")

    st.caption(
        "Choose two models, then select the review features you want to compare."
    )

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        comparison_model_1 = st.selectbox(
            "First Model",
            MODELS,
            key="comparison_model_1",
        )

    second_model_options = [
        model
        for model in MODELS
        if model != comparison_model_1
    ]

    with comparison_col2:
        comparison_model_2 = st.selectbox(
            "Second Model",
            second_model_options,
            key="comparison_model_2",
        )

    comparison_features = {
        "Review Count": ("Reviews", "reviews"),
        "Average Rating": ("Average Rating", "rating"),
        "Average Sentiment": ("Average Sentiment", "sentiment"),
        "Positive Reviews (%)": ("Positive %", "positive"),
        "Neutral Reviews (%)": ("Neutral %", "neutral"),
        "Negative Reviews (%)": ("Negative %", "negative"),
        "Recommendation Score": ("Recommendation Score", "score"),
    }

    selected_comparison_features = st.multiselect(
        "Features to compare",
        options=list(comparison_features),
        default=["Average Rating", "Average Sentiment", "Positive Reviews (%)"],
        help="Choose the review metrics to show for both models.",
        key="comparison_features",
    )

    compare_clicked = st.button(
        "🔍 Compare Models",
        type="primary",
        use_container_width=True,
    )

    if compare_clicked:
        st.session_state.comparison_requested = True

    comparison_requested = (
        st.session_state.comparison_requested
    )

    if not comparison_requested:
        st.info(
            "Select the features you care about and click Compare Models to see the results."
        )

    comparison_results = []

    for model in [
        comparison_model_1,
        comparison_model_2,
    ]:

        model_data = df[
            df["Model"] == model
        ]

        model_metrics = calculate_metrics(
            model_data
        )

        result = {"Model": model}
        for feature in selected_comparison_features:
            column_name, metric_key = comparison_features[feature]
            result[column_name] = model_metrics[metric_key]
        result["Recommendation"] = model_metrics["recommendation"]
        comparison_results.append(result)

    comparison_df = pd.DataFrame(
        comparison_results
    )

    if comparison_requested and not selected_comparison_features:
        st.warning("Select at least one feature to compare.")

    if comparison_requested and selected_comparison_features:

        selected_columns = [
            comparison_features[feature][0]
            for feature in selected_comparison_features
        ]
        decimal_places = {
            "Average Rating": 2,
            "Average Sentiment": 3,
            "Positive %": 1,
            "Neutral %": 1,
            "Negative %": 1,
            "Recommendation Score": 3,
        }

        st.dataframe(
            comparison_df[["Model", *selected_columns, "Recommendation"]].round(
                {column: places for column, places in decimal_places.items()
                 if column in selected_columns}
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_feature = st.selectbox(
            "Chart feature",
            selected_comparison_features,
            key="comparison_chart_feature",
            help="The chart shows one feature at a time so each metric keeps its own scale.",
        )
        chart_column = comparison_features[chart_feature][0]

        comparison_fig = go.Figure()
        chart_values = comparison_df[chart_column].tolist()
        comparison_fig.add_trace(
            go.Bar(
                x=[comparison_model_1, comparison_model_2],
                y=chart_values,
                text=[f"{value:,.3f}" if chart_column in {
                    "Average Sentiment", "Recommendation Score"
                } else f"{value:,.2f}" if chart_column == "Average Rating"
                    else f"{value:,.1f}%" if chart_column.endswith("%")
                    else f"{value:,.0f}" for value in chart_values],
                textposition="outside",
                marker_color=["#3949ab", "#00897b"],
                hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>",
            )
        )

        comparison_fig.update_layout(
            title=f"{chart_feature} by Model",
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#10213d", size=14),
            yaxis=dict(
                title=chart_feature,
                gridcolor="#e2e8f0",
                zeroline=chart_column in {
                    "Average Sentiment", "Recommendation Score"
                },
                zerolinecolor="#94a3b8",
            ),
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=70,
            ),
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        difference = abs(chart_values[0] - chart_values[1])
        if chart_column.endswith("%"):
            difference_text = f"{difference:,.1f} percentage points"
        elif chart_column == "Reviews":
            difference_text = f"{difference:,.0f} reviews"
        elif chart_column == "Average Rating":
            difference_text = f"{difference:,.2f}"
        else:
            difference_text = f"{difference:,.3f}"
        st.info(f"{chart_feature} difference: **{difference_text}**")


    # ========================================================
    # 7. CUSTOMER REVIEWS
    # ========================================================

    st.header("7. Customer Reviews")

    st.caption(
        "Search reviews from the selected product analysis."
    )

    with st.form("review_search_form"):

        search_col, button_col = st.columns(
            [5, 1],
            gap="small",
        )

        with search_col:
            search_input = st.text_input(
                "Search Customer Reviews",
                placeholder=(
                    "Example: battery, range, charging, "
                    "service, performance..."
                ),
                value=(
                    
                    st.session_state
                    .review_search_query
                ),
            )

        with button_col:
            st.write("")
            search_clicked = (
                st.form_submit_button(
                    "🔍 Search",
                    type="primary",
                    use_container_width=True,
                )
            )

    if search_clicked:
        st.session_state.review_search_query = (
            search_input
        )

    search_query = (
        st.session_state.review_search_query
    )

    review_results = analysis_df.copy()

    if search_query.strip():

        search_terms = [
            term.strip().lower()
            for term in (
                search_query
                .replace(",", " ")
                .split()
            )
            if term.strip()
        ]

        searchable_columns = [
            "Customer_Opinion_Experience",
            "Experience_Area",
            "Model",
            "City",
            "State",
            "Daily_Usage",
            "Ownership_Duration",
            "Sentiment",
        ]

        search_matches = pd.Series(
            False,
            index=review_results.index,
        )

        for term in search_terms:
            for column in searchable_columns:

                if column in review_results.columns:

                    search_matches = (
                        search_matches
                        |
                        review_results[column]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            term,
                            na=False,
                            regex=False,
                        )
                    )

        review_results = (
            review_results[
                search_matches
            ]
        )

    st.write(
        f"Matching Reviews: "
        f"**{len(review_results):,}**"
    )

    if not review_results.empty:

        for _, row in (
            review_results
            .head(10)
            .iterrows()
        ):

            with st.expander(
                f"Review {row['Review_ID']} • "
                f"Rating {row['Rating_1_to_5']}/5 • "
                f"{row['Sentiment']}"
            ):

                st.write(
                    row[
                        "Customer_Opinion_Experience"
                    ]
                )

                st.caption(
                    f"Experience Area: "
                    f"{row['Experience_Area']}"
                )

                st.caption(
                    f"Location: "
                    f"{row['City']}, "
                    f"{row['State']}"
                )

                st.caption(
                    f"Daily Usage: "
                    f"{row['Daily_Usage']}"
                )

                st.caption(
                    f"Ownership Duration: "
                    f"{row['Ownership_Duration']}"
                )

                st.caption(
                    f"VADER Sentiment Score: "
                    f"{row['Sentiment_Score']:.3f}"
                )

    else:
        st.info(
            "No reviews match your search."
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Select a scooter model and experience areas, "
        "then click **Analyze**."
    )

    st.write("")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ Ola EV Insight • Social Sentiment & Personalized Recommendation System"
)
