import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit.components.v1 import html


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="On-Demand Service Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Spatial-Temporal Demand & Customer Sentiment Analysis")
st.caption("Yelp Open Dataset — Data Mining Project")


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    cluster = pd.read_csv(
        "data/processed/dbscan_cluster_summary.csv"
    )

    # Updated 5-model forecasting comparison
    forecasting = pd.read_csv(
        "data/processed/forecasting_model_comparison_5models.csv"
    )

    sentiment = pd.read_csv(
        "data/processed/sentiment_results.csv"
    )

    aspect = pd.read_csv(
        "data/processed/aspect_sentiment_results.csv"
    )

    return cluster, forecasting, sentiment, aspect


cluster, forecasting, sentiment, aspect = load_data()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Spatial Hotspots",
        "Demand Forecasting",
        "Customer Sentiment",
        "Aspect Analysis"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "DBSCAN Clusters",
        f"{len(cluster):,}"
    )

    col2.metric(
        "Businesses Clustered",
        f"{cluster['business_count'].sum():,}"
    )

    col3.metric(
        "Reviews Analyzed",
        f"{len(sentiment):,}"
    )

    col4.metric(
        "Forecasting Models",
        f"{len(forecasting):,}"
    )

    st.markdown("---")

    st.subheader("Project Workflow")

    st.markdown("""
    **Yelp Dataset → Data Preprocessing → EDA
    → Temporal Activity Analysis → Spatial Clustering
    → Demand Forecasting → Sentiment Analysis
    → Aspect Analysis → Interactive Dashboard**
    """)

    st.info(
        "Review activity is used as a proxy for service demand "
        "because the Yelp Open Dataset does not directly provide "
        "service requests, bookings, or transaction-level demand."
    )

    st.subheader("Forecasting Models")

    st.write(
        "The project evaluates five forecasting models using "
        "the same time-based train-validation-test framework:"
    )

    model_list = forecasting["model"].tolist()

    for i, model_name in enumerate(model_list, start=1):
        st.write(f"{i}. {model_name}")


# ==================================================
# SPATIAL HOTSPOTS
# ==================================================

elif page == "Spatial Hotspots":

    st.header("📍 Spatial Hotspot Analysis")

    st.write(
        "DBSCAN was applied to business geographic coordinates "
        "using the Haversine distance."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Clusters",
        f"{len(cluster):,}"
    )

    col2.metric(
        "Clustered Businesses",
        f"{cluster['business_count'].sum():,}"
    )

    col3.metric(
        "Largest Cluster",
        f"{cluster['business_count'].max():,}"
    )

    st.subheader("Top Spatial Clusters")

    top_clusters = cluster.nlargest(
        20,
        "business_count"
    )

    st.dataframe(
        top_clusters[
            [
                "cluster_id",
                "business_count",
                "total_review_count",
                "avg_review_count",
                "avg_stars"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Spatial Hotspot Map")

    m = folium.Map(
        location=[
            cluster["centroid_latitude"].mean(),
            cluster["centroid_longitude"].mean()
        ],
        zoom_start=4
    )

    top_map = cluster.nlargest(
        100,
        "business_count"
    )

    for _, row in top_map.iterrows():

        radius = max(
            4,
            min(25, row["business_count"] / 500)
        )

        popup = f"""
        <b>Cluster:</b> {row['cluster_id']}<br>
        <b>Businesses:</b> {row['business_count']:,}<br>
        <b>Total Reviews:</b> {row['total_review_count']:,}<br>
        <b>Avg Reviews/Business:</b>
        {row['avg_review_count']:.2f}<br>
        <b>Avg Stars:</b> {row['avg_stars']:.2f}
        """

        folium.CircleMarker(
            location=[
                row["centroid_latitude"],
                row["centroid_longitude"]
            ],
            radius=radius,
            popup=popup,
            fill=True
        ).add_to(m)

    html(
        m._repr_html_(),
        height=600
    )


# ==================================================
# DEMAND FORECASTING
# ==================================================

elif page == "Demand Forecasting":

    st.header("📈 Demand Forecasting")

    st.write(
        "Five regression models were evaluated using the same "
        "chronological train-validation-test split."
    )

    st.info(
        "Training: 2005–2020 | Validation: 2021 | Test: 2022. "
        "A 1,000,000-row reproducible training sample was used."
    )

    # --------------------------------------------------
    # Model Comparison Table
    # --------------------------------------------------

    st.subheader("Five-Model Comparison")

    display_forecasting = forecasting.copy()

    display_forecasting.columns = [
        "Model",
        "Validation MAE",
        "Validation RMSE",
        "Validation R²",
        "Test MAE",
        "Test RMSE",
        "Test R²"
    ]

    st.dataframe(
        display_forecasting,
        use_container_width=True
    )

    # --------------------------------------------------
    # Validation Metrics
    # --------------------------------------------------

    st.subheader("Validation Performance")

    validation_plot = forecasting[
        [
            "model",
            "validation_mae",
            "validation_rmse"
        ]
    ].copy()

    validation_plot = validation_plot.melt(
        id_vars="model",
        var_name="Metric",
        value_name="Value"
    )

    validation_plot["Metric"] = validation_plot["Metric"].replace({
        "validation_mae": "MAE",
        "validation_rmse": "RMSE"
    })

    fig = px.bar(
        validation_plot,
        x="model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Validation MAE and RMSE"
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Error"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # Validation R²
    # --------------------------------------------------

    st.subheader("Validation R²")

    fig_r2 = px.bar(
        forecasting,
        x="model",
        y="validation_r2",
        title="Validation R² by Model",
        text_auto=".4f"
    )

    fig_r2.update_layout(
        xaxis_title="Model",
        yaxis_title="R²"
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    # --------------------------------------------------
    # Test Performance
    # --------------------------------------------------

    st.subheader("Test Performance")

    test_plot = forecasting[
        [
            "model",
            "test_mae",
            "test_rmse"
        ]
    ].copy()

    test_plot = test_plot.melt(
        id_vars="model",
        var_name="Metric",
        value_name="Value"
    )

    test_plot["Metric"] = test_plot["Metric"].replace({
        "test_mae": "MAE",
        "test_rmse": "RMSE"
    })

    fig2 = px.bar(
        test_plot,
        x="model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Test MAE and RMSE"
    )

    fig2.update_layout(
        xaxis_title="Model",
        yaxis_title="Error"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # --------------------------------------------------
    # Test R²
    # --------------------------------------------------

    st.subheader("Test R²")

    fig_test_r2 = px.bar(
        forecasting,
        x="model",
        y="test_r2",
        title="Test R² by Model",
        text_auto=".4f"
    )

    fig_test_r2.update_layout(
        xaxis_title="Model",
        yaxis_title="R²"
    )

    st.plotly_chart(
        fig_test_r2,
        use_container_width=True
    )

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    st.subheader("Interpretation")

    st.write(
        "The five models show different behavior between the "
        "validation and held-out 2022 test periods."
    )

    st.write(
        "HistGradientBoosting achieved a validation MAE of "
        "0.6296, validation RMSE of 1.0715, and validation R² "
        "of 0.5804."
    )

    st.write(
        "Linear Regression produced a positive test R² of "
        "0.1235, while Decision Tree, Random Forest, XGBoost, "
        "and HistGradientBoosting produced negative test R² values."
    )

    st.warning(
        "The negative test R² values for several models indicate "
        "weak generalization to the held-out 2022 period. "
        "Therefore, the forecasting models should not be described "
        "as highly accurate."
    )

    st.caption(
        "Review activity is used as a proxy for service demand "
        "because the Yelp Open Dataset does not contain direct "
        "service requests, bookings, or transaction-level demand."
    )


# ==================================================
# CUSTOMER SENTIMENT
# ==================================================

elif page == "Customer Sentiment":

    st.header("💬 Customer Sentiment Analysis")

    sentiment_counts = (
        sentiment["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="VADER Sentiment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive",
        f"{(sentiment['sentiment'] == 'Positive').sum():,}"
    )

    col2.metric(
        "Negative",
        f"{(sentiment['sentiment'] == 'Negative').sum():,}"
    )

    col3.metric(
        "Neutral",
        f"{(sentiment['sentiment'] == 'Neutral').sum():,}"
    )

    st.subheader("VADER Evaluation")

    evaluation = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1"
        ],
        "Score": [
            0.7777,
            0.7269,
            0.7777,
            0.7318
        ]
    })

    st.dataframe(
        evaluation,
        use_container_width=True
    )

    st.caption(
        "Reference labels were derived from Yelp star ratings "
        "(1–2 negative, 3 neutral, 4–5 positive) and therefore "
        "represent weak/proxy labels rather than human annotations."
    )

    st.warning(
        "VADER showed substantial limitations for the Neutral class. "
        "Therefore, the overall accuracy should not be interpreted "
        "as equally reliable performance across all three sentiment classes."
    )


# ==================================================
# ASPECT ANALYSIS
# ==================================================

elif page == "Aspect Analysis":

    st.header("🔍 Aspect-Based Sentiment Analysis")

    table = pd.crosstab(
        aspect["aspect"],
        aspect["sentiment"]
    )

    table = table.reindex(
        columns=[
            "Positive",
            "Neutral",
            "Negative"
        ],
        fill_value=0
    )

    percentage = (
        table.div(
            table.sum(axis=1),
            axis=0
        ) * 100
    ).round(2)

    st.subheader("Aspect Sentiment Distribution")

    st.dataframe(
        percentage,
        use_container_width=True
    )

    plot_data = percentage.reset_index().melt(
        id_vars="aspect",
        var_name="Sentiment",
        value_name="Percentage"
    )

    fig = px.bar(
        plot_data,
        x="aspect",
        y="Percentage",
        color="Sentiment",
        barmode="stack",
        title="Aspect-Based Sentiment Distribution"
    )

    fig.update_yaxes(
        range=[0, 100]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Aspect Findings")

    st.write(
        "Punctuality had the highest proportion of negative "
        "sentiment among the three extracted aspects, at 17.75%."
    )

    st.info(
        "Aspect extraction uses rule-based keyword matching "
        "for Price, Service Quality, and Punctuality. "
        "This approach may miss implicit aspect mentions."
    )