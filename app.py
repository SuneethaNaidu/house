import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from prediction_engine import HousePriceEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="EstateIQ",

    page_icon="🏠",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

.hero {

    padding: 35px;

    border-radius: 20px;

    background:
    linear-gradient(
        135deg,
        #111827,
        #172554
    );

    margin-bottom: 25px;
}

.hero h1 {

    font-size: 45px;

    margin-bottom: 5px;

}

.hero p {

    color: #94a3b8;

    font-size: 18px;

}

.price-card {

    padding: 30px;

    border-radius: 20px;

    background: #111827;

    border: 1px solid #263244;

    text-align: center;

}

.price {

    font-size: 48px;

    font-weight: 800;

}

.range {

    color: #94a3b8;

    font-size: 18px;

}

.metric-card {

    padding: 20px;

    border-radius: 16px;

    background: #111827;

    border: 1px solid #263244;

}

.small-label {

    color: #94a3b8;

    font-size: 14px;

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_engine():

    return HousePriceEngine(
        "house_price_model.pkl"
    )


try:

    engine = load_engine()

except Exception as e:

    st.error(
        "Model not found. Run train_model.py first."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🏠 EstateIQ</h1>

<p>
AI-powered real estate valuation and
property intelligence engine
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏡 Property Details"
)

st.sidebar.markdown(
    "Enter the property information"
)


# ============================================================
# PROPERTY INPUT
# ============================================================

st.sidebar.subheader(
    "Property"
)

age = st.sidebar.number_input(
    "House Age",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

distance = st.sidebar.number_input(
    "Distance to MRT / Main Transport",
    min_value=0.0,
    value=5.0
)

stores = st.sidebar.number_input(
    "Nearby Stores",
    min_value=0,
    max_value=50,
    value=5
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=24.97,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=121.54,
    format="%.6f"
)

population = st.sidebar.number_input(
    "Nearby Population Density",
    min_value=0.0,
    value=1000.0
)


# ============================================================
# PREDICT
# ============================================================

predict_button = st.sidebar.button(
    "🚀 Calculate AI Valuation",
    use_container_width=True
)


if predict_button:

    input_data = {

        "X2 house age": age,

        "X3 distance to the nearest MRT station":
            distance,

        "X4 number of convenience stores":
            stores,

        "X5 latitude":
            latitude,

        "X6 longitude":
            longitude
    }

    # Add other fields if your dataset contains them
    for feature in engine.features:

        if feature not in input_data:

            input_data[feature] = 0

    result = engine.predict(
        input_data
    )

    prediction = result["prediction"]

    lower = result["lower"]

    upper = result["upper"]


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.markdown(
        "## 🎯 AI Property Valuation"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="small-label">
            Estimated Price
            </div>

            <h2>
            {prediction:.2f}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="small-label">
            Lower Estimate
            </div>

            <h2>
            {lower:.2f}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="small-label">
            Upper Estimate
            </div>

            <h2>
            {upper:.2f}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="small-label">
            Prediction Spread
            </div>

            <h2>
            {(upper-lower):.2f}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRICE RANGE
    # ========================================================

    st.markdown(
        "### 📊 Estimated Price Range"
    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=[
                lower,
                prediction,
                upper
            ],

            y=[
                0,
                0,
                0
            ],

            mode="markers+lines",

            marker={
                "size": 18
            },

            line={
                "width": 5
            }

        )

    )

    fig.update_layout(

        height=180,

        xaxis_title="Price",

        yaxis=dict(
            visible=False
        ),

        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    st.markdown(
        "## 🤖 Model Predictions"
    )

    model_data = pd.DataFrame({

        "Model":
            list(
                result[
                    "model_predictions"
                ].keys()
            ),

        "Prediction":
            list(
                result[
                    "model_predictions"
                ].values()
            )

    })

    fig = px.bar(

        model_data,

        x="Model",

        y="Prediction",

        title="Prediction by Model"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # ========================================================
    # INPUT ANALYSIS
    # ========================================================

    st.markdown(
        "## 🔍 Property Analysis"
    )

    analysis = pd.DataFrame({

        "Feature": [
            "House Age",
            "Transport Distance",
            "Nearby Stores",
            "Latitude",
            "Longitude",
            "Population"
        ],

        "Value": [
            age,
            distance,
            stores,
            latitude,
            longitude,
            population
        ]

    })

    st.dataframe(

        analysis,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # LOCATION
    # ========================================================

    st.markdown(
        "## 📍 Property Location"
    )

    map_df = pd.DataFrame({

        "lat": [latitude],

        "lon": [longitude]

    })

    st.map(
        map_df
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "## 🧠 AI Valuation Summary"
    )

    st.info(

        f"""
        The trained ensemble estimates the property
        price at **{prediction:.2f} price units**.

        The estimated prediction interval is
        **{lower:.2f} – {upper:.2f}**.

        The interval represents model uncertainty
        based on the historical validation residuals.
        """

    )

else:

    st.info(
        "👈 Enter the property details and click "
        "**Calculate AI Valuation**."
    )
