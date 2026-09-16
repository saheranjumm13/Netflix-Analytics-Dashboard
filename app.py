import streamlit as st
import pandas as pd
import plotly.express as px

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide"
)

from PIL import Image
from utils import *

# Load dataset and model
df = load_data()
model = load_model()

# ------------------------
# Load CSS
# ------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ------------------------
# Sidebar
# ------------------------

st.sidebar.image(
    "images/netflix_logo.png",
    width=180
)

st.sidebar.title("Netflix Analytics")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dashboard",

        "📁 Dataset",

        "🤖 Prediction",

        "📈 Model Performance",

        "⚙ Settings"

    ]

)

        

#------------------------
#KPI CARDS
#------------------------

total, movies, tvshows, countries, ratings, years = get_kpis(df)


# ------------------------
# Header
# ------------------------

col1,col2=st.columns([1,5])

with col1:

    st.image(
        "images/netflix_logo.png",
        width=120
    )

with col2:

    st.title("Netflix Analytics Dashboard")

    st.write("Interactive Machine Learning Dashboard")

st.markdown("---")

# ------------------------
# Home Page
# ------------------------

if page=="🏠 Home":

    st.image(
        "images/netflix_logo.png",
        width=250
    )

    st.markdown("## Welcome")

    st.write("""
This project analyzes the Netflix dataset using Machine Learning and Interactive Visualizations.

### Features

- 📊 Dashboard
- 🤖 Machine Learning
- 📈 Charts
- 🌍 Maps
- 🔎 Search
- 📥 Download Data

""")

# ------------------------
# Dashboard Placeholder
# ------------------------

    
elif page == "📊 Dashboard":

    st.header("📊 Netflix Dashboard")

    # KPI Cards
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Titles", total)
    c2.metric("Movies", movies)
    c3.metric("TV Shows", tvshows)

    c4, c5, c6 = st.columns(3)

    c4.metric("Countries", countries)
    c5.metric("Ratings", ratings)
    c6.metric("Release Years", years)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_type_chart(df), use_container_width=True)

    with col2:
        st.plotly_chart(plot_country_chart(df), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(plot_rating_chart(df), use_container_width=True)

    with col4:
        st.plotly_chart(plot_year_chart(df), use_container_width=True)

    st.plotly_chart(plot_world_map(df), use_container_width=True)


# ------------------------
#Dataset-------------
#--------------------

elif page == "📁 Dataset":

    st.header("📁 Netflix Dataset")

    st.dataframe(df)

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistics")

    st.write(df.describe())

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "Netflix_Cleaned.csv",
        "text/csv"
    )

# ------------------------
#Prediction-------------
#------------------------

elif page == "🤖 Prediction":

    st.header("🤖 Netflix Type Prediction")

    country = st.number_input(
        "Country (Encoded)",
        min_value=0,
        step=1
    )

    year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2035,
        value=2021
    )

    rating = st.number_input(
        "Rating (Encoded)",
        min_value=0,
        step=1
    )

    if st.button("Predict"):

        try:

            prediction = model.predict([[country, year, rating]])

            if prediction[0] == 1:
                st.success("📺 Predicted Type : TV Show")
            else:
                st.success("🎬 Predicted Type : Movie")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
            
# ------------------------
#Model Performance--------
#----------------------------
elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    models = {
        "Random Forest": 0.7429,
        "Decision Tree": 0.7349,
        "KNN": 0.7202,
        "Logistic Regression": 0.6907,
        "Naive Bayes": 0.6419
    }

    result = pd.DataFrame(
        models.items(),
        columns=["Model", "Accuracy"]
    )

    st.dataframe(result)

    fig = px.bar(
        result,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
# ------------------------
#Setting page--------------
#---------------------------

elif page == "⚙ Settings":

    st.header("⚙ Settings")

    st.write("Netflix Dashboard Version 1.0")

    st.write("Created using Streamlit + Plotly + Scikit-Learn")
    

    
    
# Footer
# ------------------------

st.markdown("---")

st.markdown(
"""
<div class="footer">

Made with ❤️ using Streamlit | Netflix Analytics Dashboard

</div>
""",
unsafe_allow_html=True
)