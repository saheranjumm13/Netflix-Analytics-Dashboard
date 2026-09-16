import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

# -------------------------
# Load Dataset
# -------------------------

@st.cache
def load_data():
    df = pd.read_csv("Netflix_Cleaned.csv")
    return df


# -------------------------
# Load ML Model
# -------------------------

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("netflix_model.pkl")
    return model

# -------------------------
# KPI Calculations
# -------------------------

def get_kpis(df):

    total_titles = len(df)

    movies = len(df[df["Type"] == "Movie"])

    tvshows = len(df[df["Type"] == "TV Show"])

    countries = df["Country"].nunique()

    ratings = df["Rating"].nunique()

    years = df["Release_Year"].nunique()

    return (
        total_titles,
        movies,
        tvshows,
        countries,
        ratings,
        years
    )


# -------------------------
# Filter Dataset
# -------------------------

def filter_data(df, country, rating, year, content_type):

    filtered = df.copy()

    if country != "All":
        filtered = filtered[filtered["Country"] == country]

    if rating != "All":
        filtered = filtered[filtered["Rating"] == rating]

    if year != "All":
        filtered = filtered[filtered["Release_Year"] == year]

    if content_type != "All":
        filtered = filtered[filtered["Type"] == content_type]

    return filtered


# -------------------------
# Search Titles
# -------------------------

def search_movie(df, keyword):

    if keyword == "":
        return df

    return df[
        df["Title"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]


# -------------------------
# Pie Chart
# -------------------------

def plot_type_chart(df):

    fig = px.pie(
        df,
        names="Type",
        hole=0.55,
        title="Movies vs TV Shows"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -------------------------
# Top Countries
# -------------------------

def plot_country_chart(df):

    country = (
        df["Country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    country.columns = ["Country", "Count"]

    fig = px.bar(
        country,
        x="Country",
        y="Count",
        title="Top Countries"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -------------------------
# Rating Chart
# -------------------------

def plot_rating_chart(df):

    rating = (
        df["Rating"]
        .value_counts()
        .reset_index()
    )

    rating.columns = ["Rating", "Count"]

    fig = px.bar(
        rating,
        x="Rating",
        y="Count",
        color="Count",
        title="Ratings Distribution"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -------------------------
# Release Year Chart
# -------------------------

def plot_year_chart(df):

    year = (
        df["Release_Year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    year.columns = ["Year", "Count"]

    fig = px.line(
        year,
        x="Year",
        y="Count",
        markers=True,
        title="Release Year Trend"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -------------------------
# World Map
# -------------------------

def plot_world_map(df):

    country = (
        df["Country"]
        .value_counts()
        .reset_index()
    )

    country.columns = ["Country", "Count"]

    fig = px.choropleth(
        country,
        locations="Country",
        locationmode="country names",
        color="Count",
        hover_name="Country",
        title="Netflix Content Around the World"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -------------------------
# Feature Importance
# -------------------------

def feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": model.feature_importances_

        })

        fig = px.bar(

            importance,

            x="Feature",

            y="Importance",

            color="Importance",

            title="Feature Importance"

        )

        fig.update_layout(template="plotly_dark")

        return fig

    return None