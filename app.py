import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("Food_and_Nutrition__.csv")

food_df = load_data()

st.sidebar.header("Genetic Algorithm Parameters")

# Define GA Parameter for sidebar
TARGET_CALORIES = st.sidebar.slider("Target Calories", 1500, 3000, 2000)
POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 10, 200, 50)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
BUDGET = st.sidebar.slider("Max Budget (RM)", 10, 50, 20)

# Chromosome representation for 4 meals
NUM_MEALS = 4  # breakfast, lunch, dinner, snack

def create_individual():
    return random.sample(list(food_df.index), NUM_MEALS)


