import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("Food_and_Nutrition_with_Price.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

food_df = load_data()


# Column Auto-Mapping (IMPORTANT)

def find_column(possible_names):
    for col in food_df.columns:
        for name in possible_names:
            if name in col:
                return col
    return None

CAL_COL = find_column(["cal", "energy"])
PROTEIN_COL = find_column(["protein"])
PRICE_COL = find_column(["price", "cost", "rm"])


# Sidebar Parameters

st.sidebar.header("Genetic Algorithm Parameters")

TARGET_CALORIES = st.sidebar.slider("Target Calories", 1500, 3000, 2000)
BUDGET = st.sidebar.slider("Max Budget (RM)", 5, 50, 20)
POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 10, 200, 50)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)


# GA Settings

NUM_MEALS = 4

def create_individual():
    return random.sample(list(food_df.index), NUM_MEALS)


# Fitness Function (PRICE INCLUDED)

def fitness(individual):
    meals = food_df.loc[individual]

    total_calories = meals[CAL_COL].sum() if CAL_COL else 0
    total_protein = meals[PROTEIN_COL].sum() if PROTEIN_COL else 0
    total_price = meals[PRICE_COL].sum() if PRICE_COL else 0

    calorie_penalty = abs(TARGET_CALORIES - total_calories)
    protein_penalty = max(0, 50 - total_protein) * 5
    price_penalty = max(0, total_price - BUDGET) * 10

    fitness_score = (
        10000
        - calorie_penalty * 5
        - protein_penalty
        - price_penalty
    )

    return fitness_score


# Selection

def selection(population):
    candidates = random.sample(population, 3)
    candidates.sort(key=lambda x: fitness(x), reverse=True)
    return candidates[0]


# Crossover

def crossover(parent1, parent2):
    point = random.randint(1, NUM_MEALS - 1)
    child = parent1[:point] + parent2[point:]
    return list(dict.fromkeys(child))[:NUM_MEALS]


# Mutation

def mutation(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_MEALS - 1)
        individual[index] = random.choice(food_df.index)
    return individual


# Genetic Algorithm

def genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    fitness_history = []

    for _ in range(GENERATIONS):
        new_population = []

        for _ in range(POP_SIZE):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population
        best = max(population, key=lambda x: fitness(x))
        fitness_history.append(fitness(best))

    return best, fitness_history


# Streamlit UI

st.title("🍽️ Diet Meal Planning Optimisation using Genetic Algorithm")

st.write("""
This application uses a Genetic Algorithm to optimise daily meal planning
based on calorie intake, protein requirement, and food price constraints.
""")




# Run Button

if st.button("Run Optimization"):
    best_solution, fitness_history = genetic_algorithm()
    best_meals = food_df.loc[best_solution]

    st.subheader("✅ Optimised Meal Plan")
    st.dataframe(best_meals)

    st.metric("Total Calories", int(best_meals[CAL_COL].sum()) if CAL_COL else 0)
    st.metric("Total Protein (g)", round(best_meals[PROTEIN_COL].sum(), 1) if PROTEIN_COL else 0)
    st.metric("Total Price (RM)", round(best_meals[PRICE_COL].sum(), 2) if PRICE_COL else 0)

    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.grid(True)
    st.pyplot(fig)
