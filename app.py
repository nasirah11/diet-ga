import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("food_data.csv")

food_df = load_data()

# Normalize column names (VERY IMPORTANT)
food_df.columns = food_df.columns.str.strip().str.lower()

# =========================
# Sidebar – GA Parameters
# =========================
st.sidebar.header("Genetic Algorithm Parameters")

TARGET_CALORIES = st.sidebar.slider("Target Calories", 1500, 3000, 2000)
POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 10, 200, 50)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
BUDGET = st.sidebar.slider("Max Budget (RM)", 10, 50, 20)

# =========================
# GA Configuration
# =========================
NUM_MEALS = 4  # Breakfast, Lunch, Dinner, Snack

def create_individual():
    return random.sample(list(food_df.index), NUM_MEALS)

# =========================
# Fitness Function (SAFE)
# =========================
def fitness(individual):
    meals = food_df.loc[individual]

    total_calories = meals.get("calories", pd.Series(0)).sum()
    total_cost = meals.get("cost", pd.Series(0)).sum()
    total_protein = meals.get("protein", pd.Series(0)).sum()

    calorie_penalty = abs(TARGET_CALORIES - total_calories)
    cost_penalty = max(0, total_cost - BUDGET) * 10
    protein_penalty = max(0, 50 - total_protein) * 5

    fitness_score = (
        10000
        - calorie_penalty * 5
        - cost_penalty
        - protein_penalty
    )

    return fitness_score

# =========================
# Selection – Tournament
# =========================
def selection(population):
    selected = random.sample(population, 3)
    selected.sort(key=lambda x: fitness(x), reverse=True)
    return selected[0]

# =========================
# Crossover
# =========================
def crossover(parent1, parent2):
    point = random.randint(1, NUM_MEALS - 1)
    child = parent1[:point] + parent2[point:]
    return list(dict.fromkeys(child))[:NUM_MEALS]

# =========================
# Mutation
# =========================
def mutation(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_MEALS - 1)
        individual[index] = random.choice(food_df.index)
    return individual

# =========================
# Genetic Algorithm Loop
# =========================
def genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    best_fitness_history = []

    for _ in range(GENERATIONS):
        new_population = []

        for _ in range(POP_SIZE):
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)

        population = new_population
        best_individual = max(population, key=lambda x: fitness(x))
        best_fitness_history.append(fitness(best_individual))

    return best_individual, best_fitness_history

# =========================
# Streamlit UI
# =========================
st.title("🍽️ Diet Meal Planning Optimisation using Genetic Algorithm")

st.write(
    """
    This application applies a Genetic Algorithm (GA) to optimise daily meal plans
    based on calorie requirements, protein intake, and budget constraints.
    """
)

# =========================
# Run Button
# =========================
if st.button("Run Optimization"):
    best_solution, fitness_history = genetic_algorithm()
    best_meals = food_df.loc[best_solution]

    st.subheader("✅ Optimised Meal Plan")
    st.dataframe(best_meals)

    st.metric("Total Calories", int(best_meals.get("calories", pd.Series(0)).sum()))
    st.metric("Total Cost (RM)", round(best_meals.get("cost", pd.Series(0)).sum(), 2))

    # =========================
    # Fitness Convergence Plot
    # =========================
    st.subheader("📈 Fitness Convergence")

    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.grid(True)

    st.pyplot(fig)
