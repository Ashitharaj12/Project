import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.title("Food Expiry Management & Recipe Suggestion")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your food expiry CSV file", type=["csv"])
if uploaded_file:
    food_df = pd.read_csv(uploaded_file)

    # Convert expiry date to datetime
    food_df['Expiry Date'] = pd.to_datetime(food_df['Expiry Date'], errors='coerce')
    food_df.dropna(subset=['Expiry Date'], inplace=True)
    food_df['Days Left'] = (food_df['Expiry Date'] - datetime.today()).dt.days
    food_df = food_df[food_df['Days Left'] >= 0]  # Remove expired items

    # Show first few rows
    st.subheader("Processed Data")
    st.write(food_df.head())

    # Standardize Days Left column
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(food_df[['Days Left']])

    # Elbow Method for choosing clusters
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_df)
        inertia.append(kmeans.inertia_)

    # Plot elbow method
    st.subheader("Elbow Method for Optimal Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_range, inertia, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Apply K-Means clustering
    k = 3  # Pre-defined number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    food_df['Cluster'] = kmeans.fit_predict(scaled_df)

    # Show silhouette score
    sil_score = silhouette_score(scaled_df, food_df['Cluster'])
    st.write(f"**Silhouette Score:** {sil_score}")

    # Scatter plot of clusters
    st.subheader("Food Expiry Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=food_df.index, y=food_df['Days Left'], hue=food_df['Cluster'], palette='viridis', s=100, alpha=0.7, edgecolor='k', ax=ax)
    ax.set_xlabel("Index")
    ax.set_ylabel("Days Left Until Expiry")
    ax.set_title("Food Expiry Clusters")
    st.pyplot(fig)

    # Recipe suggestions
    recipes = {
        'Milk': ['Pancakes', 'Milkshake'],
        'Bread': ['Sandwich', 'Bread Pudding'],
        'Eggs': ['Omelette', 'Scrambled Eggs'],
        'Tomatoes': ['Tomato Soup', 'Salad'],
        'Cheese': ['Grilled Cheese', 'Mac & Cheese'],
        'Chicken': ['Chicken Curry', 'Grilled Chicken'],
        'Rice': ['Fried Rice', 'Rice Pudding'],
        'Lettuce': ['Salad', 'Wrap'],
        'Yogurt': ['Smoothie', 'Parfait'],
        'Apples': ['Apple Pie', 'Fruit Salad'],
        'Banana': ['Banana Bread', 'Smoothie'],
        'Fish': ['Grilled Fish', 'Fish Curry'],
        'Potatoes': ['Mashed Potatoes', 'French Fries'],
        'Carrots': ['Carrot Soup', 'Stir Fry'],
        'Oranges': ['Orange Juice', 'Fruit Salad'],
        'Peppers': ['Stuffed Peppers', 'Stir Fry'],
        'Spinach': ['Spinach Salad', 'Smoothie'],
        'Butter': ['Butter Chicken', 'Garlic Bread'],
        'Cereal': ['Cereal with Milk', 'Granola Bars'],
        'Pasta': ['Pasta Salad', 'Spaghetti']
    }

    st.subheader("Recipe Suggestions for Expiring Items")
    expiring_soon = food_df[food_df['Days Left'] <= 3]
    for _, row in expiring_soon.iterrows():
        food_item = row.get('Food Item', 'Unknown')
        suggested_recipes = recipes.get(food_item, ['No recipes available'])
        st.write(f"**{food_item}** is expiring soon! Try these: {', '.join(suggested_recipes)}")

else:
    st.write("Please upload a CSV file to proceed.")