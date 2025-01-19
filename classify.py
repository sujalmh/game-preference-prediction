import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from tabulate import tabulate

def load_data():
    """Load game data from the database."""
    conn = sqlite3.connect("games.db")
    df = pd.read_sql_query("SELECT * FROM preprocessed_games", conn)
    conn.close()
    return df

def preprocess_data(df):
    """Preprocess game data for recommendation."""
    # One-hot encode genres and platforms
    genres_encoded = df['genres'].str.get_dummies(sep=", ")
    platforms_encoded = df['platforms'].str.get_dummies(sep=", ")

    # Combine features into a single matrix
    feature_matrix = pd.concat([genres_encoded, platforms_encoded, df['rating_normalized']], axis=1)

    # Normalize the rating
    scaler = MinMaxScaler()
    feature_matrix['rating_normalized'] = scaler.fit_transform(feature_matrix[['rating_normalized']])

    return feature_matrix

def compute_similarity(feature_matrix):
    """Compute the cosine similarity matrix."""
    return cosine_similarity(feature_matrix)

def recommend_games(game_names, similarity_matrix, df, top_n=5):
    """
    Recommend games based on a list of games a user has played.

    Args:
        game_names (list): List of games the user has played.
        similarity_matrix (numpy.array): Precomputed similarity matrix.
        df (pandas.DataFrame): DataFrame containing game details.
        top_n (int): Number of recommendations to return.

    Returns:
        pandas.DataFrame: Recommended games.
    """
    # Find indices of games the user has played
    game_indices = df[df['name'].isin(game_names)].index.tolist()

    # If none of the user's games exist in the dataset, return no recommendations
    if not game_indices:
        return pd.DataFrame({"message": ["No matching games found in the database."]})

    # Compute the average similarity score for all games
    similarity_scores = np.mean(similarity_matrix[game_indices], axis=0)

    # Sort by similarity score
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Exclude games the user has already played
    recommended_indices = [i for i in sorted_indices if i not in game_indices]

    # Return top N recommended games
    recommendations = df.iloc[recommended_indices].head(top_n)
    return recommendations[['name', 'genres', 'platforms', 'rating']]

def cli():
    """Command-Line Interface for the Game Recommendation System."""
    print("Welcome to the Game Recommendation System!")
    print("Loading data...")
    
    # Load and preprocess data
    df = load_data()
    feature_matrix = preprocess_data(df)
    similarity_matrix = compute_similarity(feature_matrix)

    print("Data loaded successfully!")
    
    # Prepare the auto-completer for game names
    game_completer = WordCompleter(df['name'].tolist(), ignore_case=True)

    # Main CLI loop
    while True:
        print("\nEnter the games you've played (comma-separated), or type 'exit' to quit:")
        user_input = prompt("> ", completer=game_completer)

        if user_input.lower().strip() == 'exit':
            print("Thank you for using the Game Recommendation System!")
            break

        # Get user games
        user_games = [game.strip().title() for game in user_input.split(",")]

        # Generate recommendations
        print("\nGenerating recommendations...")
        recommendations = recommend_games(user_games, similarity_matrix, df)

        # Display recommendations
        if 'message' in recommendations.columns:
            print("\n" + recommendations['message'][0])
        else:
            print("\nRecommended Games:")
            table = tabulate(
                recommendations,
                headers=["Game Name", "Genres", "Platforms", "Rating"],
                tablefmt="fancy_grid",
                showindex=False,
            )
            print(table)

if __name__ == "__main__":
    cli()
