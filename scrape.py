import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from tabulate import tabulate


class GameCompleter(Completer):
    """Custom completer for game titles with multi-word support."""

    def __init__(self, game_list):
        self.game_list = game_list

    def get_completions(self, document, complete_event):
        text = document.text.split(",")[-1].strip()
        for game in self.game_list:
            if game.lower().startswith(text.lower()):
                yield Completion(game, start_position=-len(text))


def load_data():
    """Load game data from the database."""
    conn = sqlite3.connect("games.db")
    df = pd.read_sql_query("SELECT * FROM preprocessed_games", conn)
    conn.close()
    return df


def preprocess_data(df):
    """Preprocess game data for recommendation."""
    genres_encoded = df['genres'].str.get_dummies(sep=", ")
    platforms_encoded = df['platforms'].str.get_dummies(sep=", ")

    feature_matrix = pd.concat([genres_encoded, platforms_encoded, df['rating_normalized']], axis=1)

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
    game_indices = df[df['name'].isin(game_names)].index.tolist()

    if not game_indices:
        return pd.DataFrame({"message": ["No matching games found in the database."]})

    similarity_scores = np.mean(similarity_matrix[game_indices], axis=0)

    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommended_indices = [i for i in sorted_indices if i not in game_indices]

    recommendations = df.iloc[recommended_indices].head(top_n)
    return recommendations[['name', 'genres', 'platforms', 'rating']]


def cli():
    """Command-Line Interface for the Game Recommendation System."""
    print("Welcome to the Game Recommendation System!")
    print("Loading data...")

    df = load_data()
    feature_matrix = preprocess_data(df)
    similarity_matrix = compute_similarity(feature_matrix)

    print("Data loaded successfully!")

    game_completer = GameCompleter(df['name'].tolist())

    while True:
        print("\nEnter the games you've played (comma-separated), or type 'exit' to quit:")
        user_input = prompt("> ", completer=game_completer)

        if user_input.lower().strip() == 'exit':
            print("Thank you for using the Game Recommendation System!")
            break

        user_games = [game.strip().title() for game in user_input.split(",")]

        print("\nGenerating recommendations...")
        recommendations = recommend_games(user_games, similarity_matrix, df)

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
