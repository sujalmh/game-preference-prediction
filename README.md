# Game Recommendation System

## Overview
The Game Recommendation System is a command-line tool designed to suggest games based on user preferences. By entering the names of games you've played, the system will recommend similar games using preprocessed game data and cosine similarity metrics.

![image](https://github.com/user-attachments/assets/662361d6-ea83-4970-b6ec-c437043ebe99)

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Future Improvements](#future-improvements)

## Features
- Customizable game title autocompletion using `prompt_toolkit`.
- Game similarity computation based on genres, platforms, and normalized ratings.
- Interactive command-line interface for user inputs.
- Recommendations displayed in a clean, tabular format.

## Dependencies
The following libraries are required to run the system:

```bash
pandas
numpy
scikit-learn
sqlite3
prompt_toolkit
tabulate
```

You can install the dependencies via pip:

```bash
pip install pandas numpy scikit-learn prompt_toolkit tabulate
```

## Installation
1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Ensure the `games.db` SQLite database is present in the root directory with a `preprocessed_games` table.

3. Install the required Python packages as listed in the [Dependencies](#dependencies) section.

## Usage
To run the Game Recommendation System, execute the following command:

```bash
python <script_name>.py
```

### Interactive CLI Workflow
1. The system will load game data from `games.db`.
2. Enter the names of games you have played, separated by commas (e.g., `Game1, Game2`).
3. The system will generate and display a list of recommended games based on your input.
4. Type `exit` to quit the application.

## Code Structure

### Modules and Functions

#### **Custom Completer**
```python
class GameCompleter(Completer):
    """Custom completer for game titles with multi-word support."""
```
Provides autocompletion for game titles during user input.

#### **Data Loading**
```python
def load_data():
    """Load game data from the database."""
```
Loads game data from a SQLite database (`games.db`) into a pandas DataFrame.

#### **Preprocessing**
```python
def preprocess_data(df):
    """Preprocess game data for recommendation."""
```
Encodes genres and platforms as binary features and normalizes ratings.

#### **Similarity Computation**
```python
def compute_similarity(feature_matrix):
    """Compute the cosine similarity matrix."""
```
Computes a similarity matrix using cosine similarity.

#### **Recommendation Engine**
```python
def recommend_games(game_names, similarity_matrix, df, top_n=5):
    """Recommend games based on a list of games a user has played."""
```
Generates game recommendations excluding the user-provided games.

#### **Command-Line Interface**
```python
def cli():
    """Command-Line Interface for the Game Recommendation System."""
```
Facilitates user interaction and displays recommendations in a tabular format.

### Database Requirements
The system expects an SQLite database (`games.db`) with a table named `preprocessed_games`. The table should contain the following columns:
- `name` (str): Game title.
- `genres` (str): Comma-separated list of genres.
- `platforms` (str): Comma-separated list of platforms.
- `rating_normalized` (float): Normalized rating of the game.
