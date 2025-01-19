import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime
import json

# Initialize NLTK resources
nltk.data.path.append(r"C:\Users\sujal\AppData\Roaming\nltk_data")
try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Step 1: Connect to SQLite Database
try:
    conn = sqlite3.connect('games.db')
    cursor = conn.cursor()
except sqlite3.Error as e:
    print(f"Database connection error: {e}")
    exit()

# Step 2: Fetch the games data from the database
try:
    df = pd.read_sql_query("SELECT * FROM games", conn)
except Exception as e:
    print(f"Error reading data from the database: {e}")
    conn.close()
    exit()

# Step 3: Handle Missing Values
df['genres'] = df['genres'].fillna('Unknown')
df['platforms'] = df['platforms'].fillna('Unknown')
df['summary'] = df['summary'].fillna('No summary available')
df['rating'] = df['rating'].fillna(df['rating'].mean())  # Replace missing ratings with the average

# Step 4: Normalize Rating
scaler = MinMaxScaler()
df['rating_normalized'] = scaler.fit_transform(df[['rating']])

# Step 5: Standardize and Clean Text Data
df['name'] = df['name'].str.strip().str.title()
df['genres'] = df['genres'].str.strip().str.title()
df['platforms'] = df['platforms'].str.strip().str.title()

# Step 6: Tokenize Text (for NLP tasks, if needed)
# Tokenize summary text and convert to JSON strings
df['summary_tokens'] = df['summary'].apply(
    lambda x: json.dumps(word_tokenize(x.lower())) if isinstance(x, str) else '[]'
)

# Step 7: Format Date Fields
df['release_date'] = pd.to_datetime(df['first_release_date'], unit='s', errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Step 8: Feature Engineering
df['genre_count'] = df['genres'].apply(lambda x: len(x.split(', ')) if isinstance(x, str) else 0)
df['platform_count'] = df['platforms'].apply(lambda x: len(x.split(', ')) if isinstance(x, str) else 0)

current_year = datetime.now().year
df['popularity'] = df['rating'] * (current_year - df['release_year'].fillna(current_year) + 1)

# Step 9: Remove Duplicates
df = df.drop_duplicates(subset=['id', 'name'])

# Step 10: Filter Irrelevant or Outlier Data
df = df[df['rating'] > 50]
df = df[(df['rating'] >= df['rating'].quantile(0.05)) & (df['rating'] <= df['rating'].quantile(0.95))]

# Step 11: Encode Categorical Variables (One-Hot Encoding)
df['genres'] = df['genres'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))) if isinstance(x, str) else x)
df['platforms'] = df['platforms'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))) if isinstance(x, str) else x)

genres_encoded = df['genres'].str.get_dummies(sep=', ')
platforms_encoded = df['platforms'].str.get_dummies(sep=', ')

df = pd.concat([df, genres_encoded, platforms_encoded], axis=1)

# Remove duplicate columns, if any
df = df.loc[:, ~df.columns.duplicated()]

# Step 12: Save Preprocessed Data to Database
try:
    # Convert all columns with unsupported types to JSON or strings
    for col in df.columns:
        if df[col].apply(type).eq(list).any() or df[col].apply(type).eq(dict).any():
            df[col] = df[col].apply(json.dumps)
    df.to_sql('preprocessed_games', conn, if_exists='replace', index=False)
except sqlite3.Error as e:
    print(f"Error saving data to the database: {e}")
    conn.close()
    exit()

# Step 13: Verify Preprocessed Data
print("Preprocessed Data Sample:")
print(df.head())
print("Data Info:")
print(df.info())

# Close database connection
conn.close()
