import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer

def load_and_clean_data(movies_path="data/tmdb_5000_movies.csv", credits_path="data/tmdb_5000_credits.csv"):
    """
    Ingests raw TMDB data schemas, normalizes metadata attributes, 
    and builds an nltk-stemmed text vector baseline block.
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    # Merge on title string matrix identifiers
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    
    # Helper parser for metadata JSON string blocks
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    # Helper parser to capture top 3 cast members
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    # Helper parser to capture the Director node
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    # Execute normalization mapping pipelines
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Structural string parsing (remove spatial gaps)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
    
    # Concatenate tokens into single vector anchor tag field
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    # Apply NLTK Porter Stemming matrix normalization
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
        
    new_df['tags'] = new_df['tags'].apply(stem)
    return new_df
