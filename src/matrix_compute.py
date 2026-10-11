from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def generate_similarity_matrix(dataframe, max_features=5000):
    """
    Transforms token vectors into multidimensional space arrays and calculates pairwise similarity paths.
    """
    cv = CountVectorizer(max_features=max_features, stop_words='english')
    vectors = cv.fit_transform(dataframe['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    # Cache active model assets via serial binary blocks (Pickle dumps)
    pickle.dump(dataframe, open('movies.pkl', 'wb'))
    pickle.dump(dataframe.to_dict(), open('movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    
    return similarity

def fetch_nearest_neighbors(movie_title, dataframe, similarity_matrix, top_k=5):
    """
    Locates the targeted token record and returns the top-K spatially proximate titles.
    """
    try:
        movie_index = dataframe[dataframe['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        recommendations = []
        for i in movies_list:
            recommendations.append(dataframe.iloc[i[0]].title)
        return recommendations
    except IndexError:
        return [f"Error: Target point '{movie_title}' missing from evaluation coordinate matrix."]
