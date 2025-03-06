import pandas as pd

def load_sample_data():
    """
    Load sample movie and ratings data.
    In a real application, this would load from files or a database.
    """
    # Sample data - in a real scenario, you'd load this from a database or CSV files
    # Example user ratings (user_id, movie_id, rating)
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movie_id': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4],
        'rating': [5, 4, 3, 4, 5, 4, 3, 5, 3, 3, 4, 5]
    }

    # Example movie metadata
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight', 'Inception'],
        'genre': ['Drama', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Action, Adventure, Sci-Fi'],
        'description': ['Two imprisoned men bond over a number of years.',
                      'The aging patriarch of an organized crime dynasty transfers control to his son.',
                      'The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine.',
                      'Batman fights the menace known as the Joker.',
                      'A thief who steals corporate secrets through dream-sharing technology.']
    }

    # Create DataFrames
    ratings_df = pd.DataFrame(ratings_data)
    movies_df = pd.DataFrame(movies_data)
    
    return ratings_df, movies_df
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def create_user_movie_matrix(ratings_df):
    """
    Create a user-movie matrix from ratings data.
    """
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id', columns='movie_id', values='rating'
    ).fillna(0)
    
    return user_movie_matrix

def create_similarity_matrices(user_movie_matrix, movies_df):
    """
    Create similarity matrices for collaborative and content-based filtering.
    """
    # Create movie similarity matrix based on user ratings (collaborative filtering)
    movie_similarity_matrix = cosine_similarity(user_movie_matrix.T)
    
    # Create content similarity matrix based on movie descriptions (content-based filtering)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['description'])
    content_similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return movie_similarity_matrix, content_similarity_matrix
import numpy as np

def get_collaborative_recommendations(user_id, user_movie_matrix, movie_similarity_matrix, ratings_df, movies_df, n=3):
    """
    Generate movie recommendations based on collaborative filtering.
    
    Args:
        user_id: ID of the user to recommend movies for
        user_movie_matrix: Matrix of user-movie ratings
        movie_similarity_matrix: Matrix of movie similarities based on user ratings
        ratings_df: DataFrame of user ratings
        movies_df: DataFrame of movie information
        n: Number of recommendations to return
        
    Returns:
        List of tuples containing (movie_id, title, similarity_score)
    """
    # Get user ratings
    user_ratings = user_movie_matrix.loc[user_id].values
    
    # Get similarity scores
    sim_scores = np.zeros(len(movies_df))
    
    for i, rating in enumerate(user_ratings):
        if rating > 0:  # If the user has rated this movie
            # Add the similarity scores weighted by the user's rating
            movie_id = user_movie_matrix.columns[i]
            movie_index = movie_id - 1  # Assuming movie_ids start from 1
            sim_scores += rating * movie_similarity_matrix[movie_index]
    
    # Create a list of (movie_id, similarity score) tuples
    movie_scores = list(enumerate(sim_scores, 1))
    
    # Filter out already watched movies
    watched_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
    movie_scores = [(movie_id, score) for movie_id, score in movie_scores if movie_id not in watched_movies]
    
    # Sort by similarity score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n recommendations
    recommendations = [(movie_id, 
                        movies_df[movies_df['movie_id'] == movie_id]['title'].values[0], 
                        score) 
                      for movie_id, score in movie_scores[:n]]
    return recommendations
def get_content_based_recommendations(movie_id, content_similarity_matrix, movies_df, n=3):
    """
    Generate movie recommendations based on content similarity.
    
    Args:
        movie_id: ID of the movie to find similar movies for
        content_similarity_matrix: Matrix of content-based similarities
        movies_df: DataFrame of movie information
        n: Number of recommendations to return
        
    Returns:
        List of tuples containing (movie_id, title, similarity_score)
    """
    movie_index = movie_id - 1  # Assuming movie_ids start from 1
    
    # Get similarity scores for this movie with all other movies
    similarity_scores = list(enumerate(content_similarity_matrix[movie_index], 1))
    
    # Filter out the movie itself
    similarity_scores = [(id, score) for id, score in similarity_scores if id != movie_id]
    
    # Sort by similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n recommendations
    recommendations = [(movie_id, 
                        movies_df[movies_df['movie_id'] == movie_id]['title'].values[0], 
                        score) 
                      for movie_id, score in similarity_scores[:n]]
    return recommendations
def get_popular_movies(ratings_df, movies_df, n=3):
    """
    Get most popular movies based on average ratings.
    
    Args:
        ratings_df: DataFrame of user ratings
        movies_df: DataFrame of movie information
        n: Number of recommendations to return
        
    Returns:
        List of tuples containing (movie_id, title, average_rating)
    """
    movie_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()
    movie_ratings = movie_ratings.sort_values('rating', ascending=False)
    
    popular_movies = []
    for _, row in movie_ratings.head(n).iterrows():
        movie_id = row['movie_id']
        title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
        popular_movies.append((movie_id, title, row['rating']))
    
    return popular_movies
    from collaborative_filtering import get_collaborative_recommendations
    from content_based_filtering import get_content_based_recommendations
    from popular_movies import get_popular_movies

def get_hybrid_recommendations(user_id, user_movie_matrix, movie_similarity_matrix, 
                              content_similarity_matrix, ratings_df, movies_df, n=3):
    """
    Generate hybrid recommendations combining collaborative and content-based approaches.
    
    Args:
        user_id: ID of the user to recommend movies for
        user_movie_matrix: Matrix of user-movie ratings
        movie_similarity_matrix: Matrix of movie similarities based on user ratings
        content_similarity_matrix: Matrix of content-based similarities
        ratings_df: DataFrame of user ratings
        movies_df: DataFrame of movie information
        n: Number of recommendations to return
        
    Returns:
        List of tuples containing (movie_id, title, similarity_score)
    """
    # Get recommendations from collaborative filtering
    cf_recommendations = get_collaborative_recommendations(
        user_id, user_movie_matrix, movie_similarity_matrix, ratings_df, movies_df, n=n
    )
    
    # If the user has no recommendations from collaborative filtering,
    # recommend based on the highest-rated movie's content
    if not cf_recommendations:
        # Find the highest rated movie by this user
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            # New user, recommend the most popular movies
            return get_popular_movies(ratings_df, movies_df, n=n)
        
        highest_rated_movie = user_ratings.loc[user_ratings['rating'].idxmax()]['movie_id']
        return get_content_based_recommendations(
            highest_rated_movie, content_similarity_matrix, movies_df, n=n
        )
    
    return cf_recommendations
    from data_loader import load_sample_data
    from matrix_builder import create_user_movie_matrix, create_similarity_matrices
    from collaborative_filtering import get_collaborative_recommendations
    from content_based_filtering import get_content_based_recommendations
    from popular_movies import get_popular_movies
    from hybrid_recommender import get_hybrid_recommendations

class RecommendationSystem:
    def __init__(self, ratings_df=None, movies_df=None):
        """
        Initialize the recommendation system with provided data or use sample data.
        """
        if ratings_df is None or movies_df is None:
            self.ratings_df, self.movies_df = load_sample_data()
        else:
            self.ratings_df = ratings_df
            self.movies_df = movies_df
        
        # Create matrices
        self.user_movie_matrix = create_user_movie_matrix(self.ratings_df)
        self.movie_similarity_matrix, self.content_similarity_matrix = create_similarity_matrices(
            self.user_movie_matrix, self.movies_df
        )
    
    def collaborative_filtering(self, user_id, n=3):
        """
        Recommend movies based on collaborative filtering.
        """
        return get_collaborative_recommendations(
            user_id, self.user_movie_matrix, self.movie_similarity_matrix, 
            self.ratings_df, self.movies_df, n=n
        )
    
    def content_based_filtering(self, movie_id, n=3):
        """
        Recommend similar movies based on content.
        """
        return get_content_based_recommendations(
            movie_id, self.content_similarity_matrix, self.movies_df, n=n
        )
    
    def get_popular_movies(self, n=3):
        """
        Get most popular movies based on average ratings.
        """
        return get_popular_movies(self.ratings_df, self.movies_df, n=n)
    
    def hybrid_recommendations(self, user_id, n=3):
        """
        Combine collaborative and content-based filtering approaches.
        """
        return get_hybrid_recommendations(
            user_id, self.user_movie_matrix, self.movie_similarity_matrix,
            self.content_similarity_matrix, self.ratings_df, self.movies_df, n=n
        )
        from recommendation_system import RecommendationSystem

def main():
    # Create recommendation system
    recommender = RecommendationSystem()
    
    # Example 1: Collaborative Filtering
    print("\n===== Collaborative Filtering Recommendations =====")
    user_id = 1
    print(f"Recommendations for User {user_id}:")
    cf_recommendations = recommender.collaborative_filtering(user_id=user_id)
    for movie_id, title, score in cf_recommendations:
        print(f"Movie: {title}, Similarity Score: {score:.2f}")
    
    # Example 2: Content-Based Filtering
    print("\n===== Content-Based Recommendations =====")
    movie_id = 1
    movie_title = recommender.movies_df[recommender.movies_df['movie_id'] == movie_id]['title'].values[0]
    print(f"Movies similar to '{movie_title}':")
    cb_recommendations = recommender.content_based_filtering(movie_id=movie_id)
    for movie_id, title, score in cb_recommendations:
        print(f"Movie: {title}, Similarity Score: {score:.2f}")
    
    # Example 3: Hybrid Recommendations
    print("\n===== Hybrid Recommendations =====")
    user_id = 2
    print(f"Hybrid recommendations for User {user_id}:")
    hybrid_recommendations = recommender.hybrid_recommendations(user_id=user_id)
    for movie_id, title, score in hybrid_recommendations:
        print(f"Movie: {title}, Score: {score:.2f}")
    
    # Example 4: Popular Movies (for new users)
    print("\n===== Popular Movies =====")
    popular_movies = recommender.get_popular_movies()
    print("Recommendations for new users:")
    for movie_id, title, avg_rating in popular_movies:
        print(f"Movie: {title}, Average Rating: {avg_rating:.2f}")

if __name__ == "__main__":
    main()    