import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
pd.set_option('display.expand_frame_repr', False)

def matrix_fact_model(userID, num_recommends):
    ratings = pd.read_csv('datasets/ratings.csv', usecols=['userId', 'movieId', 'rating'])
    user_ratings = pd.read_csv('datasets/user_ratings.csv')
    movies = pd.read_csv('datasets/movies.csv')
    ratings = ratings.append(user_ratings)
    Ratings = ratings.pivot(index='userId', columns='movieId', values='rating')
    rating_matrix = Ratings.to_numpy()
    rating_mask = (~np.isnan(rating_matrix)).astype('int')
    m = np.sum(rating_mask)
    rating_matrix = np.nan_to_num(rating_matrix)
    user_mean = np.mean(rating_matrix, axis=1, keepdims=True)
    ratings_denorm = rating_matrix - user_mean
    U, sigma, Vt = svds(ratings_denorm, k=40)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_mean
    rmse = np.sqrt(1/m * np.sum(np.square(rating_matrix - predicted_ratings) * rating_mask))
    predictions = pd.DataFrame(predicted_ratings, index=Ratings.index, columns=Ratings.columns)
    sorted_predictions = predictions.loc[userID].sort_values(ascending=False)
    user_data = ratings[ratings.userId == userID]
    user_full = user_data.merge(movies, how='left', left_on='movieId', right_on='movieId').sort_values(['rating'], ascending=False)
    movies = movies[~movies['movieId'].isin(user_full['movieId'])]
    recommendations = movies.merge(pd.DataFrame(sorted_predictions).reset_index(), how='left', left_on='movieId', right_on='movieId').rename(columns = {userID:'Predictions'})
    recommendations = recommendations.sort_values('Predictions', ascending=False)
    recommendations = recommendations[:num_recommends]
    recommendations = recommendations.drop(columns=['movieId', 'genres'])
    return recommendations
