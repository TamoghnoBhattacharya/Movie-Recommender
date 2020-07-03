import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.expand_frame_repr', False)

def cosine_sim_model(name, num_recommends):
    movies = pd.read_csv('datasets/movies.csv')
    tags = pd.read_csv('datasets/tags.csv')
    ratings = pd.read_csv('datasets/ratings.csv')
    movie_list_rating = ratings.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]
    mapping = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    tags.drop(['timestamp'], 1, inplace=True)
    ratings.drop(['timestamp'], 1, inplace=True)
    mixed = pd.merge(movies, tags, on='movieId', how='left')
    mixed.fillna("", inplace=True)
    mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
    final = pd.merge(movies, mixed, on='movieId', how='left')
    final['metadata'] = final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)
    final.drop(['movieId', 'genres', 'tag'], 1, inplace=True)
    tf = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tf.fit_transform(final['metadata'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = final[final['title'] == name].index[0]
    scores = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    scores = scores.iloc[1:num_recommends+1]
    recommendations = pd.DataFrame({'Title' : final['title'][list(scores.index)].tolist(), 'Score' : scores.tolist()})
    return recommendations

