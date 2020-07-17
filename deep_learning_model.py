import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Concatenate
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
pd.set_option('display.expand_frame_repr', False)

ratings = pd.read_csv('datasets/ratings.csv', usecols=['userId', 'movieId', 'rating'])
user_enc = LabelEncoder()
item_enc = LabelEncoder()
ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
n_users = ratings['user'].nunique()
ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)
n_movies = ratings['movie'].nunique()
ratings['rating'] = ratings['rating'].values.astype(np.float32)
min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])


X = ratings[['user', 'movie']].values
Y = ratings['rating'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
n_factors = 50
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

user = Input(shape=(1,))
u = Embedding(n_users, n_factors)(user)
u = Reshape((n_factors,))(u)
movie = Input(shape=(1,))
m = Embedding(n_movies, n_factors)(movie)
m = Reshape((n_factors,))(m)
x = Concatenate()([u, m])
x = Dense(10, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x)
x = Lambda(lambda x:x * (max_rating - min_rating) + min_rating)(x)
model = Model(inputs=[user, movie], outputs=x)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(X_train_array, Y_train, batch_size=64, epochs=10, verbose=2, validation_data=(X_test_array, Y_test))

