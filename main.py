import sys
from correlation_model import *
from similarity_model import *
from matrix_factorization_model import *
from deep_learning_model import *

print('Enter 1 for item rating correlation based recommendations')
print('Enter 2 for item metadata similarity based recommendations')
print('Enter 3 for matrix-factorization model based recommendations')
print('Enter 4 for deep-learning embedding model based recommendations')
ch = int(input())

if ch==1 or ch==2:
    print('Enter movie name and year from dataset')
    name = str(input())
    print('Enter number of recommendations you want')
    num_recommends = int(input())
    if ch==1:
        recommendations = corr_model(name, num_recommends)
    else:
        recommendations = cosine_sim_model(name, num_recommends)
elif ch==3 or ch==4:
    print('Enter valid user ID')
    userID = int(input())
    print('Enter number of recommendations you want')
    num_recommends = int(input())
    if ch==3:
        recommendations = matrix_fact_model(userID, num_recommends)
    else:
        recommendations = deep_learning_embeddings_model(userID, num_recommends)  
else:
    print('Invalid Choice')
    sys.exit(0)

print(recommendations)
