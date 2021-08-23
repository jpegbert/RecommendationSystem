import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

def movie_map():
    movies = read_csv('movies.dat', sep='::', header=None,engine='python')
    value = movies[0].unique()
    index = range(movies[0].nunique())
    return dict(zip(value,index))
    
d = movie_map()

def read_rating(file_path):
    rating_mat = list()
    with open(file_path) as fp:
        for line in fp:
            line = line.strip().split('::')
            user, item, rating = line[0], d[int(line[1])], int(line[2])/5
            rating_mat.append( [user, item, rating] )
    return np.array(rating_mat).astype('float32')

def read_feature(file_path):
    item = read_csv(file_path, sep='::', header=None,engine='python')
    item[0] = item[0].apply(lambda x:d[int(x)])
    for i in range(1,3):
        item[i] = LabelEncoder().fit_transform(item[i])
    return item.as_matrix()