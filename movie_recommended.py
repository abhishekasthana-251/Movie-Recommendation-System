import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
ps=PorterStemmer()
movies =pd.read_csv('tmdb_5000_movies.csv')
credits =pd.read_csv('tmdb_5000_credits.csv')

# print(movies.head(1))
# print(credits.head(1))
# print(credits.head(1)['crew'].values)

# merging the two dateset in one as one date set on the one same parameter

movies=movies.merge(credits,on='title')
# print(movies.head(1))

# taking the important factor which help me to make my project  from the dateset

# genres,id,keyword, title,overview, cast,crew

movies=movies[['id','title','overview','genres','keywords','cast','crew']]
# print(movies.head())

# handeling the missing data 

# print(movies.isnull().sum()) ##check the missing values
# overview have three missing values ,drop that values

movies.dropna(inplace=True) ##dropna ->delete the row which have missing values
# print(movies.isnull().sum())  
# print(movies.duplicated().sum())  #there is no duplicate value

# print(movies.iloc[0].genres)
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
#{'action', 'advanture',fantasy,scifi}

# the genres is in string ,first i convert it into list and i only want the name 

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):  #ast.literal_eval is for convert the string into list
        L.append(i['name'])
    return L

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

#in the cast there is many name is present ,but i want only first 3,4 name 

def convert3(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count !=4:
            L.append(i['name'])
            count+=1
        else:
            break
    return L

movies['cast']=movies['cast'].apply(convert3)
# now in the crew i only want the director name 

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew']=movies['crew'].apply(fetch_director)



movies['overview']=movies['overview'].apply(lambda x:x.split())
# print(movies['overview'])

# removing the space between the word which help us to get the proper recommendation
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
 
movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df=movies[['id','title','tags']]
new_df.loc[:,'tags']=new_df['tags'].apply(lambda x:" ".join(x))
# print(new_df['tags'][0])

# converting the data in dataset into the lower case which help in recommendation easily

new_df.loc[:,'tags']=new_df['tags'].apply(lambda x:x.lower())

cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
# print(cv.get_feature_names_out())

# managing the similar words 

def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)

similarity=cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])
# print(similarity.shape)
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list =sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:7]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))