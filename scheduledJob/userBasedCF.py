from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import pandas as pd
import os
import json

#Connect to the database
client = MongoClient('mongodb://%s:%s@localhost:27017/%s?authMechanism=SCRAM-SHA-1' % 
                     (os.environ.get('LIPSTICKFINDER_USER'), os.environ.get('LIPSTICKFINDER_PSW'), os.environ.get('LIPSTICKFINDER_DATABSE')))
db = client['LipstickFinder']

#Parameters to be set
top_n_neighbour = 3
numOfRecommendation = 5

#Read Data from MongoDB
data = db.rating.find({},{'_id':0, "userID":1,"lipstickID":1,"rating":1})
data = list(data)
df = pd.DataFrame(data)

#Adjust the rating by mean value
Mean = df.groupby(by="userID",as_index=False)['rating'].mean()
Rating_avg = pd.merge(df,Mean,on='userID')
Rating_avg['adj_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']

#Construct the rating matrix
check = pd.pivot_table(Rating_avg,values='rating_x',index='userID',columns='lipstickID')
final = pd.pivot_table(Rating_avg,values='adj_rating',index='userID',columns='lipstickID')

# Replacing NaN by Lipstick Average
final_lipstick = final.fillna(final.mean(axis=0))

# Replacing NaN by user Average
final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

# user similarity on replacing NAN by user avg
b = cosine_similarity(final_user)
np.fill_diagonal(b, 0 )
similarity_with_user = pd.DataFrame(b,index=final_user.index)
similarity_with_user.columns=final_user.index
similarity_with_user.head()

# user similarity on replacing NAN by item(lipstick) avg
cosine = cosine_similarity(final_lipstick)
np.fill_diagonal(cosine, 0 )
similarity_with_lipstick = pd.DataFrame(cosine,index=final_lipstick.index)
similarity_with_lipstick.columns=final_user.index
similarity_with_lipstick.head()

# Function used to find the similarity of n-nearest neighbours 
def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

# top n similarest neighbours for each user
sim_top_n = find_n_neighbours(similarity_with_lipstick, top_n_neighbour)

# top n similarest neighbours for each user
# sim_top_n = find_n_neighbours(similarity_with_user, top_n_neighbour)

Lipstick_user = Rating_avg.groupby(by = 'userID')['lipstickID'].apply(lambda x:','.join(x))

def getRecommendation(user,sim_top_n,numOfRecommendation):
    Lipstick_rated_by_user = check.columns[check[check.index==user].notna().any()].tolist()
    a = sim_top_n[sim_top_n.index==user].values
    b = a.squeeze().tolist()
    d = Lipstick_user[Lipstick_user.index.isin(b)]
    l = ','.join(d.values)
    Lipstick_rated_by_similar_users = l.split(',')
    Lipstick_under_consideration = list(set(Lipstick_rated_by_similar_users)-set(Lipstick_rated_by_user))
    Lipstick_under_consideration = list(Lipstick_under_consideration)
    score = []
    for item in Lipstick_under_consideration:
        c = final_lipstick.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = Mean.loc[Mean['userID'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity_with_lipstick.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        score.append(final_score)
    data = pd.DataFrame({'lipstickID':Lipstick_under_consideration,'score':score})
    recommendation = data.sort_values(by='score',ascending=False).head(numOfRecommendation)
    recommendation = recommendation.lipstickID.values.tolist()
    return recommendation

Users = Mean.userID.values.tolist()
# db.cfRecommendation.drop()
cfRecommendation = db["cfRecommendation"]
for userID in Users:
    recommendation = getRecommendation(userID, sim_top_n, numOfRecommendation)
    count = cfRecommendation.count_documents({"userID":userID})
    if count == 0:
        cfRecommendation.insert_one({"userID":userID, "lipsticksID":recommendation})
    else:
        cfRecommendation.update_one({"userID":userID}, {"$set":{"lipsticksID":recommendation}})
    
    
    
    
    
    
    
    
    
    
    
    
    