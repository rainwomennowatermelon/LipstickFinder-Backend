import pymongo
import numpy as np
import pandas as pd
import math

colorDict = {'Red'   :'#FF0000', 
             'Pink'  :'#FFC0CB', 
             'Orange':'#FF7300',
             'Brown' :'#A16B47',
             'Purple':'#7400A1',}

def hexaToRgb(hexa):
    r = int(hexa[1:3], 16)
    g = int(hexa[3:5], 16)
    b = int(hexa[5:7], 16)
    return [r,g,b]

def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

def getLipstickWithShortestNDistance(lipstickdf, item, num):
    lipstickdf['distance'] = lipstickdf.apply(lambda x: ColourDistance(hexaToRgb(x.color),hexaToRgb(colorDict[item])),axis = 1)
    lipstickdf.sort_values(by='distance',ascending=True).head(num)
    lipstickdf=lipstickdf.drop(['distance'], axis=1)
    return lipstickdf
    
def getLipstickinfo(client, lipstickID):
    return (client.db.lipsticks.find({'lipstick_id':lipstickID},{'_id': 0, 'brand': 1, 'series': 1, 'liquid': 1, 'texture': 1, 'color': 1, 'price': 1, 'name': 1, 'lipstick_id': 1}))[0]

def getRecommendationFromCfRecommendation(client, userID):
    count = client.db.cfRecommendation.count_documents({"userID": userID})
    cfResult = []
    if count == 1:
        reply = client.db.cfRecommendation.find({"userID": userID})[0]
        lipsticks = reply["lipsticksID"]
        for lipstickID in lipsticks:
            cfResult.append(getLipstickinfo(client, lipstickID))
    return cfResult  
    
def getRecommendationFromTopLipsticks(client, userID, num, result):
    topLipResult = []
    #Get likelist
    likelist = []
    reply = client.db.likes.count_documents({"userID":userID})
    if reply == 1:
        likelist = client.db.likes.find({"userID":userID})[0]['lipstickID']
    
    #Get cfrecommendationlist
    cfrecommendationlist =[]
    if len(result) != 0:
        cfRecommendationdata = pd.DataFrame(result)
        cfrecommendationlist = cfRecommendationdata.lipstick_id.values.tolist()
        
    #Get the features users like
    features_dict = {}
    reply = client.db.Users.count_documents({"ID":userID})
    if reply == 1:
        features_dict = client.db.Users.find({"ID":userID},{'_id':0, 'color':1, 'kind':1, 'texture':1})[0]
    print(features_dict)
    color = features_dict['color']
    kind = features_dict['kind']
    texture = features_dict['texture']
    
    liquid = []
    if 'Lipstick' in kind:
        liquid.append(False)  
    if 'Lip glaze' in kind:
        liquid.append(True)
        
    #Get the lipsticks from Toplipsticks meet the requirement of kind and texture
    filteredresult = client.db.topLipsticks.find({'liquid':{'$in':liquid},'texture':{'$in':texture}},
                                                 {'_id': 0, 'rank': 1,'brand': 1, 'series': 1,
                                                  'liquid': 1, 'texture': 1, 'color': 1, 
                                                  'price': 1, 'name': 1, 'lipstick_id': 1})
    data = list(filteredresult)
    df = pd.DataFrame(data)
    
    #Filter the lipsticks existing in likelist and result
    lipsticklist = df.lipstick_id.values.tolist()
    lipsticklist = list(set(lipsticklist)-set(likelist)-set(cfrecommendationlist))
    df = df[df["lipstick_id"].isin(lipsticklist)]
    
    #Get Top num lipsticks closet to different color
    topLipsticksResult = pd.DataFrame(columns=['rank','brand', 'series', 'liquid', 'texture', 'color', 'price', 'name', 'lipstick_id'])
    for item in color:
        itemResult = getLipstickWithShortestNDistance(df, item, num)
        topLipsticksResult = pd.concat([itemResult, topLipsticksResult])
    
    topLipsticksResult = topLipsticksResult.drop_duplicates()
    topLipsticksResult = topLipsticksResult.sort_values(by='rank',ascending=False).head(num)
    topLipsticksResult = topLipsticksResult.drop(['rank'], axis=1)
    topLipsticksResult = topLipsticksResult.values
    for item in topLipsticksResult:
        topLipResult.append({'brand':item[0],'series':item[1],'name':item[2],'liquid':item[3],'texture':item[4],'color':item[5],'price':item[6],'lipstick_id':item[7]})
    print(topLipResult)    

    return topLipResult
    
def getRecommendation(client, userID, designated_num=5):
    cfResult = getRecommendationFromCfRecommendation(client, userID)
    cfcount = len(cfResult)
    print(cfcount)
    
    if cfcount == designated_num:
        return cfResult

    else:
        result = cfResult + getRecommendationFromTopLipsticks(client, userID, designated_num-cfcount, cfResult)
        return result
