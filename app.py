#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.model import BiSeNet
import src.usersManagement as UM
import src.lipstickRecommendation as LR
from utils.colorMethods import ColourDistance
from utils.colorMethods import HexadecimalColor2RGB

import torch
import torchvision.transforms as transforms

from flask import Flask, request, Response, send_file, jsonify
from flask_pymongo import PyMongo

import numpy as np
from PIL import Image
import cv2
import math
import time
import json
import io
import os


# Lip makeup function works in the HSV color space
def LipMakeUp(image, parsing, color):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    not12and13 = (parsing != 12) & (parsing != 13)
    changed[not12and13] = image[not12and13]
    return changed


# Load into memory
# Load model
n_classes = 19
net = BiSeNet(n_classes=n_classes)
save_pth = './res/cp/79999_iter.pth'
net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
net.eval()

# Tensor
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Other para
predictDataDir = './res/data/face-parsing-predictImgDir/'
makeupDataDir = './res/data/face-parsing-makeupImgDir/'

# Flask with MongoDB connected, you can access collection with mongo.db.collection_name.find(query)
app = Flask(__name__)
app.config["MONGO_URI"] = 'mongodb://%s:%s@localhost:27017/%s?authMechanism=SCRAM-SHA-1' % (
    os.environ.get('LIPSTICKFINDER_USER'), os.environ.get('LIPSTICKFINDER_PSW'), os.environ.get('LIPSTICKFINDER_DATABSE')
)
mongo = PyMongo(app)

# Add RGB to lipsticks
lipsticks_db = list(mongo.db.lipsticks.find({}, {'_id': 0, 'link': 0}))
for lipstick in lipsticks_db:
    lipstick['RGB'] = HexadecimalColor2RGB(lipstick['color'])


# Triggers
# Return last _id and total number to user to check database lipstick updation
@app.route('/checkLipstickUpdate', methods=['GET'])
def checkLipstickUpdate():
    return jsonify('%s-%d' % ( # _id-count
        str(list(mongo.db.lipsticks.find().sort([('_id', -1)]).limit(1))[0]['_id']),
        mongo.db.lipsticks.estimated_document_count()
    ))

# Lipstick recognition
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Save image
        image = request.files['file']
        fileName = str(time.time())
        filePath = predictDataDir+fileName+'.jpg'
        image.save(filePath)
        # Exact rgb value
        with torch.no_grad():
            image = Image.open(filePath).convert('RGB')
            imgArray = np.array(image)
            image = to_tensor(image)
            image = torch.unsqueeze(image, 0)
            out = net(image)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            upperLipPos = np.where(parsing == 12)
            lowerLipPos = np.where(parsing == 13)
            if len(upperLipPos[0]) > 0 or len(lowerLipPos[0]) > 0:
                pointsCount = len(upperLipPos[0])+len(lowerLipPos[0])
                rgb = [0, 0, 0]
                if len(upperLipPos) > 0:
                    rgbUpper = np.sum(imgArray[upperLipPos[0], upperLipPos[1], :], axis=0)
                    rgb = np.sum([rgb, rgbUpper], axis=0)
                if len(lowerLipPos) > 0:
                    rgbLower = np.sum(imgArray[lowerLipPos[0], lowerLipPos[1], :], axis=0)
                    rgb = np.sum([rgb, rgbLower], axis=0)
                res = [math.floor(i/pointsCount) for i in rgb]
                # Calculate the distance between extracted RGB and all lipsticks' RGB
                for lipstick in lipsticks_db:
                    lipstick['distance'] = ColourDistance(res, lipstick['RGB'])
                result = sorted(lipsticks_db, key=lambda x: x['distance'])[:3]
                return jsonify(result)
            else: return jsonify([])

# Lip makeup
@app.route('/makeup', methods=['POST'])
def makeup():
    if request.method == 'POST':
        # Save image
        image = request.files['file']
        predictFileName = str(time.time())
        predictFilePath = makeupDataDir + predictFileName + '.jpg'
        image.save(predictFilePath)
        # Get RGB
        hexadecimalColor = request.values.get("color")
        rgb = HexadecimalColor2RGB(hexadecimalColor)
        # Make up lip
        with torch.no_grad():
            image = Image.open(predictFilePath)
            image = to_tensor(image)
            image = torch.unsqueeze(image, 0)
            out = net(image)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            upperLipPos = np.where(parsing == 12)
            lowerLipPos = np.where(parsing == 13)
            if len(upperLipPos[0]) > 0 or len(lowerLipPos[0]) > 0:
                bgr = list(reversed(rgb))
                ori = cv2.imread(predictFilePath)
                ori = LipMakeUp(ori, parsing, bgr)
                # Save img
                makeupFileName = str(time.time())
                makeupFilePath = makeupDataDir + makeupFileName + '.jpg'
                cv2.imwrite(makeupFilePath, ori)
                with open(makeupFilePath, 'rb') as bites:
                    return send_file(
                        io.BytesIO(bites.read()),
                        attachment_filename='makeup.jpg',
                        mimetype='image/jpg'
                    )
            else:
                with open(filePath, 'rb') as bites:
                    return send_file(
                        io.BytesIO(bites.read()),
                        attachment_filename='nolip.jpg',
                        mimetype='image/jpg'
                    )

# Return object used for recommendation
class recommendLipstickInfoVos:
    def __init__(self):
        self.recommendLipstickInfoVos = []

# Get recommend lipsticks     
@app.route('/getRecommendLipsticks', methods=['POST'])
def getRecommendLipsticks():
    lipsticks = recommendLipstickInfoVos()
    data = request.get_data(parse_form_data=False)
    data = json.loads(data, encoding='utf-8')
    userID = int(data["userID"])
    password = data["password"]
    if mongo.db.Users.count_documents({'ID': userID,'password':password})==1:
        for lipstick in mongo.db.lipsticks.find({}, {'_id': 0}).limit(2):
            lipstick['like'] = False
            lipsticks.recommendLipstickInfoVos.append(lipstick)
    return jsonify(lipsticks.__dict__)

# Get lipsticks' detailed information
@app.route('/getRecommendLipstickInfo', methods=['POST'])
def getRecommendLipstickInfo():
    lipsticks = recommendLipstickInfoVos()
    data = request.get_data(parse_form_data=False)
    data = json.loads(data, encoding='utf-8')
    userID = int(data["userID"])
    password = data["password"]
    if mongo.db.Users.count_documents({'ID': userID,'password':password})==1:
        likelist = []
        if mongo.db.likes.count_documents({'userID': userID})==1:
            reply = mongo.db.likes.find({'userID': userID})[0]
            likelist = reply["lipstickID"]
        for lipstick in LR.getRecommendation(mongo, userID):
            lipstick['like'] = (lipstick["lipstick_id"] in likelist)
            lipsticks.recommendLipstickInfoVos.append(lipstick)
    return jsonify(lipsticks.__dict__)

# Get lipstick brand and brand_id
@app.route('/lipstick-brand', methods=['GET'])
def getLipstickBrand():
    if request.method == 'GET':
        return jsonify(list(mongo.db.mapping.find(
            {'type': 'brand'}, 
            {'name': 1, 'id': 1, '_id': 0}
        )))

# Given brand_id, return corresponding series and series_id
@app.route('/lipstick-series', methods=['GET'])
def getLipstickSeries():
    if request.method == 'GET':
        brand_id = int(request.args.get("brand_id"))
        return jsonify(list(mongo.db.mapping.find(
            {'brand_id': brand_id, 'type': 'series'}, 
            {'name': 1, 'id': 1, '_id': 0}
        )))

# Given brand_id and series_id, return corresponding lipsticks
@app.route('/lipsticks', methods=['GET'])
def getLipsticks():
    if request.method == 'GET':
        brand_id = int(request.args.get("brand_id"))
        series_id = int(request.args.get("series_id"))
        lipsticks = []
        for row in mongo.db.lipsticks.find(
            {'brand_id': brand_id, 'series_id': series_id},
            {'_id': 0, 'brand': 1, 'series': 1, 'liquid': 1, 'texture': 1, 'color': 1, 'price': 1, 'name': 1, 'lipstick_id': 1}
        ):
            row['color'] = row['color'].replace(' ', '')
            lipsticks.append(row)
        return jsonify(lipsticks)

# Given userID and profile image, save image to file system and image path to MongoDB
@app.route('/updateProfileImage', methods=['POST'])
def updateProfileImage(root = "./res/data/profiles/"):
    if request.method == 'POST':
        userID = int(request.values.get("userID"))
        pwd = request.values.get("pwd")
        image = request.files['file']      
        password = mongo.db.Users.find({'ID': userID}, {'_id': 0, 'password': 1})[0]
        if (pwd==password['password']): 
            path = os.path.join(root, '%d.%s' % (userID, image.content_type.replace('image/', '')))
            image.save(path)
            if os.path.exists(path):
                result = mongo.db.Users.update_one(
                    {'ID': userID}, {"$set": { "path": path}}
                )
                if result.modified_count == 1:
                    return Response("OK", status=200)
                return Response("Fail to update database with profileImage_path", status=500)
            else:
                return Response("Fail to save image", status=500)
        else:
            return Response("Wrong Password", status=500)

# Given userID and password, save user info to MongoDB
@app.route('/updateProfileInfo', methods=['POST'])
def updateProfileInfo(root = "./res/data/profiles"):
    data = request.get_json()
    userID = data["userID"]
    pwd = data["pwd"]
    name = data["name"]
    gender = data["gender"]
    password = mongo.db.Users.find({'ID': userID}, {'_id': 0, 'password': 1})[0]
    if (pwd==password['password']):
        result = mongo.db.Users.update_one(
            {'ID': userID}, {"$set": {"name": name, "gender": gender}}
        )
        if result.modified_count == 1:
            return Response("OK", status=200)
        return Response("Fail to update database with name and gender", status=500)
    
# Given userID and password, return user information
@app.route('/getUserInfo', methods=['GET'])
def getUserInfo():
    if request.method == 'GET':
        userID = int(request.args.get("userID"))
        pwd = request.args.get("pwd")
        password = mongo.db.Users.find({'ID': userID}, {'_id': 0, 'password': 1})[0]
        if (pwd==password['password']):
            record = list(mongo.db.Users.find({'ID': userID}, {'_id': 0, 'path': 0}))[0]
            return jsonify(record)
        else:
            return Response("Wrong Password", status=500)

# Given userID and password, return user image
@app.route('/getUserProfileImage', methods=['GET'])
def getUserProfileImage():
    if request.method == 'GET':
        userID = int(request.args.get("userID"))
        pwd = request.args.get("pwd")
        user = mongo.db.Users.find_one({'ID': userID,'password':pwd})
        if user is None:
            return Response("Wrong Password", status=500)
        else:
            if 'path' not in user or user['path'] is None:
                print("No image")
                return Response("No image", status=500)
            else:
                if os.path.exists(user['path']):
                    with open(user['path'], 'rb') as bites:
                        return send_file(
                            io.BytesIO(bites.read()),
                            attachment_filename='profile.jpg',
                            mimetype='image/jpg'
                        )
                else:
                    return Response("No image", status=500)

# Given userID and password, return user like list (full lipstick info)
@app.route('/getLipstickLike', methods=['GET'])
def getLipstickLike():
    if request.method == 'GET':
        userID = int(request.args.get("userID"))
        pwd = request.args.get("pwd")
        user = mongo.db.Users.find({'ID': userID, 'password':pwd})
        if user is None:
            return Response("Wrong Password", status=500)
        else: 
            if mongo.db.likes.count_documents({'userID': userID})==1:
                reply = mongo.db.likes.find({'userID': userID}, {'_id': 0})[0]
                lipsticks = []
                for lipstickID in reply['lipstickID']:
                    lipstickInfo = list(mongo.db.lipsticks.find({'lipstick_id': lipstickID},{'_id': 0, 'brand': 1, 'series': 1, 'liquid': 1, 'texture': 1, 'color': 1, 'price': 1, 'name': 1, 'lipstick_id': 1}))[0]
                    lipsticks.append(lipstickInfo)
                return jsonify(lipsticks)
            else:
                lipsticks_example = [{'brand': 'Here is a example', 'series': 'Storge', 'name': ' Lipstick You Like', 'liquid': True, 'texture': 'Texture', 'color': '#CA7476', 'price': 280, 'lipstick_id': '00000'},]
                return jsonify(lipsticks_example)
    
# Marked as unlike. Given userID, pwd and specific lipstickID, remove this lipstickID from like-lipstick-list
@app.route('/markIfLike', methods=['POST'])
def markIfLike():
    if request.method == 'POST':
        data = request.get_json()
        userID = int(data["userID"])
        pwd = data["pwd"]
        likeLipstickID = data["likeLipstickID"] #string
        likeLabel = data["likeLabel"]
        user = mongo.db.Users.find({'ID': userID, 'password':pwd})
        print("here")
        if user is None:
            print("Wa")
            return Response("Wrong Password", status=500)
        else: 
            if mongo.db.likes.count_documents({'userID': userID}) is 1:
                reply = mongo.db.likes.find({'userID': userID})[0]
                newLikeLipsticks = reply['lipstickID']
                # labeled as like
                if likeLabel is True: 
                    # add this new lipstick in like list
                    if likeLipstickID not in newLikeLipsticks: 
                        newLikeLipsticks.append(likeLipstickID)
                    print(newLikeLipsticks)
                    # update rating
                    if mongo.db.rating.count_documents({"userID": userID, 'lipstickID': likeLipstickID}) is 1:
                        mongo.db.rating.update_one({'userID': userID, 'lipstickID': likeLipstickID}, {"$set": {"rating": 10}})
                    else:
                        mongo.db.rating.insert_one({'userID': userID, 'lipstickID': likeLipstickID, 'rating': 10})
                # labeled as unlike
                else: 
                    # remove this lipstick in like list
                    newLikeLipsticks.remove(likeLipstickID)
                    # update rating
                    mongo.db.rating.update_one({'userID': userID, 'lipstickID': likeLipstickID}, {"$set": {"rating": 0}})
                mongo.db.likes.update_one({'userID': userID}, {"$set": {"lipstickID": newLikeLipsticks}})
                print(mongo.db.rating.find({'userID': userID, 'lipstickID': likeLipstickID})[0])
            else:
                newLikeLipsticks = []
                # add this new lipstick in like list
                newLikeLipsticks.append(likeLipstickID)
                # update rating
                mongo.db.rating.insert_one({'userID': userID, 'lipstickID': likeLipstickID, 'rating': 10})
                
                mongo.db.likes.insert_one({'userID': userID, "lipstickID": newLikeLipsticks})
                
            return Response("OK")
            

# Given userID, pwd and specific lipstickID, rating+1
@app.route('/focusOnLipstick', methods=['POST'])
def focusOnLipstick():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        userID = int(data["userID"])
        pwd = data["pwd"]
        lipstickID = data["lipstickID"] #string
        print(lipstickID)
        user = mongo.db.Users.find({'ID': userID, 'password':pwd})
        if user is None:
            return Response("Wrong Password", status=500)
        else: 
            if mongo.db.rating.count_documents({"userID": userID, 'lipstickID': lipstickID}) is 1:
                reply = mongo.db.rating.find({'userID': userID, 'lipstickID': lipstickID})[0]
                if reply['rating']<10:
                    newrating = reply['rating']+1
                    mongo.db.rating.update_one({'userID': userID, 'lipstickID': lipstickID}, {"$set": {"rating": newrating}})
                    print(mongo.db.rating.find({'userID': userID, 'lipstickID': lipstickID})[0])
            else:
                mongo.db.rating.insert_one({'userID': userID, 'lipstickID': lipstickID, 'rating': 1})
                print(mongo.db.rating.find({'userID': userID, 'lipstickID': lipstickID})[0])
            return Response("OK")   
                

# Reset Password
@app.route('/resetPwd', methods=['POST'])
def resetPwd():
    data = request.get_json()
    userID = int(data["userID"])
    pwd = data["pwd"]
    newPwd = data["newPwd"]
    password = mongo.db.Users.find({'ID': userID}, {'_id': 0, 'password': 1})[0]
    if (pwd==password['password']):
        result = mongo.db.Users.update_one(
            {'ID': userID}, {"$set": {"password": newPwd}}
        )
    return Response("OK")


# Login Check the Email and Password
@app.route('/checkLoginInfo', methods=['POST'])
def checkLoginInfo():
    data = request.get_json()
    email = data["email"]
    password = data["password"]
    res = {"email":email, "password":password}
    uid, flag =  UM.login(mongo, email, password);
    if flag:
      res["uid"] = uid
      res["result"] = "True"
    else:
      res["result"] = "False"
    return jsonify(res)

# Login Check the Email and Password
@app.route('/signUp', methods=['POST'])
def signUp():
    data = request.get_json()
    email = data["email"]
    password = data["password"]
    res = {"email":email, "password":password}
    uid, flag =  UM.signup(mongo, email, password);
    if flag:
      res["uid"] = uid
      res["result"] = "True"
    else:
      res["result"] = "False"   
    return jsonify(res)

# Upload questionaire
@app.route('/uploadQuestionnaire', methods=['POST'])
def uploadQuestionnaire():
    data = request.get_json()
    email = data["email"]
    password = data["password"]
    gender = data["gender"]
    kind = data["kind"]
    texture = data["texture"]
    color = data["color"]
    res = {"email":email, "password":password}
    if UM.answerquestions(mongo, email, password, gender, kind, texture, color):
      res["result"] = "True"
    else:
      res["result"] = "False" 
    return jsonify(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
