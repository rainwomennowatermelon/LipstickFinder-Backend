#!/usr/bin/python3
import pymongo

schema = ["ID","email","password", "color","gender","kind","texture","name","path"]

def login(client, email, password):
    myquery = {"email":email}
    reply = client.db.Users.find(myquery)
    if (reply.count() == 1) and (password == reply[0]["password"]):
        return reply[0]["ID"], True
    else:
        print("login here")
        return "", False

def signup(client, email, password):
    myquery = {"email":email}
    reply = client.db.Users.find(myquery);
    if (reply.count() == 1):
        print("signup here")
        return "", False
    else:
        result = list(client.db.Users.find().sort("ID",-1).limit(1))
        print(result)
        result = result[0]
        maxvalue = result["ID"]
        mydict = {"ID":maxvalue+1, "email":email, "password":password}
        client.db.Users.insert_one(mydict)
        return login(client, email, password)

def answerquestions(client, email, password, gender, kind, texture, color):
    myquery = {"email":email}
    mydict = {"$set":{"gender":gender, "kind":kind, "texture": texture, "color": color}}
    reply = client.db.Users.find(myquery)
    if (reply.count() == 0):
        return False
    elif (reply.count() == 1) and (password == reply[0]["password"]):
        reply = client.db.Users.update_one(myquery, mydict)
        if(reply.modified_count == 1):
            return True
        else:
            return False
    else:
        return False

