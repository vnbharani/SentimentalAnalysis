import re,csv
import tweepy,datetime
import json,os
from tweepy import OAuthHandler


#Variables that contains the user credentials to access Twitter API
class TwitterDataGenerator():

    def connectTwitter(self):
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'twitterDevproperties.json')
        with open(filename) as data_file:
            data = json.load(data_file)
        consumerKey = data["consumer_key"]
        consumerSecretKey = data["consumer_secret"]
        accessToken = data["access_token"]
        accessTokenSecret = data["access_token_secret"]
        auth = OAuthHandler(consumerKey, consumerSecretKey)
        auth.set_access_token(accessToken, accessTokenSecret)
        # authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(consumerKey, consumerSecretKey)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)
        return api

    def getData(self,query):
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'tweets.csv')
        # Open/Create a file to append data
        csvFile = open(filename, 'w')
        #Use csv Writer
        csvWriter = csv.writer(csvFile)
        api = self.connectTwitter()

        for tweet in tweepy.Cursor(api.search,q=query,lang="en").items(40):
            print(datetime.datetime.now())
            difference =  datetime.datetime.utcnow() - tweet.created_at
            diff_hours = difference.seconds / 3600
            if diff_hours <1:
                print(tweet.created_at, tweet.text)
                csvWriter.writerow([ tweet.text.encode('utf-8')])
        #os.system("HADOOP_USER_NAME=hdfs hadoop fs -put tweets.csv /user/hadoop/input/")

