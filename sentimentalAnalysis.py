# coding=utf8
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import os,sys,json
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes,  NaiveBayesModel
from createTwitterCSV import TwitterDataGenerator
import argparse,re
from operator import add
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()



class SentimentAnalysis():

    def removePunctuations(self,tweet):
        return " ".join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def doProcess(self,rdd, toLowerCase=False):
        preProcessedRDD = rdd.map(self.removePunctuations)
        if toLowerCase:
            preProcessedRDD = preProcessedRDD.map(lambda l: l.strip().lower())
        return preProcessedRDD.filter(lambda l : len(l) > 0)

    def replaceTwoOrMore(self,tweet):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", tweet)
    
    def preProcess(self,rdd,toLowerCase = True):
        noPuncRDD = self.doProcess(rdd,toLowerCase)
        replaceCharRDD = noPuncRDD.map(self.replaceTwoOrMore)
        noSpaceRDD = self.spiltBySpace(replaceCharRDD)
        tokenizedRDD = self.removeStopWords(noSpaceRDD)
        return tokenizedRDD

    def spiltBySpace(self,rdd):
        splitbySpace = rdd.map(lambda x : x.strip().split(" "))
        return splitbySpace

    def removeStopWords(self,rdd):
        stopWordsFile = open("finalproject/stopwords.txt", 'r')
        words = stopWordsFile.readlines()
        list_StopWord = []
        for w in words:
            if len(w.strip()) > 0:
                list_StopWord.append(w.strip())
        rddWithoutStopWords = rdd.map(lambda l: [w for w in l if w not in list_StopWord])
        return rddWithoutStopWords

    def processInputFile(self,rdd):
        preProcessedRDD = rdd.map(self.removePunctuations)
        preprocess_id = preProcessedRDD.zipWithIndex().map(lambda l:(l[1],l[0]))
        rddZipped = rdd.zipWithIndex().map(lambda l:(l[1],l[0]))
        o = rddZipped.join(preprocess_id)
        o_sorted = o.sortByKey().map(lambda l:l[1])
        return o_sorted.filter(lambda l:len(l[1])>0)

    def transformData(self,input):
        featureVector = input.strip().split(" ")
        return (featureVector)

    def calculateSentiment(self,sc,query):
        model = NaiveBayesModel.load(sc,"finalproject/model/NaiveBayesModel")
        query = query
        print (query)
        twitDG = TwitterDataGenerator()
        twitDG.getData(query)
        inputFile = sc.textFile("finalproject/tweets.csv").distinct()
        input_id = inputFile.zipWithIndex().map(lambda l:(l[1],l[0]))
        preprocessedData = self.preProcess(inputFile)
        inputFileProcessed = self.processInputFile(inputFile)
        print("#################################################################################################")
        print(preprocessedData.take(5))
        print("--------------------------------------------------------------------------------------------------")
        print(inputFileProcessed.take(5))
        print("input file processed ",inputFileProcessed.count())
        print("preprocessed count",preprocessedData.count())
        hashingTF = HashingTF()
        tfData = preprocessedData.map(lambda tup: hashingTF.transform(tup))
        idfData = IDF().fit(tfData)
        tfidfData = idfData.transform(tfData)
        output = tfidfData.map(lambda rec: model.predict(rec))
        i_I=inputFileProcessed.map(lambda l: l[0]).zipWithIndex().map(lambda l:(l[1],l[0]))
        print("input file count",inputFile.count())
        print ("output file count",output.count())
        o_I=output.zipWithIndex().map(lambda l:(l[1],l[0]))
        i_o =i_I.join(o_I).map(lambda l:l[1])
        print(i_o.take(i_o.count()))
        print(i_o.count())
        outputJson = {}
        tweetList = []
        tweet = {}
        positiveCount =0
        negativeCount =0
        for i in i_o.take(i_o.count()):
            print(i)
                #print data,data1
            if i[1] == 0.0:
                negativeCount = negativeCount+1
                text = "This is a negative Tweet"
            elif i[1] == 1.0:
                positiveCount = positiveCount + 1
                text = "This is a positive Tweet"
                    #data = text
            #replace(u"\u2022", "*").encode("utf-8")
            if len(i[0]) > 4:
                tweet = {}
                tweet['value'] = i[0].encode("ascii","ignore")
                tweet['sentiment'] = text
                tweetList.append(tweet)
                print i[0].encode("ascii","ignore")
                print text
                print "-------------------------------------"

                #print unicode(str(data),"utf-8")
        print (positiveCount)
        print (negativeCount)
        outputJson["tweets"] = json.dumps(tweetList)
        outputJson["positiveTweetCount"] = positiveCount
        outputJson["negativeTweetCount"] = negativeCount
        wordflatMap = preprocessedData.flatMap(lambda xs: [x for x in xs]).map(lambda x:x.encode("ascii","ignore")).map(lambda x: (x, 1)).reduceByKey(add)
        wordFlatMap_reversed = wordflatMap.map(lambda l:(l[1],l[0])).filter(lambda l: (l[1]!="rt" and l[1]!=query))
        wordFlatMap_sorted = wordFlatMap_reversed.sortByKey(False)
        print (wordFlatMap_sorted.take(10))
        outputFrequencyList = {}
        mostFrequentWordList = []
        wordCount = {}
        words =[]
        counts = []
        for i in wordFlatMap_sorted.take(10):
            wordCount = {}
            wordCount['word'] = i[1]
            wordCount['count'] = i[0]
            mostFrequentWordList.append(wordCount)
        outputJson["frequency"] = json.dumps(mostFrequentWordList)
        return outputJson

