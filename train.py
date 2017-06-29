#Authors: Bharani V, Sruthi A
#Generate Naive Bayes model for sentimental analysis
#Used TF-IDF model for features extraction
#Sentimen140 data for traning
#Data is available at http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes,  NaiveBayesModel
#from cleanData import TextPreProcessor
#from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

def removePunctuations(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def doProcess(rdd, toLowerCase=False):
    preProcessedRDD = rdd.map(removePunctuations)
    if toLowerCase:
        preProcessedRDD = preProcessedRDD.map(lambda l: l.strip().lower())
    return preProcessedRDD.filter(lambda l : len(l) > 0)

def replaceTwoOrMore(tweet):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", tweet)
    
def preProcess(rdd,toLowerCase = True):
    noPuncRDD = doProcess(rdd,toLowerCase)
    replaceCharRDD = noPuncRDD.map(replaceTwoOrMore)
    noSpaceRDD = spiltBySpace(replaceCharRDD)
    tokenizedRDD = removeStopWords(noSpaceRDD)
    return tokenizedRDD

def spiltBySpace(rdd):
    splitbySpace = rdd.map(lambda x : x.strip().split(" "))
    return splitbySpace
        
def removeStopWords(rdd):
    stopWordsFile = open("stopwords.txt", 'r')
    words = stopWordsFile.readlines()
    list_StopWord = []
    for w in words:
        if len(w.strip()) > 0:
            list_StopWord.append(w.strip())
    rddWithoutStopWords = rdd.map(lambda l: [w for w in l if w not in list_StopWord])
    return rddWithoutStopWords

def stemAndLemmatize(rdd):
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    return rdd.map(lambda line:[ lemmatiser.lemmatize(w) for w in line])


sc = SparkContext(appName = "processTwitterStream")


label_encoding={'\"0\"':0.0,'\"2\"':1.0,'\"4\"':2.0}

def transformData(input):
    data = input.split(",")
    label = data[0]
    feature = data[5]
    return (label,feature)

def getFeatures(data):
    hashingTF = HashingTF()
    tfData = data.map(lambda tup: hashingTF.transform(tup))
    idfData = IDF().fit(tfData)
    tfidfData = idfData.transform(tfData)
    return tfidfData

def createLabeledData(tfidfData,data):
    tfidfIdx = tfidfData.zipWithIndex()
    dataIdx = data.zipWithIndex()

    idxData = dataIdx.map(lambda line: (line[1], line[0]))
    idxTfidf = tfidfIdx.map(lambda line: (line[1], line[0]))

    joinedTfidf_Data = idxData.join(idxTfidf) 
    dataLabeled = joinedTfidf_Data.map(lambda tup: tup[1])
    return dataLabeled

def train_and_Model():
    trainFile = sc.textFile("/user/hadoop/trainData/training.1600000.processed.noemoticon.csv")
    trainData = trainFile.map(transformData)
    features = trainData.map(lambda x:x[1])
    #print(trainData.map(lambda x:x[0]).collect())
    label = trainData.map(lambda x:x[0]).distinct().zipWithIndex().collectAsMap()
    #print(label)
    preprocessedData = preProcess(features,True)
    #print(preprocessedData.collect())
    featuresData = getFeatures(preprocessedData)
    trainingLabeled = createLabeledData(featuresData,trainData)
    
    print(trainingLabeled.take(5))
    labeledTrainingData = trainingLabeled.map(lambda record: LabeledPoint(label[record[0][0]], record[1]))
    print(trainingLabeled.take(5))
    model = NaiveBayes.train(labeledTrainingData, 1.0)
    model.save(sc,"/user/hadoop/model/NaiveBayesModel")




def test_Model():
    model =NaiveBayesModel.load(sc,"finalproject/model/NaiveBayesModel")
    testFile = sc.textFile("testdata.manual.csv")
    testData = testFile.map(transformData)
    testData = testData.filter(lambda x: x[0]!='\"2\"')
    features = testData.map(lambda x:x[1])
    #print(testData.map(lambda x:x[0]).collect())
    label = testData.map(lambda x:x[0]).distinct().zipWithIndex().collectAsMap()
    print(label)
    preprocessedData = preProcess(features)
    featuresData = getFeatures(preprocessedData)
    print("train",featuresData.take(5))
    testingLabeled = createLabeledData(featuresData,testData)
    print("train", testingLabeled.take(5))
    labeledTestingData = testingLabeled.map(lambda record: LabeledPoint(label[record[0][0]], record[1]))


    predictionAndLabel = labeledTestingData.map(lambda p : (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labeledTestingData.count()
    print("Accuracy: ",accuracy)





#train_and_Model()
test_Model()

