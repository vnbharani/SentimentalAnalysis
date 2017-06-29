#!flask/bin/python
from flask import Flask, jsonify
from finalproject.sentimentalAnalysis import SentimentAnalysis
from flask_cors import CORS, cross_origin
from pyspark import SparkConf, SparkContext
import json
from flask import g


app = Flask(__name__)
cors = CORS(app, resources={r"/sentimentanalysis/api/*": {"origins": "*"}})

 

@app.route('/sentimentanalysis/api/v1.0/get/<string:query>', methods=['GET'])
def get_tasks(query):
    global sentimentanalysis
    #conf = SparkConf().setAppName("SentimentAnalysis")
       # IMPORTANT: pass aditional Python modules to each worker
    #sc = SparkContext(conf=conf, pyFiles=['finalproject/sentimentalAnalysis.py'])
    sc = getattr(g, '_sc', None)
    if sc is None:
        sc = g._sc = SparkContext()
    sentimentanalysis = SentimentAnalysis()
    result = sentimentanalysis.calculateSentiment(sc,query)
    return json.dumps(result)

@app.teardown_appcontext
def teardown_sparkcontext(exception):
    sc = getattr(g, '_sc', None)
    if sc is not None:
        sc.stop()
        
if __name__ == '__main__':
    app.run(debug=True)
    
