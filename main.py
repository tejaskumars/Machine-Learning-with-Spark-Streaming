from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession
import json
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover 
import nltk 
from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



sc=SparkContext.getOrCreate()
ssc=StreamingContext(sc,1)
spark=SparkSession(sc)

sc.setLogLevel('OFF')


try:
	record=ssc.socketTextStream('localhost',6100)
except Exception as e:
	print(e)
	
	
	
def modelBuild(df):
	X = df.select('features')
	X_a = np.array(X).tolist().collect()
	y = df.select('Sentiment')
	y_a = np.array(y).tolist().collect()
	
	try:
		X_train,X_test,y_train,y_test=train_test_split(X_a,y_a,test_size=0.1,random_state=42)
	except:
		print(e)
	
	
	model1 = Perceptron().partial_fit(X_train,y_train,classes=[0,4]) #model1
	model2 = PassiveAggressiveClassifier().partial_fit(X_train,y_train,classes=[0,4]) #model2
	pred1 = model1.predict(X_test)
	pred2 = model2.predict(X_test)
	print("accuracy using Perceptron() is:",accuracy_score(pred1,y_test))
	print("accuracy using PassiveAggressiveClassifier() is:",accuracy_score(pred2,y_test))
	
	
	try:
		with open('model1.pkl','wb') as file:
			pickle.dump(model1,file)
		with open('model2.pkl','wb') as file:
			pickle.dump(model2,file)
	except:
		print(e)
		
'''
	
def testData(df):
	X = df.select('features')
	X_a = np.array(X).tolist().collect()
	y = df.select('Sentiment')
	y_a = np.array(y).tolist().collect()
	
	test_model1 = pickle.load(open("model1.pkl","rb"))
	test_pred1 = test_model1.predict(X_a)
	print("accuracy using Perceptron() is:",accuracy_score(test_pred1,y_test))
	
	test_model2 = pickle.load(open("model2.pkl","rb"))
	test_pred2 = test_model2.predict(X_a)
	print("accuracy using PassiveAggressiveClassifier()  is:",accuracy_score(test_pred2,y_test))
	
'''
	
	
	
	
	
	
def preprocess(df):
	df = df.withColumn("ppt", lower(df['Tweet'])) 	#lowering the case of strings
	df = df.withColumn("ppt", F.regexp_replace('ppt', r'http\S+', ' ')) 	#removing links
	df = df.withColumn("ppt", F.regexp_replace('ppt', '@\w+', ' ')) #removing usernames 
	df = df.withColumn("ppt", F.regexp_replace('ppt', '#\w+', ' ')) #removing all hashtags
	df = df.withColumn("ppt", F.regexp_replace('ppt', "\w+'\w+", ' ')) #removing contractions
	df = df.withColumn("ppt", F.regexp_replace('ppt', r'[^\w\s]' , ' ')) #removing other punctuations and special characters 
	df = df.withColumn("ppt", F.regexp_replace('ppt', r'[0-9]' , ' ')) #removing numbers
	df = df.withColumn("ppt", F.regexp_replace('ppt', r'\b\w{1,2}\b', ' ')) #removing words with length 1 or 2
	df = df.withColumn("ppt", trim(df.ppt)) #trimming the whitespaces in the start and end of the tweet
	df = df.withColumn("ppt", F.regexp_replace('ppt', r'\s\s+' , ' ')) #replacing two or more spaces with a single space
	
	
	
	
	#tokenization 
	tokenizer = Tokenizer(inputCol="ppt", outputCol="tokenized")
	df = tokenizer.transform(df)
	
	#removing stop words
	swr = StopWordsRemover(inputCol="tokenized", outputCol="MeaningfulWords")
	df = swr.transform(df)
	
	#lemmatization 
	lemmatizer = WordNetLemmatizer()
	def lem(x):
		listt = x["MeaningfulWords"]
		for i in listt:
			k = lemmatizer.lemmatize(i)
			ind = listt.index(i)
			listt[ind] = k
	df.foreach(lem)
	
	#hashingTF
	hashTF = HashingTF(inputCol="MeaningfulWords", outputCol="features")
	df = hashTF.transform(df).select('Sentiment', 'Tweet', 'features')
	
	
	modelBuild(df)  #for training the dataset
	
	#testdata(df)  #for testing the dataset
	 	

def readstream(rdd):
	df=spark.read.json(rdd)
	if(df.count() > 0):
		pdf = preprocess(df)
	
record.foreachRDD(lambda x:readstream(x))
		
	
	
ssc.start()             
ssc.awaitTermination()  
