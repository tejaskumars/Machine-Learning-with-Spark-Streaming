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




sc=SparkContext.getOrCreate()
ssc=StreamingContext(sc,1)
spark=SparkSession(sc)

sc.setLogLevel('OFF')


try:
	record=ssc.socketTextStream('localhost',6100)
except Exception as e:
	print(e)
	
	
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
	'''lemmatizer = LemmatizerModel.pretrained('lemma_antbnc', 'en').setInputCols(["MeaningfulWords"]).setOutputCol("lemmatizedwords")
	df = lemmatizer.transform(df)
	'''
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
	
	
	return df
	 	

def readstream(rdd):
	df=spark.read.json(rdd)
	if(df.count() > 0):
		pdf = preprocess(df)
		pdf.show()
	
record.foreachRDD(lambda x:readstream(x))
		
	
	
ssc.start()             
ssc.awaitTermination()  
