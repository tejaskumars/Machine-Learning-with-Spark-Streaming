from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import json



sc=SparkContext.getOrCreate()
ssc=StreamingContext(sc,1)
spark=SparkSession(sc)


try:
	record=ssc.socketTextStream('localhost',6100)
except Exception as e:
	print(e)

def readstream(rdd):
	myDf=spark.read.json(rdd)
	myDf.show()
	
record.foreachRDD(lambda x:readstream(x))
		
	
	
ssc.start()             
ssc.awaitTermination()  
