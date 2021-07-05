##############################
import os
import numpy
import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()

#######data loading and preprocessing

'''
combine the training and test data
'''

train = sqlContext.read.option('header', True).csv('train.csv')
train.registerTempTable('train')

test = sqlContext.read.option('header', True).csv('test.csv')
test.registerTempTable('test')

train.write.mode("Overwrite").json("gts_train.json")
test.write.mode("Overwrite").json("gts_test.json")

sqlContext.read.json("gts_*.json").registerTempTable("gts")

'''
process the null value 
and transform the data type of numumrical features
'''

gts = sqlContext.sql(u"""
	select *,
	case when city_score is not null then city_score else 'null %' end as city_score_1,
	double(studied_credits) as studied_credits1,
	case when prelim_score is not null then double(prelim_score) else 0.0 end as prelim_score1,
	double(online_clicks) as online_clicks1,
	int(full_term_subscription) as label
	from gts
	""").drop('studied_credits')\
	.drop('prelim_score')\
	.drop('online_clicks')

gts.registerTempTable("gts")

#########building the feature transformation pipeline

feature_stages = []
feature_vec_columns = []

'''
building the string indexing and
one hot stages for the categorical 
features, to vectorize the categorical 
features
'''

for categorial_features in [
	"gender",
	"region",
	"highest_qualification",
	"city_score_1",
	"age",
	"disability",
	]:
	feature_stages.append(
		StringIndexer(
		inputCol = categorial_features, 
		outputCol = "%s_idx"%(categorial_features))
		)
	feature_stages.append(
		OneHotEncoder(
		inputCol = "%s_idx"%(categorial_features), 
		outputCol = "%s_vec"%(categorial_features))
		)
	feature_vec_columns.append("%s_vec"%(categorial_features))

'''
add the numumrical featres
'''

feature_vec_columns+=[
"studied_credits1",
"prelim_score1",
"online_clicks1",
]

'''
combine the categorical and numumrical features to a longer feature vector
'''

feature_stages.append(
	VectorAssembler(
    	inputCols = feature_vec_columns,
    	outputCol="original_features")
    )

'''
standardizing  the features
'''

feature_stages.append(
	StandardScaler(
	inputCol="original_features", 
	outputCol="features",
    	withStd=True, 
    	withMean=False)
	)

##########transforming the features 

pipeline = Pipeline(stages=feature_stages)

model = pipeline.fit(gts)
features = model.transform(gts)
features.registerTempTable("features")

features.write.mode("Overwrite").parquet("features")

sqlContext.read.parquet("features").registerTempTable("features")

sqlContext.sql(u"""
	SELECT id, features, label 
	FROM features 
	""").show()

############building the machine learning pipeline

model_stages = [
	GBTClassifier(
	labelCol="label", 
	featuresCol="features", 
	maxIter = 100)
]

pipeline_model = Pipeline(
	stages = model_stages)

##########training the pipeline

training = sqlContext.sql(u"""
	SELECT id, features, label 
	FROM features WHERE label IS NOT NULL
	""")

pipeline_model_trained = pipeline_model.fit(training)

'''
show the feature importance
'''

pipeline_model_trained.stages[0].featureImportances

'''
transforming the training data
'''

training_prediction = pipeline_model_trained.transform(training)

training_prediction.registerTempTable("training_prediction")

sqlContext.sql(u"""
	SELECT label, prediction, count(*)
	FROM training_prediction
	GROUP BY label, prediction
	""").show()

'''
+-----+----------+--------+
|label|prediction|count(1)|
+-----+----------+--------+
|    1|       0.0|      31|
|    0|       0.0|    1096|
|    1|       1.0|    1240|
|    0|       1.0|      83|
+-----+----------+--------+
'''

##########prediction over the test data

test = sqlContext.sql(u"""
	SELECT id, features 
	FROM features WHERE label IS NULL
	""")

test_prediction = pipeline_model_trained.transform(test)

test_prediction.registerTempTable("test_prediction")

submissions = sqlContext.sql(u"""
	SELECT id, int(prediction) as full_term_subscription
	FROM test_prediction
	""")

submissions.show()

'''
saving the results to csv
'''

submissions.toPandas().to_csv(
	"submissions.csv",
	index = False)

##############################
