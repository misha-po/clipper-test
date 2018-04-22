from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from sys import exit

sc = SparkContext('local')
spark = SparkSession(sc)

#####################################################
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

train_data = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

test_data = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])


def TrainModel():
    # Configure an M3L pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to train_data documents.
    model = pipeline.fit(train_data)
    return model

def TestModel(model):
    # Prepare test_data documents, which are unlabeled (id, text) tuples.
    # Make predictions on test_data documents and print columns of interest.
    prediction = model.transform(test_data)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        rid, text, prob, prediction = row
        print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
    

import os.path
if __name__ == "__main__":
#    model_path = "/tmp/spark-logistic-regression-model"
    model_name = "test-model1"
    model_dir = "."
    model_path = os.path.join(model_dir, model_name)
    if (os.path.exists(model_path)):
        print("ERROR: %s already exists" % model_path)
        exit(1)
    model = TrainModel()
    TestModel(model)
    print('------------------------>Saving the model in %s' % model_path)
    model.save(model_path)
