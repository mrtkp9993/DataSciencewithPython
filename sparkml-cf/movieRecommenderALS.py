from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

if __name__ == '__main__':
    spark = SparkSession.builder.appName("MovieRecs").getOrCreate()

    locale = spark.sparkContext._jvm.java.util.Locale
    locale.setDefault(locale.forLanguageTag("en-US"))
    
    lines = spark.read.text("ml-latest-small/ratings.csv").rdd
    parts = lines.map(lambda row: row.value.split(","))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]) ,movieId=int(p[1]), rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(training)
    
    predictions = model.transform(test)
    predictions = predictions.dropna()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error: {}".format(rmse))

    # Predict movies for User 66
    print("\nTop 20 recommendations:")
    # Find movies rated more than 100 times
    ratingCounts = ratings.groupBy("movieId").count().filter("count > 100")
    # Construct a "test" dataframe for user 66 with every movie rated more than 100 times
    popularMovies = ratingCounts.select("movieId").withColumn('userId', lit(66))

    # Run our model on that list of popular movies for user ID 0
    recommendations = model.transform(popularMovies)

    # Get the top 20 movies with the highest predicted rating for this user
    topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in topRecommendations:
        print(recommendation)
    
    spark.stop()
