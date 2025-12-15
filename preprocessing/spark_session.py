from pyspark.sql import SparkSession
import os

# Fix for Docker hostname resolution issue
# Spark tries to resolve the container hostname but fails in Docker
# Solution: explicitly set driver host to localhost
driver_host = os.environ.get("SPARK_DRIVER_HOST", "localhost")

# Initialize a single SparkSession for reuse across modules
spark = SparkSession.builder \
    .appName("CDLClassifier") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .config("spark.driver.host", driver_host) \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.master", "local[*]") \
    .getOrCreate()
