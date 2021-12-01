# Import general libraries
from PIL import Image
import numpy as np
import FINd_Edva as fn
from glob import glob
import pandas as pd
import random
import time
import FINd_CV2 as npfind
import json
import pickle

# Get Spark-related imports
import pyspark
import spark
from pyspark.sql.types import IntegerType, ArrayType, StringType
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf, col


def init_spark(app_name, master_config):
    """
    :params app_name: Name of the app
    :params master_config: eg. local[4]
    :returns SparkContext, SQLContext, SparkSession:
    """
    conf = (pyspark.SparkConf().setAppName(app_name).setMaster(master_config))

    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sql_ctx = SQLContext(sc)
    spark = SparkSession(sc)

    return (sc, sql_ctx, spark)


@udf(returnType=StringType())
def hash_udf(img):
	"""
	Spark UDF that returns the hash from a file location
	"""
	return findHasher.fromFile(img)


def get_images_df(n, to_array=False):
	"""
	Extracts images in a folder and returns the dataframe
	with n image files
	"""
	all_imgs = [x for x in glob('./das_images/*.jpg')]
	imgs = random.sample(all_imgs, n)

	df = pd.DataFrame(imgs) 
	df.columns = ['img_names']

	if to_array:
		df['imgs'] = df['img_names'].map(Image.open)
		df['imgs'] = df['imgs'].map(np.asarray)

	return df, imgs


def launch_spark(df, imgs, showpartitions=False, repartition=False, n_repartition=8, verbose=True, export=False):
	"""
	Launches spark and performs the computations on the performed dataframe.
	Partitions can be manually changed based on the data size. The wall-clock and
	CPU time are calculated.
	"""

	# Create spark dataframe
	df = spark.createDataFrame(df)

	# Show pratitions
	if showpartitions:
		print(df.rdd.getNumPartitions())

	# Repartition the dataframe
	if repartition:
		df = df.repartition(n_repartition, 'img_names')
	
	start_wallclock = time.time()
	start_cpu = time.process_time()

	# Get the hashes
	df = df.withColumn("hash_value", hash_udf(col("img_names")))
	df.collect()

	if export:
		df.toPandas().to_csv('values_hash.csv')

	end_wallclock = time.time()
	end_cpu = time.process_time()

	# calculate the times
	wallclock_time = end_wallclock - start_wallclock
	cpu_time = end_cpu - start_cpu

	if verbose:
		print("---- SPARK -----")
		print(f"----------------Time (in seconds): {wallclock_time}---------------------")
		print(f"----------------Time per image (in miliseconds): {((wallclock_time)/len(imgs)) * 1000}---------------------")
		print(f"----------------Time per image per core (in miliseconds): {(((wallclock_time)/len(imgs)) * no_cores) * 1000}---------------------")

	return wallclock_time, cpu_time

def check_correctness(spark):
	"""
	Check the correctness of the output hash in relation to the CV2 algorithm.
	The CV2 algorithm's output is used because Spark is running on that particular
	algorithm. An import from a pickle file is used.
	"""

	# Load the pickle data
	check_data = pickle.load(open( "find2_numpy_truth.pickle", "rb" ) )

	# Get images and hashes
	imgs = list(check_data.keys())
	hashes = list(check_data.values())

	# Get Spark Result
	df_check = pd.DataFrame(imgs)
	df_check.columns = ['img_to_check']
	df_check = spark.createDataFrame(df_check)
	df_check = df_check.withColumn("hash_value", hash_udf(col("img_to_check")))
	
	# Convert values to pandas dataframe
	df_check_pandas = df_check.toPandas()
	hash1 = df_check_pandas.iloc[0,1]
	hash2 = df_check_pandas.iloc[1,1]

	# Assert if outputs are identical
	assert hash1 == hashes[0], 'The first hash does not match'
	assert hash2 == hashes[1], 'The second hash does not match'
	print("The outputs match")

if __name__ == '__main__':

	times_wallclock = {}
	times_cpu = {}

	# Set parameters to test
	image_numbers_to_test = [12, 24, 50, 100, 200, 500, 1000]
	cores_to_test = [1,4,8,12]

	for no_cores in cores_to_test:
		
		# Instantiate dictionaries with the given cores
		times_wallclock[no_cores] = {}
		times_cpu[no_cores] = {}

		# Get test images
		for no_images in image_numbers_to_test:
			
			# Initiate spark
			sc, sql_ctx, spark = init_spark('images_process', f'local[{no_cores}]')

			# Calling DataFrame constructor on list 
			df, imgs = get_images_df(no_images)

			# Get findHasher so that the output is string
			findHasher = npfind.FINDHasher(True)		
			wallclock_time, cpu_time = launch_spark(df, imgs)

			times_wallclock[no_cores][no_images] = wallclock_time
			times_cpu[no_cores][no_images] = cpu_time

			# Check if the output is correct
			check_correctness(spark)

			# Stop the cluster with the exsiting context
			sc.stop()

	
	print("Spark CPU Times:")
	print(times_cpu)

	print("Spark Wallclock Times:")
	print(times_wallclock)

	with open('2021-01-17 spark_cpu_times.json', 'w') as fp:
		json.dump(times_wallclock, fp)


	with open('2021-01-17 spark_wallclock_times.json', 'w') as fp:
		json.dump(times_cpu, fp)