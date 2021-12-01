#!/usr/bin/env python
import math
from PIL import Image
import concurrent.futures
from imagehash import ImageHash
import numpy as np
from timeit import default_timer as timer
import time
import pandas as pd
import json
from multiprocessing.pool import Pool
import subprocess
from glob import glob
import FINd_CV2 as npfind
import sys
import pysparkex


def multiprocessing_imgs(findHasher, no_cores, imgs):
	"""
	Multiprocessing images based on the number of cores and the hashing algorithm.
	"""
	start_wallclock = time.time()
	start_cpu = time.process_time()		
	
	print(f"IMGS: {imgs}, No cores: {no_cores}")
	# Multiprocess that dataframe
	with concurrent.futures.ProcessPoolExecutor(no_cores) as executor:
		results=executor.map(findHasher.fromFile, imgs)

	# End time
	end_wallclock = time.time()
	end_cpu = time.process_time()

	# calculate the times
	wallclock_time = end_wallclock - start_wallclock
	cpu_time = end_cpu - start_cpu

	return wallclock_time, cpu_time


def main(image_numbers_to_test = [12, 24, 50, 100, 200, 500, 1000],
		cores_to_test = [1,2,4,8,12]):
	"""
	Runs the hashing algorithm with different number of processors and images.
	Done in order to benchmark with the performance of Spark.
	"""

	# Initiate the findHasher (Numpy version)
	findHasher= npfind.FINDHasher()

	# Number of cores and images to test (equal to the Spark version)

	# Instantiate dictionaries
	times_wallclock = {}
	times_cpu = {}

	for no_cores in cores_to_test:

		# Instantiate dictionaries with the given cores
		times_wallclock[no_cores] = {}
		times_cpu[no_cores] = {}

		for no_images in image_numbers_to_test:

			# Extract the dataframe
			_, imgs = pysparkex.get_images_df(no_images)

			# Calculate the processing
			wallclock_time, cpu_time = multiprocessing_imgs(findHasher, no_cores, imgs)
			
			times_wallclock[no_cores][no_images] = wallclock_time
			times_cpu[no_cores][no_images] = cpu_time
		
	# Print to the console
	print("Wallclock time: ")
	print(times_wallclock)

	print("CPU Time: ")
	print(times_cpu)

	with open('2021-01-17 multiprocessing_times_wallclock.json', 'w') as fp:
		json.dump(times_wallclock, fp)
	
	with open('2021-01-17 multiprocessing_times_cpu.json', 'w') as fp:
		json.dump(times_cpu, fp)

if __name__ == "__main__":
	main()