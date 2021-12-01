import subprocess
import time
import json
from glob import glob
import random
from FINd import FINDHasher

# Set variables
filename = 'das_images/103*.jpg'
scriptname = 'FINd_hash_size.py'

def calculate_image_times(jpeg_sizes, hash_size = 16, scriptname=scriptname, filename=filename):
	"""
	Calculates the time it takes to run a file with a specified
	height and width.
	"""
	times_for_items = {}

	for image_size in jpeg_sizes:
		start = time.time()
		subprocess.run(f'python {scriptname} {hash_size} {image_size} {filename}', shell=True)
		end = time.time()
		ttime = end-start
		times_for_items[image_size] = ttime

	return times_for_items

def calculate_hash_times(hash_sizes, image_size=512, scriptname=scriptname, filename=filename):
	"""
	Calculates the time it takes to run a file with a specified
	hash output
	"""
	times_for_items = {}

	for hash_size in hash_sizes:
		start = time.time()
		subprocess.run(f'python {scriptname} {hash_size} {image_size} {filename}', shell=True)
		end = time.time()
		ttime = end-start
		times_for_items[hash_size] = ttime

	return times_for_items

def sample_images(n=100):
	"""
	Sample random n images from the whole set of the images
	"""
	all_images = [x for x in glob('./das_images/*.jpg')]
	images = random.sample(all_images, n)
	return images

def get_hashes(images, FINDHasher=FINDHasher):
	"""
	Get the hashes of a list of images
	"""
	hashes = []
	findHasher = FINDHasher()
	for i, im in enumerate(images):
		hash_ = findHasher.fromFile(im)
		hashes.append(hash_)
	return hashes

def get_image_times(no_images):
	"""
	Gets image times based on the number of images provided
	"""
	times_dict = {}

	for im_nr in no_images:
		images = sample_images(im_nr)
		start = time.time()
		hashes = get_hashes(images)
		end = time.time()
		ttime = end-start
		times_dict[im_nr] = ttime
	
	return times_dict


if __name__ == '__main__':

	# Set the hash parameters
	hash_sizes = [2, 4, 6, 8, 10, 12, 15]
	jpeg_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
	no_images = [1,2,5,10,15,20]

	# Get the times for JPGs
	jpeg_times_json = calculate_image_times(jpeg_sizes)
	with open('times_jpg_sizes.json', 'w') as hs:
			json.dump(jpeg_times_json, hs)

	# Get the times for the hashes
	hash_times_json = calculate_hash_times(hash_sizes)
	with open('times_hash_sizes.json', 'w') as hs:
		json.dump(hash_times_json, hs)

	#Get the times for the hashes
	times_multiple_images = get_image_times(no_images)
	with open('times_multiple_images.json', 'w') as hs:
		json.dump(times_multiple_images, hs)