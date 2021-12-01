#!/usr/bin/env python

import math
from PIL import Image

from matrix import MatrixUtil #Ensure that matrix.py is in the same directory as this file!
from imagehash import ImageHash
import numpy as np
import cv2

class FINDHasher:

	#  From Wikipedia: standard RGB to luminance (the 'Y' in 'YUV').
	LUMA_FROM_R_COEFF = float(0.299)
	LUMA_FROM_G_COEFF = float(0.587)
	LUMA_FROM_B_COEFF = float(0.114)

	#  Since FINd uses 64x64 blocks, 1/64th of the image height/width
	#  respectively is a full block.
	FIND_WINDOW_SIZE_DIVISOR = 64

	def compute_dct_matrix(self):
		matrix_scale_factor = math.sqrt(2.0 / 64.0)
		d = [0] * 16
		for i in range(0, 16):
			di = [0] * 64
			for j in range(0, 64):
				di[j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))
			d[i] = di
		return d

	def __init__(self, hash_as_string=False):
		"""See also comments on dct64To16. Input is (0..63)x(0..63); output is
		(1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
		Returns 16x64 matrix."""
		self.DCT_matrix = self.compute_dct_matrix()
		self.hash_as_string = hash_as_string

	def fromFile(self, filepath):
		img = None
		try:

			img = cv2.imread(filepath)
			# Reshape the image
			if img.shape[0] > 512:
				img = cv2.resize(img, (512,512))

			# CV2 iS BGR, PIL is RGB, so convert to RGB
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except IOError as e:
			raise e
		return self.fromImage(img)

	def fromImage(self,img):
		numCols, numRows, _ = img.shape
		#numCols, numRows = img.size
		buffer1 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer2 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer64x64 = MatrixUtil.allocateMatrix(64, 64)
		buffer16x64 = MatrixUtil.allocateMatrix(16, 64)
		buffer16x16 = MatrixUtil.allocateMatrix(16, 16)
		self.fillFloatLumaFromBufferImage_CV(img, buffer1)
		return self.findHash256FromFloatLuma(
			buffer1, buffer2, numRows, numCols, buffer64x64, buffer16x64, buffer16x16
		)

		# Dot product implementation offers worse performance
	def fillFloatLumaFromBufferImage(self, img, luma):
		numCols, numRows, _ = img.shape
		#numCols, numRows = img.size
		ratios = np.array([0.299, 0.587, 0.114])
		luma[:] = np.dot(np.array(img).reshape(-1,3), ratios)
		print(len(luma))


	def fillFloatLumaFromBufferImage_CV(self, img, luma):
		numCols, numRows, _ = img.shape
		luma[:] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(numCols*numRows,)
	

	def findHash256FromFloatLuma(
		self,
		fullBuffer1,
		fullBuffer2,
		numRows,
		numCols,
		buffer64x64,
		buffer16x64,
		buffer16x16,
	):
		windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
		windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)
		
		self.boxFilter2(fullBuffer1,fullBuffer2,numRows,numCols,windowSizeAlongRows,windowSizeAlongCols)
		
		self.decimateFloat(fullBuffer2, numRows, numCols, buffer64x64)
		self.dct64To16(buffer64x64, buffer16x64, buffer16x16)
		hash = self.dctOutput2hash(buffer16x16)
		return hash

	@classmethod
	def decimateFloat(
		cls, in_, inNumRows, inNumCols, out  # numRows x numCols in row-major order
):
		for i in range(64):
			ini = int(((i + 0.5) * inNumRows) / 64)
			for j in range(64):
				inj = int(((j + 0.5) * inNumCols) / 64)
				out[i][j] = in_[ini * inNumCols + inj]

	def dct64To16(self, A, T, B):
		""" Full 64x64 to 64x64 can be optimized e.g. the Lee algorithm.
		But here we only want slots (1-16)x(1-16) of the full 64x64 output.
		Careful experiments showed that using Lee along all 64 slots in one
		dimension, then Lee along 16 slots in the second, followed by
		extracting slots 1-16 of the output, was actually slower than the
		current implementation which is completely non-clever/non-Lee but
		computes only what is needed."""
		D = self.DCT_matrix

		T = np.matmul(D, A)
		#B = np.matmul(T, np.array(D))
		# B = D A Dt
		# B = (D A) Dt ; T = D A
		# T is 16x64;

		# T = D A
		# Tij = sum {k} Dik Akj

		# B = T Dt
		# Bij = sum {k} Tik Djk

		for i in range(16):
			for j in range(16):
				sumk = float(0.0)
				for k in range(64):
					sumk += T[i][k] * D[j][k]
				B[i][j] = sumk

	def dctOutput2hash(self, dctOutput16x16):
		"""
		Each bit of the 16x16 output hash is for whether the given frequency
		component is greater than the median frequency component or not.
		"""
		hash = np.zeros((16,16),dtype="int")
		dctMedian = np.median(dctOutput16x16)
		
		for i in range(16):
			for j in range(16):
				if dctOutput16x16[i][j] > dctMedian:
					hash[15-i,15-j]=1

		if self.hash_as_string: return str(np.array((hash.reshape((256,)))))
		else: return np.array((hash.reshape((256,))))

	@classmethod
	def computeBoxFilterWindowSize(cls, dimension):
		""" Round up."""
		return int(
			(dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
			/ cls.FIND_WINDOW_SIZE_DIVISOR
		)


	def boxFilter2(cls,input,output,rows,cols,rowWin,colWin):
		output[:] = cv2.blur(src = np.float32(input),
						 dst = np.float32(output),
						 ksize = (6,6),
						 borderType = cv2.BORDER_CONSTANT)


	@classmethod
	def prettyHash(cls,hash):
		#Hashes are 16x16. Print in this format
		if len(hash.hash)!=256:
			print("This function only works with 256-bit hashes.")
			return
		return np.array(hash.hash).astype(int).reshape((16,16))


if __name__ == "__main__":
	import sys
	find=FINDHasher()
	for filename in sys.argv[1:]:
		h=find.fromFile(filename)
		print("{},{}".format(h,filename))
		#print(find.prettyHash(h))