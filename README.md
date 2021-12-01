# Oxford-Data-Analytics-at-Scale
This is a repository containing a sample of the code for my final project at Oxford for the class "Data Analytics at Scale" (DAS). The goal of the final project was to optimize an image hashing algorithm, profile it, and benchmark it to other image hashing alternatives.

There are three files in the main directory:
- FINd Report.pdf - the main report for the DAS submission
- findNotebook.html - the main notebook used for the analysis. Use this to inspect the code
- findNotebook.ipynb - the main notebook used for the analysis. Use this to run the code

There are three folders:
1. Optimizations - provides the three optimizations (details below).
2. Time measurements - provides the code for extracting the times well as the actual time measurements used in the notebook for multiprocessing and image scaling. 
3. Groundtruth data - offers some pickle files used to test the accuracy of images.

The optimization folder includes three optimizations with the following files:
1. Cython:
- FINd_Cython.pyx
- FINd_Cython.c
- FINd_Cython.so
- setup.py

2. Numpy/CV2:
- FINd_CV2.py

3. Spark
- pyspark.ex.py - the code for the Spark optimizations. Note that the spark optimization uses the numpy optimized version for speed purposes
- FINd_multiprocessing.py - multiprocessing file used to benchmark Spark benefits

Please note that in the implementation and code, it is assumed that the relevant files are in the same directory. If you would like to run the code and the file is missing, please put them in the same directories or change the path.

Some code, such as line profiling, is not added, since it requires only simple modifications to the code presented in this file.
