import cython
from PIL import Image
import numpy as np
#cimport numpy as np # <--- have to remove since does not compile

from imagehash import ImageHash
import matplotlib.image as mpimg
cimport libc.math as math
# from cython.parallel import prange <--- did not compile


# Define Customer min and max functions
cdef int mymax(int a, int b):
    return a if a > b else b

cdef int mymin(int a, int b):
    return a if a < b else b

    
cdef class FINDHasherCython:
    

    cdef double[:,:] DCT_matrix
    cdef int[:,:,:] img
    
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self):
        """See also comments on dct64To16. Input is (0..63)x(0..63); output is
        (1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
        Returns 16x64 matrix."""
        cdef int i,j
        cdef double matrix_scale_factor 
        
        # Instantiate the DCT Matrix during init
        cdef float[:] DCT_matrix
        self.DCT_matrix = np.empty((16,64), dtype='double')
        matrix_scale_factor = math.sqrt(2.0 / 64.0)
        for i in range(0, 16):
            for j in range(0, 64):
                self.DCT_matrix[i,j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))

    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fromFile(self, filepath):

        img = Image.open(filepath)
        return self.fromImage(img)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fromImage(self, img):
        try:
            img.thumbnail((512, 512)) 
        except IOError as e:
            raise e
            
        img = np.asarray(img, dtype=np.dtype("i"))
        cdef int numCols, numRows, _
        cdef int FIND_WINDOW_SIZE_DIVISOR = 64
        
        cdef float[:] buffer1
        cdef float[:] buffer2 
        cdef float[:,:] buffer64x64 
        cdef float[:,:] buffer16x64
        cdef float[:,:] buffer16x16
        
        numCols = img.shape[0]
        numRows = img.shape[1]
        buffer1 = np.zeros(numRows*numCols, dtype = np.float32)
        buffer2 = np.zeros(numRows*numCols, dtype =  np.float32)
        buffer64x64 = np.zeros((64,64), dtype =  np.float32)
        buffer16x64 = np.zeros((16,64), dtype =  np.float32)
        buffer16x16 = np.zeros((16,16), dtype =  np.float32)
        
        self.fillFloatLumaFromBufferImage(img, buffer1, numCols, numRows)
        
        # findhash265fromfloatluma function
        
        cdef int windowSizeAlongRows = (numCols + FIND_WINDOW_SIZE_DIVISOR - 1) / FIND_WINDOW_SIZE_DIVISOR
        cdef int windowSizeAlongCols = (numRows + FIND_WINDOW_SIZE_DIVISOR -1) / FIND_WINDOW_SIZE_DIVISOR

        self.boxFilter(buffer1,buffer2,numRows,numCols,windowSizeAlongRows,windowSizeAlongCols)

        self.decimatedouble(buffer2, numRows, numCols, buffer64x64)
        self.dct64To16(buffer64x64, buffer16x64, buffer16x16)
        hash = self.dctOutput2hash(buffer16x16)
        return hash


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fillFloatLumaFromBufferImage(self, int[:,:,:] img, float[:] buffer1, int numCols, int numRows):

        cdef float LUMA_FROM_R_COEFF, LUMA_FROM_G_COEF, LUMA_FROM_B_COEF 
        LUMA_FROM_R_COEFF = 0.299
        LUMA_FROM_G_COEFF = 0.587
        LUMA_FROM_B_COEFF = 0.114
        
        cdef int i, j,
        for i in range(0, numRows):
            for j in range(numCols):

                buffer1[i * numCols + j] = (
                     LUMA_FROM_R_COEFF * img[i,j,0]
                    + LUMA_FROM_G_COEFF * img[i,j,1]
                    + LUMA_FROM_B_COEFF * img[i,j,2]
                )


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef decimatedouble(
        cls, float[:] buffer1, int inNumRows, int inNumCols, float[:,:] buffer64x64  # numRows x numCols in row-major order
    ):
        cdef int i, j, ini, inj        
        for i in range(64):
            ini = int(((i + 0.5) * inNumRows) / 64)
            for j in range(64):
                inj = int(((j + 0.5) * inNumCols) / 64)
                buffer64x64[i][j] = buffer1[ini * inNumCols + inj]

    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef dct64To16(self, float[:,:] buffer64x64, float[:,:] doublbuffer16x64,  float[:,:] buffer16x16): # A T B
        """ Full 64x64 to 64x64 can be optimized e.g. the Lee algorithm.
        But here we only want slots (1-16)x(1-16) of the full 64x64 output.
        Careful experiments showed that using Lee along all 64 slots in one
        dimension, then Lee along 16 slots in the second, followed by
        extracting slots 1-16 of the output, was actually slower than the
        current implementation which is completely non-clever/non-Lee but
        computes only what is needed."""
        
        buffer16x64 = np.zeros((16,64), dtype =  'double')
        
        cdef int i, j
        cdef double tij
        cdef float[:] ti
        
        for i in range(0, 16):
            ti = np.zeros([64], dtype = np.float32)

            for j in range(0, 64):
                tij = 0.0
                for k in range(0, 64):
                    tij += self.DCT_matrix[i][k] * buffer64x64[k][j]
                ti[j] = tij

            buffer16x64[i] = ti

        cdef float sumk
        for i in range(16):
            for j in range(16):
                sumk = float(0.0)
                for k in range(64):
                    sumk += buffer16x64[i][k] * self.DCT_matrix[j][k]
                buffer16x16[i][j] = sumk
                
      
    @cython.cdivision(True)
    @cython.boundscheck(False)  
    @cython.wraparound(False)
    cdef torben(self, float[:,:] m, int numRows, int numCols):
        
        cdef int n, midn, less, greater, equal, i, j, _i, _j
        cdef float min, max, guess, maxltguess, mingtguess, v, final_result
        cdef bint should_keep_searching = True

        min = m[0, 0]
        max = m[0, 0]

        n = numRows * numCols
        midn = int((n + 1) / 2)

        
        for i in range(numRows):
            for j in range(numCols):
                v = m[i, j]
                if v < min:
                    min = v
                if v > max:
                    max = v

        while should_keep_searching:
            guess = float((min + max) / 2)
            less = 0
            greater = 0
            equal = 0
            maxltguess = min
            mingtguess = max

            for _i in range(numRows):
                for _j in range(numCols):
                    v = m[_i, _j]
                    if v < guess:
                        less += 1
                        if v > maxltguess:
                            maxltguess = v
                    elif v > guess:
                        greater += 1
                        if v < mingtguess:
                            mingtguess = v
                    else:
                        equal += 1
            if less <= midn and greater <= midn:
                break
            elif less > greater:
                max = maxltguess
            else:
                min = mingtguess
        if less >= midn:
            final_result = maxltguess
            should_keep_searching = False
        elif less + equal >= midn:
            final_result = guess
            should_keep_searching = False
        else:
            final_result = mingtguess
            should_keep_searching = False

        return final_result
    
                
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef dctOutput2hash(self, float[:, :]dctOutput16x16):
        """
        Each bit of the 16x16 output hash is for whether the given frequency
        component is greater than the median frequency component or not.
        """
        
        hash = np.zeros([16,16], dtype='int')
        cdef float dctMedian = self.torben(dctOutput16x16, 16, 16)
        
        cdef int i, j
        for i in range(16):
            for j in range(16):
                if dctOutput16x16[i][j] > dctMedian:
                    hash[15-i,15-j]=1
        return ImageHash(hash.reshape((256,)))

      
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef computeBoxFilterWindowSize(cls, int dimension, int FIND_WINDOW_SIZE_DIVISOR):
        """ Round up."""
        
        return int(
            (dimension + FIND_WINDOW_SIZE_DIVISOR - 1)
            / FIND_WINDOW_SIZE_DIVISOR
        )
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef boxFilter(cls, float[:] buffer1, float[:] buffer2, int rows, int cols, int rowWin, int colWin):
        cdef int halfColWin = int((colWin + 2) / 2)  
        cdef int halfRowWin = int((rowWin + 2) / 2)
        cdef int i, j, xmin, xmax, ymin, ymax, k, l
        cdef double s
        cdef int zero = 0
        
        for i in range(0,rows):
            for j in range(0,cols):
                s=0

                xmin=mymax(zero,i-halfRowWin)                
                xmax=mymin(rows,i+halfRowWin)
                ymin=mymax(zero,j-halfColWin)
                ymax=mymin(cols,j+halfColWin)
                for k in range(xmin,xmax):
                    for l in range(ymin,ymax):
                        s+=buffer1[k*rows+l]
                buffer2[i*rows+j]=s/((xmax-xmin)*(ymax-ymin))

                
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef prettyHash(cls, int hash):
        #Hashes are 16x16. Print in this format
        if len(hash.hash)!=256:
            print("This function only works with 256-bit hashes.")
            return
        return np.array(hash.hash).astype(int).reshape((16,16))