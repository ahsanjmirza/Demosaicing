import numpy as np
from scipy import ndimage

class CFAtoRGB:
    def __init__(self):


        return

    # Returns the color of certain pixel in the CFA according to the passed Bayer pattern.
    def get_CFAColor(self, i, j):
        if self.CFA == 'RGGB' or self.CFA == 'rggb':
            if i%2 == 0  and j%2 == 0: return 'R'
            elif i%2 == 1  and j%2 == 0: return 'G'
            elif i%2 == 0  and j%2 == 1: return 'G'
            elif i%2 == 1  and j%2 == 1: return 'B'
        
        elif self.CFA == 'BGGR' or self.CFA == 'bggr':
            if i%2 == 0  and j%2 == 0: return 'B'
            elif i%2 == 1  and j%2 == 0: return 'G'
            elif i%2 == 0  and j%2 == 1: return 'G'
            elif i%2 == 1  and j%2 == 1: return 'R'
        
        elif self.CFA == 'GRBG' or self.CFA == 'grbg':
            if i%2 == 0  and j%2 == 0: return 'G'
            elif i%2 == 1  and j%2 == 0: return 'B'
            elif i%2 == 0  and j%2 == 1: return 'R'
            elif i%2 == 1  and j%2 == 1: return 'G'

        elif self.CFA == 'GBRG' or self.CFA == 'gbrg':
            if i%2 == 0  and j%2 == 0: return 'G'
            elif i%2 == 1  and j%2 == 0: return 'R'
            elif i%2 == 0  and j%2 == 1: return 'B'
            elif i%2 == 1  and j%2 == 1: return 'G'
        return

    # The function takes a single channel raw image with a certain bitdepth 
    # and any of the four Bayer patterns as input.
    def set_raw(self, raw, CFA, bitdepth):
        # The bitdepth value should be between 8 and 16.
        self.bitdepth = bitdepth
        self.maxValBD = (2**self.bitdepth)-1
        
        # The CFA can be any of the four Bayer patterns.
        self.CFA = CFA
        self.raw = np.float32(raw)
        self.bayer_3ch = np.zeros((self.raw.shape[0], self.raw.shape[1], 3), dtype=np.float32)

        # The pixelColorArr is a char single channel array which holds the color
        # the color of each pixel in the CFA for the entire image.
        self.pixelColorArr = np.zeros(self.raw.shape, dtype='S1')
        for i in range(self.raw.shape[0]):
            for j in range(self.raw.shape[1]):
                color = self.get_CFAColor(i, j)
                self.pixelColorArr[i, j] = color
                if color == 'R': self.bayer_3ch[i, j, 0] = self.raw[i, j]
                elif color == 'G': self.bayer_3ch[i, j, 1] = self.raw[i, j]
                elif color == 'B': self.bayer_3ch[i, j, 2] = self.raw[i, j]

        # Since the Buffer size requirements of the algorithm is 7x7,
        # therefore a padding of 3 is added to the raw image.

        # i_start, j_start, i_end and j_end hold the values of the starting and 
        # ending indices of the padded raw image.  
        (self.i_start, self.j_start) = (3, 3)
        (self.i_end, self.j_end) = (self.raw.shape[0]+3, self.raw.shape[1]+3)
        self.raw = np.pad(self.raw, 3, 'reflect')

        self.bayer_3ch = np.pad(self.bayer_3ch, ((3, 3), (3, 3), (0, 0)), 'reflect')
        self.pixelColorArr = np.pad(self.pixelColorArr, 3, 'reflect')
        return