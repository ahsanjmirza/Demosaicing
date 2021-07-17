import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from tqdm import tqdm
import itertools

class CFAtoRGB:
    def __init__(self, 
                 bitdepth, 
                 CFA,
                 g_grad_th,
                 rb_grad_th):
        self.bitdepth = bitdepth
        self.maxValBD = (2**self.bitdepth)-1
        self.CFA = CFA
        self.g_grad_th = g_grad_th
        self.rb_grad_th = rb_grad_th
        return

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

    def set_raw(self, raw):
        self.raw = np.float32(raw)
        self.bayer_3ch = np.zeros((self.raw.shape[0], self.raw.shape[1], 3), dtype=np.float32)

        self.pixelColorArr = np.zeros(self.raw.shape, dtype='S1')
        for i in range(self.raw.shape[0]):
            for j in range(self.raw.shape[1]):
                color = self.get_CFAColor(i, j)
                self.pixelColorArr[i, j] = color
                if color == 'R': self.bayer_3ch[i, j, 0] = self.raw[i, j]
                elif color == 'G': self.bayer_3ch[i, j, 1] = self.raw[i, j]
                elif color == 'B': self.bayer_3ch[i, j, 2] = self.raw[i, j]

        (self.i_start, self.j_start) = (3, 3)
        (self.i_end, self.j_end) = (self.raw.shape[0]+3, self.raw.shape[1]+3)
        self.raw = np.pad(self.raw, 3, 'reflect')

        self.bayer_3ch = np.pad(self.bayer_3ch, ((3, 3), (3, 3), (0, 0)), 'reflect')
        self.pixelColorArr = np.pad(self.pixelColorArr, 3, 'reflect')
        return

    def calc_weights(self):
        gauss_sigma1 = np.array([[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]])
        gauss_sigma1_2D = np.multiply(gauss_sigma1.T, gauss_sigma1)

        gradh = ndimage.correlate(np.abs(self.raw - ndimage.correlate(self.raw, \
            gauss_sigma1)), gauss_sigma1_2D)
        gradv = ndimage.correlate(np.abs(self.raw - ndimage.correlate(self.raw, \
            gauss_sigma1.T)), gauss_sigma1_2D)

        self.w_h = np.exp(-gradh)
        self.w_v = np.exp(-gradv)

        
        index_ = np.array(list(itertools.product(range(self.i_start, self.i_end), 
                                                 range(self.j_start, self.j_end))))
        i, j = index_[:, 0], index_[:, 1]
        self.grad135 = np.zeros(self.raw.shape, dtype=np.float32)
        self.grad45 = np.zeros(self.raw.shape, dtype=np.float32)
        self.grad135[i, j] = np.abs(self.raw[i-1, j-1]-self.raw[i+1, j+1])
        self.grad135[i, j] += np.abs(self.raw[i-2, j-2]-self.raw[i, j])
        self.grad135[i, j] += np.abs(self.raw[i, j]-self.raw[i+2, j+2])
        self.grad135[i, j] += np.abs(self.raw[i-1, j-2]-self.raw[i, j-1])
        self.grad135[i, j] += np.abs(self.raw[i, j-1]-self.raw[i+1, j])
        self.grad135[i, j] += np.abs(self.raw[i+1, j]-self.raw[i+2, j+1])
        self.grad135[i, j] += np.abs(self.raw[i-2, j-1]-self.raw[i-1, j])
        self.grad135[i, j] += np.abs(self.raw[i-1, j]-self.raw[i, j+1])
        self.grad135[i, j] += np.abs(self.raw[i, j+1]-self.raw[i+1, j+2])
        self.grad45[i, j] = np.abs(self.raw[i-2, j+2]-self.raw[i, j])
        self.grad45[i, j] += np.abs(self.raw[i-1, j+1]-self.raw[i+1, j-1])
        self.grad45[i, j] += np.abs(self.raw[i, j]-self.raw[i+2, j-2])
        self.grad45[i, j] += np.abs(self.raw[i-2, j+1]-self.raw[i-1, j])
        self.grad45[i, j] += np.abs(self.raw[i-1, j]-self.raw[i, j-1])
        self.grad45[i, j] += np.abs(self.raw[i, j-1]-self.raw[i+1, j-2])
        self.grad45[i, j] += np.abs(self.raw[i-1, j+2]-self.raw[i, j+1])
        self.grad45[i, j] += np.abs(self.raw[i, j+1]-self.raw[i+1, j])
        self.grad45[i, j] += np.abs(self.raw[i+1, j]-self.raw[i+2, j-1])
        self.grad45[self.grad45==0], self.grad135[self.grad135==0] = 1, 1
        return

    def demosaic(self):
        self.calc_weights()

        self.red = np.zeros(self.raw.shape, dtype=np.float32)
        self.blue = np.zeros(self.raw.shape, dtype=np.float32)
        self.green = np.zeros(self.raw.shape, dtype=np.float32)

        for i in tqdm(range(self.i_start, self.i_end), desc='Interpolating Green'):
            for j in range(self.j_start, self.j_end):

                if self.pixelColorArr[i, j] != b'G':
                    self.green[i, j] = self.get_interpolatedG(i, j)

                else: self.green[i, j] = self.raw[i, j]

        self.bayer_3ch[:,:,1] = self.green
        del self.green
        
        for i in tqdm(range(self.i_start, self.i_end), desc='Interpolating RB on BR'):
            for j in range(self.j_start, self.j_end):

                if self.pixelColorArr[i, j] == b'R':
                    self.blue[i, j] = self.get_interpolatedRBonBR(i, j)
                    self.red[i, j] = self.raw[i, j]

                elif self.pixelColorArr[i, j] == b'B':
                    self.red[i, j] = self.get_interpolatedRBonBR(i, j)
                    self.blue[i, j] = self.raw[i, j]

        self.bayer_3ch[:, :, 0] = self.red
        self.bayer_3ch[:, :, 2] = self.blue

        for i in tqdm(range(self.i_start, self.i_end), desc='Interpolating RB on G'):
            for j in range(self.j_start, self.j_end):
                if self.pixelColorArr[i, j] == b'G':
                    (self.red[i, j], self.blue[i, j]) = self.get_interpolatedRBonG(i, j) 


        self.bayer_3ch[:, :, 0] = self.red
        self.bayer_3ch[:, :, 2] = self.blue

        self.bayer_3ch = np.clip(self.bayer_3ch, 0, self.maxValBD)
        self.bayer_3ch = np.uint16(self.bayer_3ch)>>(self.bitdepth-8)
        self.bayer_3ch = np.uint8(self.bayer_3ch)
        return self.bayer_3ch[self.i_start:self.i_end, self.j_start:self.j_end, :]

    def getGreen(self, i, j, dir):
        hg = -0.25*self.raw[i, j-2] + 0.5*self.raw[i, j-1] + \
                0.5*self.raw[i, j] + 0.5*self.raw[i, j+1] - 0.25*self.raw[i, j+2]
        vg = -0.25*self.raw[i-2, j] + 0.5*self.raw[i-1, j] + \
                0.5*self.raw[i, j] + 0.5*self.raw[i+1, j] - 0.25*self.raw[i+2, j]
        if dir == 'H': return hg
        elif dir == 'V': return vg
        hg *= self.w_h[i, j]
        vg *= self.w_v[i, j]
        return (hg+vg)/(self.w_h[i, j]+self.w_v[i, j])
        
        
    def get_interpolatedG(self, i, j):
        if self.g_grad_th*self.w_h[i, j] > self.w_v[i, j]:
            return self.getGreen(i, j, 'H')
        elif self.g_grad_th*self.w_v[i, j] > self.w_h[i, j]:
            return self.getGreen(i, j, 'V')
        return self.getGreen(i, j, 'W')

    def get_interpolatedRBonBR(self, i, j):
        deltas = self.bayer_3ch[i-1:i+2, j-1:j+2, 1]-self.raw[i-1:i+2, j-1:j+2]
        if self.grad135[i, j]*self.rb_grad_th > self.grad45[i, j]:
            residual = (deltas[2, 0]+deltas[0, 2])/2
        elif self.grad45[i, j]*self.rb_grad_th > self.grad135[i, j]:
            residual = (deltas[0, 0]+deltas[2, 2])/2
        else: 
            residual = (deltas[2, 0]+deltas[0, 2]+deltas[0, 0]+deltas[2, 2])/4
        out = self.bayer_3ch[i, j, 1] - residual
        return out
    
    def get_interpolatedRBonG(self, i, j):
        red = self.bayer_3ch[i, j, 1]-((self.w_v[i, j]*(
            self.bayer_3ch[i-1, j, 1]-self.bayer_3ch[i-1, j, 0]+self.bayer_3ch[i+1, j, 1]-self.bayer_3ch[i+1, j, 0]))/(2*(self.w_v[i, j]+self.w_h[i, j])))
        red -= ((self.w_h[i, j]*(self.bayer_3ch[i, j-1, 1]-self.bayer_3ch[i, j-1, 0] +
                                                                  self.bayer_3ch[i, j+1, 1]-self.bayer_3ch[i, j+1, 0]))/(2*(self.w_v[i, j]+self.w_h[i, j])))
        blue = self.bayer_3ch[i, j, 1]-((self.w_v[i, j]*(
            self.bayer_3ch[i-1, j, 1]-self.bayer_3ch[i-1, j, 2]+self.bayer_3ch[i+1, j, 1]-self.bayer_3ch[i+1, j, 2]))/(2*(self.w_v[i, j]+self.w_h[i, j])))
        blue -= ((self.w_h[i, j]*(self.bayer_3ch[i, j-1, 1]-self.bayer_3ch[i, j-1, 2] +
                                                                  self.bayer_3ch[i, j+1, 1]-self.bayer_3ch[i, j+1, 2]))/(2*(self.w_v[i, j]+self.w_h[i, j])))
        return (red, blue)
