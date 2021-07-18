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
                 rb_grad_th,
                 var_th):
        self.bitdepth = bitdepth
        self.maxValBD = (2**self.bitdepth)-1
        self.CFA = CFA
        self.g_grad_th = g_grad_th
        self.rb_grad_th = rb_grad_th
        self.var_th = var_th
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
        gauss_sigma1 = np.array([[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]], dtype=np.float32)
        gauss_sigma1_2D = np.multiply(gauss_sigma1.T, gauss_sigma1)

        gradh = ndimage.correlate(np.abs(self.raw - ndimage.correlate(self.raw, \
            gauss_sigma1)), gauss_sigma1.T)
        gradv = ndimage.correlate(np.abs(self.raw - ndimage.correlate(self.raw, \
            gauss_sigma1.T)), gauss_sigma1)

        # print('gradh:',gradh.min(),'-',gradh.max())
        # print('gradv:',gradv.min(),'-',gradv.max())

        self.w_h1 = np.exp(-gradh)
        self.w_v1 = np.exp(-gradv)

        gauss_sigma1 = np.array([[0.0385, 0.1538, 0.6154, 0.1538, 0.0385]], dtype=np.float32)

        gradh = ndimage.correlate(np.abs(ndimage.correlate(self.raw, np.array([[-1, 2, 0, -2, 1]]))), gauss_sigma1.T)
        gradv = ndimage.correlate(np.abs(ndimage.correlate(self.raw, np.array([[-1, 2, 0, -2, 1]]).T)), gauss_sigma1)
        gradh[gradh<1], gradv[gradv<1] = 1, 1

        self.w_h2, self.w_v2 = gradv, gradh

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

    def calc_variance(self):
        index_ = np.array(list(itertools.product(range(self.i_start, self.i_end), 
                                                 range(self.j_start, self.j_end))))
        i, j = index_[:, 0], index_[:, 1]

        varRB = np.zeros(self.raw.shape, dtype=np.float32)
        varG = np.zeros(self.raw.shape, dtype=np.float32)
        meanRB = ndimage.correlate(self.raw, np.array([[1, 0, 1, 0, 1],
                                                       [0, 0, 0, 0, 0],
                                                       [1, 0, 1, 0, 1],
                                                       [0, 0, 0, 0, 0],
                                                       [1, 0, 1, 0, 1]]))/9

        meanG = ndimage.correlate(self.raw, np.array([[0, 1, 0, 1, 0],
                                                      [1, 0, 1, 0, 1],
                                                      [0, 1, 0, 1, 0],
                                                      [1, 0, 1, 0, 1],
                                                      [0, 1, 0, 1, 0]]))/12

        varRB[i, j] = ((self.raw[i-2, j-2]-meanRB[i, j])**2)+((self.raw[i-2, j]-meanRB[i, j])**2)
        varRB[i, j] += ((self.raw[i-2, j+2]-meanRB[i, j])**2)+((self.raw[i, j-2]-meanRB[i, j])**2)
        varRB[i, j] += ((self.raw[i, j]-meanRB[i, j])**2)+((self.raw[i, j+2]-meanRB[i, j])**2)
        varRB[i, j] += ((self.raw[i+2, j-2]-meanRB[i, j])**2)+((self.raw[i+2, j]-meanRB[i, j])**2)
        varRB[i, j] += ((self.raw[i+2, j+2]-meanRB[i, j])**2)

        varG[i, j] = ((self.raw[i-2, j-1]-meanG[i, j])**2)+((self.raw[i-2, j+1]-meanG[i, j])**2)
        varG[i, j] += ((self.raw[i-1, j-2]-meanG[i, j])**2)+((self.raw[i-1, j]-meanG[i, j])**2)
        varG[i, j] += ((self.raw[i-1, j+2]-meanG[i, j])**2)+((self.raw[i, j-1]-meanG[i, j])**2)
        varG[i, j] += ((self.raw[i, j+1]-meanG[i, j])**2)+((self.raw[i+1, j-2]-meanG[i, j])**2)
        varG[i, j] += ((self.raw[i+1, j]-meanG[i, j])**2)+((self.raw[i+1, j+2]-meanG[i, j])**2)
        varG[i, j] += ((self.raw[i+2, j-1]-meanG[i, j])**2)+((self.raw[i+2, j+1]-meanG[i, j])**2)
        varRB, varG = varRB/9, varG/12

        self.var5x5 = np.abs(varG-varRB)
        return

    def demosaic(self):
        self.calc_weights()
        self.calc_variance()

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
        elif dir == 'W':
            if self.var5x5[i, j]>self.var_th:
                hg *= self.w_h1[i, j]
                vg *= self.w_v1[i, j]
                return (hg+vg)/(self.w_h1[i, j]+self.w_v1[i, j])
            else:
                hg *= self.w_h2[i, j]
                vg *= self.w_v2[i, j]
                return (hg+vg)/(self.w_h2[i, j]+self.w_v2[i, j])
        
    def get_interpolatedG(self, i, j):
        if self.g_grad_th*self.w_h2[i, j] > self.w_v2[i, j]:
            return self.getGreen(i, j, 'H')
        elif self.g_grad_th*self.w_v2[i, j] > self.w_h2[i, j]:
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
        red = self.bayer_3ch[i, j, 1]-((self.w_v1[i, j]*(
            self.bayer_3ch[i-1, j, 1]-self.bayer_3ch[i-1, j, 0]+self.bayer_3ch[i+1, j, 1]-self.bayer_3ch[i+1, j, 0]))/(2*(self.w_v1[i, j]+self.w_h1[i, j])))
        red -= ((self.w_h1[i, j]*(self.bayer_3ch[i, j-1, 1]-self.bayer_3ch[i, j-1, 0] +
                                                                  self.bayer_3ch[i, j+1, 1]-self.bayer_3ch[i, j+1, 0]))/(2*(self.w_v1[i, j]+self.w_h1[i, j])))
        blue = self.bayer_3ch[i, j, 1]-((self.w_v1[i, j]*(
            self.bayer_3ch[i-1, j, 1]-self.bayer_3ch[i-1, j, 2]+self.bayer_3ch[i+1, j, 1]-self.bayer_3ch[i+1, j, 2]))/(2*(self.w_v1[i, j]+self.w_h1[i, j])))
        blue -= ((self.w_h1[i, j]*(self.bayer_3ch[i, j-1, 1]-self.bayer_3ch[i, j-1, 2] +
                                                                  self.bayer_3ch[i, j+1, 1]-self.bayer_3ch[i, j+1, 2]))/(2*(self.w_v1[i, j]+self.w_h1[i, j])))
        return (red, blue)
