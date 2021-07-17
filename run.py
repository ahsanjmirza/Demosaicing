import numpy as np
from imageio import imwrite, imread
from CFAFilt import CFAtoRGB
import json
import os

def main():
    f = open('config.json',)
    config = json.load(f)
    f.close()
    dm = CFAtoRGB(CFA=config['cfa'], 
                  bitdepth=config['bitdepth'],
                  g_grad_th=config['g_grad_th'], 
                  rb_grad_th=config['rb_grad_th'])
    for f in os.listdir(config['image_path']):
        print(f)
        # raw = np.fromfile(os.path.join(config['image_path'], f), dtype=np.uint16)
        # raw = raw.reshape((config['rows'], config['cols']))
        img = imread(os.path.join(config['image_path'], f))
        raw = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
        raw[::2, ::2] = img[::2, ::2, 0]
        raw[1::2, ::2] = img[1::2, ::2, 1]
        raw[::2, 1::2] = img[::2, 1::2, 1]
        raw[1::2, 1::2] = img[1::2, 1::2, 2]
        dm.set_raw(raw)
        out = dm.demosaic()
        imwrite(os.path.join(config['output_path'], 'dm_'+f.replace('.raw', '.png')), out)
    return

if __name__ == '__main__':
    main()