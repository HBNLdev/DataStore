''' code to create small images of channel names.
    creates 128 x 128 .png's using matplotlib, then
    uses PIL to shrink them to 32 x 32 '''

import os
import matplotlib.pyplot as plt
from PIL import Image

# change this output folder to what you want
folder = '/active_projects/mort/peak_picker/chanlogos/'

layout = [[None, 'FP1', 'Y', 'FP2', 'X'],
          ['F7', 'AF1', None, 'AF2', 'F8'],
          [None, 'F3', 'FZ', 'F4', None],
          ['FC5', 'FC1', None, 'FC2', 'FC6'],
          ['T7', 'C3', 'CZ', 'C4', 'T8'],
          ['CP5', 'CP1', None, 'CP2', 'CP6'],
          [None, 'P3', 'PZ', 'P4', None],
          ['P7', 'PO1', None, 'PO2', 'P8'],
          [None, 'O1', None, 'O2', None],
          ['AF7', 'FPZ', 'AFZ', None, 'AF8'],
          ['F5', 'F1', None, 'F2', 'F6'],
          ['FT7', 'FC3', 'FCZ', 'FC4', 'FT8'],
          ['C5', 'C1', None, 'C2', 'C6'],
          ['TP7', 'CP3', 'CPZ', 'CP4', 'TP8'],
          ['P5', 'P1', 'POZ', 'P2', 'P6'],
          [None, 'PO7', 'OZ', 'PO8', None]]

chans = []
for row in layout:
    for chan in row:
        if chan:
            chans.append(chan)

def make_logo(txt, folder):
    width = 1.08
    height = 1.08
    f = plt.figure(figsize=(width, height))
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_axis_bgcolor('black')
    #ax.set_axis_off()
    ax.text(0.5, 0.5, txt,
           color='white',
           family='monospace',
           ha='center',
           va='center',
           weight='bold',
           size=32)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    outpath = os.path.join(folder, txt+'.png')
    plt.savefig(outpath, bbox_inches='tight', dpi=100)

def make_smaller(file, size):
    im = Image.open(file)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(file, 'PNG')

def __main__():    
    for chan in chans:
        make_logo(chan, folder)
        make_smaller(folder+chan+'.png', (20, 20))