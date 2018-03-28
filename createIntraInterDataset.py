import random
import os
import functions
import shlex, subprocess
import numpy as np
import time
import cv2
import functions
import struct
import math
import showBitstreamInfo
import encodeYUV

# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

saveFolder = "qpFiles/"

datadir = '/Volumes/LaCie/data/yuv'
#videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
#annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
#baseFileName = 'ILSVRC2017_train_00000000'

x264 = "x264"
ldecod = "ldecod"
ffmpeg = "ffmpeg"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

fileSizes = [
    ['qcif', 176, 144],
    ['512x384', 512, 384],
    ['384x512', 384, 512],
    ['cif', 352, 288],
    ['sif', 352, 240],
    ['720p', 1280, 720],
    ['1080p', 1920, 1080]
]

quants = [0, 7, 14, 21, 28, 35, 42, 49]

def splitTraceFileIntoComponentFiles(filename):
    frameData, width, height = showBitstreamInfo.processAfile(filename)
    mbwidth = int(width/16)
    mbheight = int(height/16)
    size = int((width/16)*(height/16))
    frames = int(frameData.shape[0]/size)
    print("There are {} frames".format(frames))
    #print(frameData[0:(size*2), :])



    basefilename, ext = os.path.splitext(filename)
    YUVfilename = "{}.yuv".format(basefilename)


    #frameNos = np.reshape(frameNos, (frames, mbheight, mbwidth))
    frameNos = frameData[:, 0]
    FrameNofilename = "{}.frameno".format(basefilename)
    frameNos = frameNos.flatten()
    functions.saveToFile(frameNos, FrameNofilename)

    mbNos = frameData[:, 1]
    MbNofilename = "{}.mbno".format(basefilename)
    mbNos = mbNos.flatten()
    functions.saveToFile(mbNos, MbNofilename)

    modes = frameData[:, 2] #inter/intra
    skipped = frameData[:, 3]
    modes = (1-modes) + skipped
    MBModefilename = "{}.mbmode".format(basefilename)
    modes = modes.flatten()
    functions.saveToFile(modes, MBModefilename)

    qps = frameData[:, 4]
    QPfilename = "{}.qp".format(basefilename)
    qps = qps.flatten()
    functions.saveToFile(qps, QPfilename)

    # motion vectors. There are 16 of them for each macroblock. x and y for every 4x4 unit.
    mvs = frameData[:, 5:]
    MVfilename = "{}.mv".format(basefilename)
    mvs = np.reshape(mvs, (frames, mbheight, mbwidth, 4, 4, 2))
    mvs = np.swapaxes(mvs, 2,3) # translate mbs with 4x4s into just rows
    mvs = np.reshape(mvs, (frames, mbheight*4, mbwidth*4, 2))
    mvs = np.swapaxes(mvs, 2, 3) #this and the next line moving to 2 planar channels
    mvs = np.swapaxes(mvs, 1, 2)
    mvs = mvs.flatten() + 128
    functions.saveToFile(mvs, MVfilename)


if __name__ == "__main__":

    #encodeYUV.encodeAWholeFolderAsH264("/Users/pam/Documents/data/h264/", takeAll=True)
    #quit()

    filename = "/Users/pam/Documents/data/h264/carphone_qcif_q0.h264"
    filename = "/Users/pam/Documents/data/DeepFakes/creepy1.h264"
    #splitTraceFileIntoComponentFiles(filename)

