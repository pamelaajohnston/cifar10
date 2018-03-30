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

def splitTraceFileIntoComponentFiles(filename, width, height):
    frameData, width, height = showBitstreamInfo.processAfile(filename, width=width, height=height, noRedos=False)
    mbwidth = int(width/16)
    mbheight = int(height/16)
    size = int((width/16)*(height/16))
    frames = int(frameData.shape[0]/size)
    print("There are {} frames".format(frames))
    #print(frameData[0:(size*2), :])



    basefilename, ext = os.path.splitext(filename)
    YUVfilename = "{}.yuv".format(basefilename)
    maybeYUVfilename = "{}_ViewID0000.yuv".format(basefilename)
    if os.path.isfile(maybeYUVfilename):
        print("renaming...")
        os.rename(maybeYUVfilename, YUVfilename)
    maybeYUVfilename = "{}_ViewID0001.yuv".format(basefilename)
    if os.path.isfile(maybeYUVfilename):
        os.remove(maybeYUVfilename)


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

def extractPixelPatch(filename, frameNo, mbNo, frameW, frameH, channels=1.5, patchW=16, patchH=16, xstride=16, ystride=16, lborder=8, rborder=8, tborder=8, bborder=8):
    frameSize = int(frameW * frameH * channels)
    bytePos = frameSize * frameNo
    mbwidth = int(width/16)
    mbheight = int(height/16)
    mbsize = mbwidth*mbheight

    with open(filename, "rb") as f:
        f.seek(bytePos)
        pixels = f.read(frameSize)
        pixels = bytearray(pixels)
        # Adding the border to the frame saves us worrying about testing borders during patch-ification
        frame444 = functions.YUV420_2_YUV444(pixels, height, width)
        frame444, newWidth, newHeight = encodeYUV.addABorder(frame444, width, height, lborder, rborder, tborder, bborder)

    # with the border, (lborder, tborder) is the top left of the first macroblock
    ySize = newWidth * newHeight
    uvSize = newWidth * newHeight
    frameData = frame444
    yData = frameData[0:ySize]
    uData = frameData[ySize:(ySize + uvSize)]
    vData = frameData[(ySize + uvSize):(ySize + uvSize + uvSize)]
    yData = yData.reshape(newHeight, newWidth)
    uData = uData.reshape(newHeight, newWidth)
    vData = vData.reshape(newHeight, newWidth)

    xCo = int((mbNo % mbsize) * 16)
    yCo = int(((mbNo-xCo) % mbsize) * 16)

    patchY = yData[yCo:(yCo + patchH), xCo:(xCo + patchW)]
    patchU = uData[yCo:(yCo + patchH), xCo:(xCo + patchW)]
    patchV = vData[yCo:(yCo + patchH), xCo:(xCo + patchW)]


    yuv = np.concatenate(
        (np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
    yuv = yuv.flatten()
    return yuv

def extractPatchesAndReturn(fileName, frameNo, frameW, frameH, channels=1.5, patchW=32, patchH=32, xstride=16, ystride=16, lborder=8, rborder=8, tborder=8, bborder=8):
    patchesList = []
    numPatches = 0
    frameSize = int(frameW * frameH * channels)
    bytePos = frameSize * frameNo
    print("The byte position in the yuv file {}: {}".format(fileName, bytePos))
    mbwidth = int(width/16)
    mbheight = int(height/16)
    mbsize = mbwidth*mbheight
    with open(fileName, "rb") as f:
        f.seek(bytePos)
        pixels = f.read(frameSize)
        pixels = bytearray(pixels)
        # Adding the border to the frame saves us worrying about testing borders during patch-ification
        frame444 = functions.YUV420_2_YUV444(pixels, height, width)
        frame444, newWidth, newHeight = encodeYUV.addABorder(frame444, width, height, lborder, rborder, tborder, bborder)

    frameData = frame444
    ySize = newWidth * newHeight
    uvSize = newWidth * newHeight
    frameData = frame444
    yData = frameData[0:ySize]
    uData = frameData[ySize:(ySize + uvSize)]
    vData = frameData[(ySize + uvSize):(ySize + uvSize + uvSize)]
    yData = yData.reshape(newHeight, newWidth)
    uData = uData.reshape(newHeight, newWidth)
    vData = vData.reshape(newHeight, newWidth)
    pixelSample = 0
    xCo = 0
    yCo = 0
    maxPixelSample = ((height-patchH) * width) + (width-patchW)
    #print("maxPixelSample: {}".format(maxPixelSample))
    #print("newHeight: {} and {}".format(newHeight, (1 + newHeight - patchH)))
    while yCo < (1 + newHeight - patchH):
        #print("Taking sample from: ({}, {})".format(xCo, yCo))
        patchY = yData[yCo:(yCo + patchH), xCo:(xCo + patchW)]
        patchU = uData[yCo:(yCo + patchH), xCo:(xCo + patchW)]
        patchV = vData[yCo:(yCo + patchH), xCo:(xCo + patchW)]

        #print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
        yuv = np.concatenate((np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
        #print("patch dims: {}".format(yuv.shape))
        yuv = yuv.flatten()
        patchesList.append(yuv)
        numPatches = numPatches + 1


        xCo = xCo + xstride
        if xCo > (1 + newWidth - patchW):
            xCo = 0
            yCo = yCo + ystride
        #print("numPatches: {}".format(numPatches))
        #print(yCo)


    patches_array = np.array(patchesList)
    #np.random.shuffle(patches_array)
    return patches_array

if __name__ == "__main__":

    #encodeYUV.encodeAWholeFolderAsH264("/Users/pam/Documents/data/h264/", takeAll=True)
    #quit()
    dataDir = "/Users/pam/Documents/data/h264/"
    intraDir = "/Users/pam/Documents/data/h264/intra"
    nonintraDir = "/Users/pam/Documents/data/h264/intra_not"

    outputDir = "{}patches/".format(dataDir)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    filenames_Is  = showBitstreamInfo.createFileList(intraDir, takeAll = True, format='.h264', shuffle=False)
    filenames_IPs = showBitstreamInfo.createFileList(nonintraDir, takeAll = True, format='.h264', shuffle=False)

    #filenames = showBitstreamInfo.createFileList(intraDir, takeAll = False, format='.h264', desiredNamePart = '720', shuffle=False)
    #filenames = showBitstreamInfo.createFileList(nonintraDir, takeAll = False, format='.h264', desiredNamePart = '720', shuffle=False)

    #The intersection of the lists
    for filename1 in filenames_Is:
        head, tail = os.path.split(filename1)
        foundIt = False
        for filename2 in filenames_IPs:
            if tail in filename2:
                foundIt = True
        if not foundIt:
            filenames_Is.remove(filename1)

    for filename1 in filenames_IPs:
        head, tail = os.path.split(filename1)
        foundIt = False
        for filename2 in filenames_Is:
            if tail in filename2:
                foundIt = True
        if not foundIt:
            filenames_IPs.remove(filename1)

    print(filenames_Is)
    print("and")
    print(filenames_IPs)

    for filename in filenames_IPs:
        relFilename = os.path.relpath(filename, nonintraDir)
        nonintraFilename = filename
        intraFilename = os.path.join(intraDir, relFilename)
        print(intraFilename)
        print(nonintraFilename)
        basefilename, ext = os.path.splitext(filename)
        f, width, height = encodeYUV.getFile_Name_Width_Height(basefilename)
        splitTraceFileIntoComponentFiles(intraFilename, width, height)
        splitTraceFileIntoComponentFiles(nonintraFilename, width, height)


        #filename = "/Users/pam/Documents/data/h264/carphone_qcif_q0.h264"
        #filename = "/Users/pam/Documents/data/DeepFakes/creepy1.h264"
        #splitTraceFileIntoComponentFiles(filename)

        # need two files: the yuv file and mbmode file
        #YUVFilename = "/Users/pam/Documents/data/h264/carphone_qcif_q0.yuv"
        #basefilename, ext = os.path.splitext(YUVFilename)
        MbModeFilename = "{}.mbmode".format(basefilename)
        mbwidth = width/16
        mbheight = height/16

        frameSize = int(width * height)
        mbframeSize = int(mbwidth * mbheight)
        frameNumber = 0


        with open(MbModeFilename, "rb") as f:
            mbModes = f.read()

        mbModes = bytearray(mbModes)
        patchesList = []
        numPatches = 0
        firstFrame = 2 # Don't want to start at 0
        firstMB = firstFrame * mbframeSize
        patches_intra = extractPatchesAndReturn(intraFilename, firstFrame, width, height)
        patches_nonintra = extractPatchesAndReturn(nonintraFilename, firstFrame, width, height)
        currentFrameNo = 2
        print(patches_nonintra.shape)

        for _idx, mbMode in enumerate(mbModes[firstMB:]):
            idx = _idx + firstMB
            print("Mb idx: {}, value: {}".format(idx, mbMode))
            mbNo = idx % mbframeSize
            frameNo = int((idx-mbNo) / mbframeSize)
            print("frameNo: {} vs currentFrameNo: {}".format(frameNo, currentFrameNo))
            if frameNo != currentFrameNo:
                patches_intra = extractPatchesAndReturn(intraFilename, frameNo, width, height)
                patches_nonintra = extractPatchesAndReturn(nonintraFilename, frameNo, width, height)
                currentFrameNo = frameNo


            label = mbMode
            # The intra patches will come from the intra files, so only look for inter patches
            if label != 1:
                yuv = patches_nonintra[mbNo]
                datayuv = np.concatenate((np.array([label]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1

                intralabel = 0
                yuv = patches_intra[mbNo]
                datayuv = np.concatenate((np.array([intralabel]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1

        patches_array = np.array(patchesList)
        print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
        ############## Here's where you name the files!!!!###########
        patchNumber = numPatches // 1000
        outFileName = "patches_{}.bin".format(patchNumber)
        outFileName = os.path.join(outputDir, outFileName)
        functions.appendToFile(patches_array, outFileName)
        patchesList = []



