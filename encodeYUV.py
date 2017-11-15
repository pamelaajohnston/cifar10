import random
import os
import functions
import shlex, subprocess
import numpy as np
import time

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
    ['cif', 352, 288],
    ['sif', 352, 240],
    ['720p', 1280, 720],
    ['1080p', 1920, 1080]
]

quants = [0, 7, 14, 21, 28, 35, 42, 49]

def createFileList(myDir):
    fileList = []
    takeAll = False
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(myDir):
        for filename in filenames:
            if filename.endswith('.yuv'):
                if takeAll or '_cif' in filename:
                    fileName = os.path.join(myDir, dirName, filename)
                    baseFileName, ext = os.path.splitext(filename)
                    #print("The filename is {} baseFileName {}".format(fileName, baseFileName))

                    for fileSize in fileSizes:
                        if fileSize[0] in fileName:
                            tuple = [fileName, fileSize[1], fileSize[2]]
                            fileList.append(tuple)
                            break

    #hacky hack
    #fileList = [['/Volumes/LaCie/data/yuv_quant_noDeblock/quant_0/mobile_cif_q0.yuv', 352, 288],]
    random.shuffle(fileList)
    #print(fileList)
    return fileList

def encodeAWholeFolderAsH264(myDir):
    fileList = createFileList()

    for quant in quants:
        # make a directory
        dirName = "quant_{}".format(quant)
        dirName = os.path.join(myDir, dirName)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        for entry in fileList:
            filename = entry[0]
            baseFileName = os.path.basename(filename)
            baseFileName, ext = os.path.splitext(baseFileName)
            baseFileName = "{}/{}".format(dirName, baseFileName)

            #print("The baseFileName is: {}".format(baseFileName))
            width = entry[1]
            height = entry[2]
            print("width: {} height: {} filename:{}".format(entry[1], entry[2], entry[0]))
            h264Filename = "{}_q{}.h264".format(baseFileName, quant)
            compYuvFilename = "{}_q{}.yuv".format(baseFileName, quant)
            isize, psize, bsize = functions.compressFile(x264, filename, width, height, quant, h264Filename, compYuvFilename)

import re

def getQuant(inputString):
    print(inputString)
    m = re.search('(?<=quant_)(\d{1,2})', inputString)
    quant = int(m.group(0))
    quant = quant/7
    return quant

def extractPatches(fileList, outFileBaseName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3):
    patchesList = []
    numPatches = 0
    filesWritten = 0
    for file in fileList:
        filename = file[0]
        width = file[1]
        height = file[2]
        frameSize = height * width * 3/2 # for i420 only
        quant = getQuant(filename)
        with open(filename, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        numFrames = allTheData.shape[0] / frameSize

        allTheData = allTheData.reshape(numFrames, frameSize)
        frameSample = 0
        lastFrame = numFrames
        if width == 1280:
            frameSample = 2
            lastFrame = numFrames - 2

        while frameSample < lastFrame:
            frameData = allTheData[frameSample]
            ySize = width*height
            uvSize = (width*height)/4
            yData = frameData[0:ySize]
            uData = frameData[ySize:(ySize+uvSize)]
            vData = frameData[(ySize+uvSize):(ySize+uvSize+uvSize)]
            #print("yData shape: {}".format(yData.shape))
            #print("uData shape: {}".format(uData.shape))
            #print("vData shape: {}".format(vData.shape))
            yData = yData.reshape(height, width)
            uData = uData.reshape(height/2, width/2)
            vData = vData.reshape(height/2, width/2)
            pixelSample = 0
            xCo = 0
            yCo = 0
            maxPixelSample = ((height-patchDim) * width) + (width-patchDim)
            #print("maxPixelSample: {}".format(maxPixelSample))
            while yCo < (height - patchDim):
                #print("Taking sample from: ({}, {})".format(xCo, yCo))
                patchY = yData[yCo:(yCo+patchDim), xCo:(xCo+patchDim)]
                patchU = uData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchU = np.repeat(patchU, 2, axis=0)
                patchU = np.repeat(patchU, 2, axis=1)
                patchV = vData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchV = np.repeat(patchV, 2, axis=0)
                patchV = np.repeat(patchV, 2, axis=1)

                #print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
                yuv = np.concatenate((np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
                #print("patch dims: {}".format(yuv.shape))
                yuv = yuv.flatten()
                datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1


                xCo = xCo + patchStride
                if xCo > (width - patchDim):
                    xCo = 0
                    yCo = yCo + patchStride
                pixelSample = (yCo*width) + xCo
                #print("numPatches: {}".format(numPatches))

                if numPatches > 9999:
                    patches_array = np.array(patchesList)
                    print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
                    np.random.shuffle(patches_array)
                    outFileName = "{}_{}.bin".format(outFileBaseName, filesWritten)
                    functions.appendToFile(patches_array, outFileName)
                    filesWritten += 1
                    patchesList = []
                    numPatches = 0

            frameSample = frameSample + frameSampleStep

    patches_array = np.array(patchesList)
    print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
    np.random.shuffle(patches_array)
    outFileName = "{}_{}.bin".format(outFileBaseName, filesWritten)
    functions.appendToFile(patches_array, outFileName)

def extractPatches_byQuant(fileList, outFileBaseName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3):
    patchesList = []
    numPatches = 0
    filesWritten = 0
    for file in fileList:
        filename = file[0]
        width = file[1]
        height = file[2]
        frameSize = height * width * 3/2 # for i420 only
        quant = getQuant(filename)
        with open(filename, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        numFrames = allTheData.shape[0] / frameSize

        allTheData = allTheData.reshape(numFrames, frameSize)
        frameSample = 0
        lastFrame = numFrames
        if width == 1280:
            frameSample = 2
            lastFrame = numFrames - 2

        while frameSample < lastFrame:
            frameData = allTheData[frameSample]
            ySize = width*height
            uvSize = (width*height)/4
            yData = frameData[0:ySize]
            uData = frameData[ySize:(ySize+uvSize)]
            vData = frameData[(ySize+uvSize):(ySize+uvSize+uvSize)]
            #print("yData shape: {}".format(yData.shape))
            #print("uData shape: {}".format(uData.shape))
            #print("vData shape: {}".format(vData.shape))
            yData = yData.reshape(height, width)
            uData = uData.reshape(height/2, width/2)
            vData = vData.reshape(height/2, width/2)
            pixelSample = 0
            xCo = 0
            yCo = 0
            maxPixelSample = ((height-patchDim) * width) + (width-patchDim)
            #print("maxPixelSample: {}".format(maxPixelSample))
            while yCo < (height - patchDim):
                #print("Taking sample from: ({}, {})".format(xCo, yCo))
                patchY = yData[yCo:(yCo+patchDim), xCo:(xCo+patchDim)]
                patchU = uData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchU = np.repeat(patchU, 2, axis=0)
                patchU = np.repeat(patchU, 2, axis=1)
                patchV = vData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchV = np.repeat(patchV, 2, axis=0)
                patchV = np.repeat(patchV, 2, axis=1)

                #print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
                yuv = np.concatenate((np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
                #print("patch dims: {}".format(yuv.shape))
                yuv = yuv.flatten()
                datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1


                xCo = xCo + patchStride
                if xCo > (width - patchDim):
                    xCo = 0
                    yCo = yCo + patchStride
                pixelSample = (yCo*width) + xCo
                #print("numPatches: {}".format(numPatches))

                patches_array = np.array(patchesList)
                print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
                outFileName = "{}_{}.bin".format(outFileBaseName, quant)
                functions.appendToFile(patches_array, outFileName)
                filesWritten += 1
                patchesList = []
                numPatches = 0

            frameSample = frameSample + frameSampleStep


def main(argv=None):
    print("Butcher the test files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_test'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_train'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName)

if __name__ == "__main__":
    main()