import random
import os
import shlex, subprocess
import numpy as np
import time

# You've got a train and a test set but you want a different split
# train data is a bunch of binary files, all with the same format: data (picture), label
# Theoretically it works on more than one file but I've only developed and tested on a dataset that is one train and one test file.

datasrcdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_binary_yuv'
datadstdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_refactored_yuv'
trainFileNames = ['train_X.bin', ]
testFileNames = ['test_X.bin', ]
percentageTrain = 0.8

#datastats
numberLabels = 10
height = 96
width = 96
channels = 3
RecordDataSize = channels * width * height
RecordSize = RecordDataSize + 1 # because 1 byte is the label



#def editData(srcdir, dstdir, trainFileNames, testFileNames, numLabels, split=0.8):
def normaliseLabels(srcFile, dstFile):
    with open(srcFile, "rb") as f:
        allTheData = np.fromfile(f, 'u1')
        num_cases = allTheData.shape[0] / RecordSize
        print("number of records = {}".format(num_cases))
        allTheData = allTheData.reshape(num_cases, RecordSize)

        for i in range(0, num_cases):
            allTheData[i][0] = allTheData[i][0] - 1

        allTheData = np.asarray(allTheData, 'u1')
        allTheData = bytearray(allTheData)

        if not os.path.exists(os.path.dirname(dstFile)):
            try:
                os.makedirs(os.path.dirname(dstFile))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(dstFile, 'wb') as f:
            f.write(allTheData)



datasrcdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_binary_yuv'
datadstdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_refactored_yuv'
mapFileName = 'map.txt'
readMap = True
trainFileNames = ['train_X.bin', ]
testFileNames = ['test_X.bin', ]
percentageTrain = 0.8

basesrc = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_binary_'
basedst = '/Volumes/LaCie/stl10_binary/tinytest/refactored_norm/datasetstl10_binary_'

#basesrc = '/Volumes/LaCie/stl10_binary/constantQuant/datasetstl10_binary_'
basesrc = '/Volumes/LaCie/stl10_binary/constantQuant/refactored/datasetstl10_binary_'
basedst = '/Volumes/LaCie/stl10_binary/constantQuant/refactored_anew/datasetstl10_binary_'


import errno
def main(argv=None):

    # generate the datadirs
    saveFrames = (0, 2, 3, 6)
    quants = (10, 25, 37, 41, 46, 50)
    myDatadirs = ["yuv", "y_quv", "y_squv", "interlaced"]
    for quant in quants:
        for idx, frame in enumerate(saveFrames):
            name = "q{}_f{}".format(quant, saveFrames[idx])
            myDatadirs.append(name)

    print(myDatadirs)

    for dataDir in myDatadirs:
        datasrcdir = "{}{}".format(basesrc, dataDir)
        datadstdir = "{}{}".format(basedst, dataDir)
        print("src {} to dst {}".format(datasrcdir, datadstdir))

        for trainFileName in trainFileNames:
            s = os.path.join(datasrcdir, trainFileName)
            d = os.path.join(datadstdir, trainFileName)
            normaliseLabels(s, d)

        for testFileName in testFileNames:
            s = os.path.join(datasrcdir, testFileName)
            d = os.path.join(datadstdir, testFileName)
            normaliseLabels(s, d)



if __name__ == "__main__":
    main()