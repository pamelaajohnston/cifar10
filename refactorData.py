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
mapFileName = 'map.txt'
readMap = True
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


def refactorFromMap(srcdir, dstdir, mapFileName, trainFileNames, testFileNames):
    #with open(mapFileName, "r") as f:
    #    mapData = np.fromfile(f)
    #mapData = mapData.reshape(-1, 3)

    openSrcFileName = ""
    openDstFileName = ""
    openDstFile = 0


    import csv
    with open(mapFileName, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(' * '.join(row))
            #words = row.split(' ')

            try:
                idx = int(row[1])
                srcFile = os.path.join(srcdir, row[0])
                if openSrcFileName == "" or openSrcFileName != srcFile:
                    openSrcFileName = srcFile
                    print("Opening {}".format(openSrcFileName))
                    with open(openSrcFileName, 'rb') as openSrcFile:
                        openSrcFileData = np.fromfile(openSrcFile, 'u1')
                        num_cases_per_batch = openSrcFileData.shape[0] / RecordSize
                        print("number of records = {}".format(num_cases_per_batch))
                        openSrcFileData = openSrcFileData.reshape(num_cases_per_batch, RecordSize)

                dstFile = os.path.join(dstdir, row[2])
                if openDstFileName == "" or openDstFileName != dstFile:
                    openDstFileName = dstFile
                    print("Opening destination {}".format(openDstFileName))
                    #if openDstFileName == "":
                    #    # copy the original file across...
                    #    origFile = os.path.join(srcdir, row[2])
                    #    from shutil import copyfile
                    #    copyfile(origFile, dstFile)
                    if openDstFile:
                        openDstFile.close()
                    if not os.path.exists(os.path.dirname(openDstFileName)):
                        try:
                            os.makedirs(os.path.dirname(openDstFileName))
                        except OSError as exc:  # Guard against race condition
                            if exc.errno != errno.EEXIST:
                                raise
                    openDstFile = open(openDstFileName, 'ab')

                copyLine = openSrcFileData[idx]
                copyLine = np.asarray(copyLine, 'u1')
                copyLine = bytearray(copyLine)
                openDstFile.write(copyLine)
                print("Taking from {} line {} putting to {}".format(srcFile, idx, dstFile))
            except:
                idx = row[1]
                print("The row said {})".format(row))
        openDstFile.close()

    return True

def refactor(srcdir, dstdir, mapFileName, trainFileNames, testFileNames, numLabels, split=0.8):
    # First calculate the total number of records
    mapFile = open(mapFileName, 'w')
    mapFile.write("old_batchfile old_idx new_batchfile\n")

    totalTrainRecords = 0
    for trainFileName in trainFileNames:
        fp = os.path.join(srcdir, trainFileName)
        statinfo = os.stat(fp)
        totalTrainRecords = totalTrainRecords + (statinfo.st_size/RecordSize)

    totalTestRecords = 0
    for testFileName in testFileNames:
        fp = os.path.join(srcdir, testFileName)
        statinfo = os.stat(fp)
        totalTestRecords = totalTestRecords + (statinfo.st_size/RecordSize)

    curSplit = (float)(totalTrainRecords / (float)(totalTrainRecords + totalTestRecords))

    print("Train records: {}, Test records {}, current split: {}".format(totalTrainRecords, totalTestRecords, curSplit))

    transRecordNum = 0

    if (curSplit < split):
        transRecordNum = ((totalTrainRecords + totalTestRecords) * split) - totalTrainRecords
        print("Need to transfer test data to train data: {} records".format(transRecordNum))
        donerFileNames = testFileNames
        receiverFileNames = trainFileNames
    else:
        transRecordNum = ((totalTrainRecords + totalTestRecords) * (1-split)) - totalTestRecords
        print("Need to transfer train data to test data: {} records".format(transRecordNum))
        donerFileNames = trainFileNames
        receiverFileNames = testFileNames

    #How many from each label from each file?
    takeNum = int(round((transRecordNum / (len(donerFileNames) * numberLabels)) + 0.5))
    print("We're taking {} from each file's lable".format(takeNum))

    takenData = []
    leftData = []

    # You'll need to sort the data so we don't screw up the percentages of labels in the train/test data
    for fileName in donerFileNames:
        fp = os.path.join(srcdir, fileName)
        with open(fp, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        num_cases_per_batch = allTheData.shape[0] / RecordSize
        print("number of records = {}".format(num_cases_per_batch))
        allTheData = allTheData.reshape(num_cases_per_batch, RecordSize)
        indices = np.arange(num_cases_per_batch)
        indices = indices.reshape(num_cases_per_batch, 1)
        allTheData = np.concatenate((allTheData, indices), axis=1)
        print("allTheData: {}".format(allTheData[0]))
        print("allTheData: {}".format(allTheData[1]))
        print("allTheData: {}".format(allTheData[2]))
        print("Shape of array: {}".format(allTheData.shape))
        #sortedData = np.sort(allTheData, axis=0)
        indexes = np.argsort(allTheData[:, 0])
        sortedData = allTheData[indexes]
        print("sortedData: {}".format(sortedData[0]))
        print("sortedData: {}".format(sortedData[1]))
        print("sortedData: {}".format(sortedData[2]))
        print("...")
        print("sortedData: {}".format(sortedData[-3]))
        print("sortedData: {}".format(sortedData[-2]))
        print("sortedData: {}".format(sortedData[-1]))
        #for idx, id in enumerate(indexes):
        #    mapFile.write("{} {} {} {} \n".format(fileName, id, "newfile", idx))

        sarrays = np.split(sortedData, np.where(np.diff(sortedData[:, 0]))[0] + 1)
        print("Now we have {} arrays".format(len(sarrays)))

        leftData = []
        takenData = []
        takenIndices = []
        leftIndices = []
        for myArray in sarrays:
            takenData.append(myArray[0:takeNum, :])
            #takenIndices.append(myArray[0:takeNum, -1])
            leftData.append(myArray[takeNum:, :])
            #leftIndices.append(myArray[takeNum:, -1])

        #put the left data back into the file and create a new file for the new data
        leftData = np.asarray(leftData)
        leftData = leftData.reshape(leftData.shape[0] * leftData.shape[1], leftData.shape[2])
        print("Shape of leftData: {}".format(leftData.shape))
        takenData = np.asarray(takenData)
        takenData = takenData.reshape(takenData.shape[0] * takenData.shape[1], takenData.shape[2])
        print("Shape of takenData: {}".format(takenData.shape))

        print("bs leftData: {}".format(leftData[0]))
        print("bs leftData: {}".format(leftData[1]))
        print("bs leftData: {}".format(leftData[2]))
        leftIndexes = np.argsort(leftData[:, RecordSize])
        leftData = leftData[leftIndexes]
        print("leftData: {}".format(leftData[0]))
        print("leftData: {}".format(leftData[1]))
        print("leftData: {}".format(leftData[2]))
        leftIndexes = leftData[:, RecordSize]
        print("This is the original leftIndexes:{}".format(leftIndexes))

        print("bs takenData: {}".format(takenData[0]))
        print("bs takenData: {}".format(takenData[1]))
        print("bs takenData: {}".format(takenData[2]))
        takenIndexes = np.argsort(takenData[:, RecordSize])
        takenData = takenData[takenIndexes]
        print("takenData: {}".format(takenData[0]))
        print("takenData: {}".format(takenData[1]))
        print("takenData: {}".format(takenData[2]))
        takenIndexes = takenData[:, RecordSize]

        leftData = np.asarray(leftData, 'u1')
        takenData = np.asarray(takenData, 'u1')

        leftData = bytearray(leftData[:, :RecordSize])
        takenData = bytearray(takenData[:, :RecordSize])

        leftName = os.path.join(dstdir, fileName)
        takenName = os.path.join(dstdir, receiverFileNames[0])
        srcRxFileName = os.path.join(srcdir, receiverFileNames[0])
        sizeOrigTakenData = 0
        with open(srcRxFileName, 'rb') as f:
            origTakenData = np.fromfile(f, 'u1')
            sizeOrigTakenData = origTakenData.shape[0] / RecordSize
            origTakenData = bytearray(origTakenData)
        with open(leftName, 'wb') as f:
            f.write(leftData)
        with open(takenName, 'wb') as f:
            f.write(origTakenData)
            f.write(takenData)

        # We just did a copy across of the original file, write that into the map file
        for x in np.arange(sizeOrigTakenData):
            mapFile.write("{} {} {}\n".format(receiverFileNames[0], x, receiverFileNames[0]))

        #Preparing the map file (I want the source indices in order)
        dummy = np.full((leftIndexes.shape[0],1), 0)
        print("The shape of the left index thingy: {}".format(leftIndexes.shape))
        leftIndexes = leftIndexes.reshape(leftIndexes.shape[0], 1)
        leftIndexes = np.concatenate((dummy, leftIndexes, dummy), axis=1)
        print("leftIndexes: {}".format(leftIndexes[0]))
        print("leftIndexes: {}".format(leftIndexes[1]))
        print("leftIndexes: {}".format(leftIndexes[2]))

        dummy = np.full((takenIndexes.shape[0],1), 0)
        dummyLeft = np.full((takenIndexes.shape[0],1), 1)
        takenIndexes = takenIndexes.reshape(takenIndexes.shape[0], 1)
        takenIndexes = np.concatenate((dummy, takenIndexes, dummyLeft), axis=1)
        print("takenIndexes: {}".format(takenIndexes[0]))
        print("takenIndexes: {}".format(takenIndexes[1]))
        print("takenIndexes: {}".format(takenIndexes[2]))

        allTheIndexes = np.concatenate((leftIndexes, takenIndexes), axis=0)
        print("bs allTheIndexes: {}".format(allTheIndexes[0]))
        print("bs allTheIndexes: {}".format(allTheIndexes[1]))
        print("bs allTheIndexes: {}".format(allTheIndexes[2]))
        allTheIndexes = allTheIndexes[np.argsort(allTheIndexes[:, 1])]
        print("allTheIndexes: {}".format(allTheIndexes[0]))
        print("allTheIndexes: {}".format(allTheIndexes[1]))
        print("allTheIndexes: {}".format(allTheIndexes[2]))

        listOfNames = [fileName, receiverFileNames[0]]

        for line in allTheIndexes:
            mapFile.write("{} {} {}\n".format(listOfNames[int(line[0])], int(line[1]), listOfNames[int(line[2])]))

    mapFile.close()
    return True



datasrcdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_binary_yuv'
datadstdir = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_refactored_yuv'
mapFileName = 'map.txt'
readMap = True
trainFileNames = ['train_X.bin', ]
testFileNames = ['test_X.bin', ]
percentageTrain = 0.8

basesrc = '/Volumes/LaCie/stl10_binary/tinytest/datasetstl10_binary_'
basedst = '/Volumes/LaCie/stl10_binary/tinytest/refactored/datasetstl10_binary_'

basesrc = '/Volumes/LaCie/stl10_binary/constantQuant/datasetstl10_binary_'
basedst = '/Volumes/LaCie/stl10_binary/constantQuant/refactored/datasetstl10_binary_'


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


        if readMap:
            print("We're reading from map file {}".format(mapFileName))
            refactorFromMap(datasrcdir, datadstdir, mapFileName, trainFileNames=trainFileNames, testFileNames=testFileNames)
        else:
            print("We're reading from map file {}".format(mapFileName))
            refactor(datasrcdir, datadstdir, mapFileName, trainFileNames=trainFileNames, testFileNames=testFileNames, numLabels=numberLabels)

    #logfile = "{}/log.txt".format(dstdatasetdir)
    #log = open(logfile, 'w')
    #log.write("machine: {}".format(machine))


if __name__ == "__main__":
    main()