import random
import os
import shlex, subprocess
import numpy as np
import time

# You've got a train and a test set but you want a different split
# train data is a bunch of binary files, all with the same format: data (picture), label
# Theoretically it works on more than one file but I've only developed and tested on a dataset that is one train and one test file.

def turnFileIntoDictionary(fileName):
    dataArray = []
    names = []
    quants = [0]

    with open(fileName, 'r') as f:
        allTheData = f.read()

    textLines = allTheData.splitlines()

    for line in textLines:
        # print("Begin")
        # print(line)
        words = line.split()
        if words[0] == 'Evaluating':
            #trainedOn = name
            trainedOn = words[3].split('-')
            trainedOn = trainedOn[0]
            testedOn = words[1]
            precision = words[-1]

            trainedQuant = 0
            trainedFrame = -1
            temp = list(trainedOn)
            if trainedQuant == 'yuv':
                trainedQuant = 0
            if temp[0] == 'q':
                theNumbers = (temp[1], temp[2])
                theString = ''.join(theNumbers)
                trainedQuant = int(theString)
                if trainedQuant not in quants:
                    quants.append(trainedQuant)
                if temp[4] == 'f':
                    trainedFrame = int(temp[5])

            testedQuant = 0
            testedFrame = -1
            temp = list(testedOn)
            if testedOn == 'yuv':
                testedFrame = 0
            if temp[0] == 'q':
                theNumbers = (temp[1], temp[2])
                theString = ''.join(theNumbers)
                testedQuant = int(theString)
                if temp[4] == 'f':
                    testedFrame = int(temp[5])

            dataEntry = [trainedOn, testedOn, trainedQuant, trainedFrame, testedQuant, testedFrame, precision]
            # print(dataEntry)
            dataArray.append(dataEntry)
            if trainedOn not in names:
                # print("adding: {}".format(trainedOn))
                names.append(trainedOn)

    return dataArray

#datadir = '/Users/pam/Documents/data/stl10/local_firstRepeatingEffort/'
#fileBaseName = 'results_yuv_network1_try'
datadir = '/Users/pam/Documents/data/stl10/'
fileBaseName = 'results_nw1.txt'
averagesFileName = 'results_nw1_averages.txt'
averagesFileName = 'results_nw1_maxes.txt'
fileBaseName = 'results_nwa3.txt'
averagesFileName = 'results_nwa3_averages.txt'
#averagesFileName = 'results_nwa3_maxes.txt'
numTrys = 3

def main(argv=None):
    print("datadir = {}".format(datadir))
    print("fileBaseName = {}".format(fileBaseName))
    #print("numTrys = {}".format(numTrys))

    allResults = []

    #for i in range(1, (numTrys+1)):
        #fileName = "{}{}.txt".format(fileBaseName, i)
        #fileName = os.path.join(datadir, fileName)
    fileName = os.path.join(datadir, fileBaseName)
    print(fileName)
    mydict = turnFileIntoDictionary(fileName)
    #mydict = np.asarray(mydict)
    #print(mydict)
    allResults.append(mydict)

    allResults = np.asarray(allResults)
    allResults = allResults.flatten()
    allResults = np.reshape(allResults, (-1, 7))
    print(allResults)

    trainedList = []
    testedList = []

    for line in allResults:
        if line[0] not in trainedList:
            trainedList.append(line[0])
        if line[1] not in testedList:
            testedList.append(line[1])

    print(trainedList)
    print(testedList)

    resultsMatrix = [] # trained, tested, avgPrecision, varPrecision

    for trainedOn in trainedList:
        for testedOn in testedList:
            precisions = []
            for line in allResults:
                if line[0] == trainedOn and line[1] == testedOn:
                    if line[-1] != 0:
                        precisions.append(float(line[-1]))
                    else:
                        print("Missed a test: trainedOn {} testedOn {}".format(trainedOn, testedOn))
            print("{}, {} {}".format(trainedOn, testedOn, precisions))
            avgPrecision = np.mean(precisions)
            varPrecision = np.var(precisions)
            #avgPrecision = np.min(precisions)
            result = [trainedOn, testedOn, avgPrecision, varPrecision]
            resultsMatrix.append(result)


    print(resultsMatrix)

    resFName= os.path.join(datadir, averagesFileName)
    with open(resFName, 'w') as f:
        for result in resultsMatrix:
            f.write("Evaluating {} on {}-trained network: {} \n".format(result[1], result[0], result[2]))
            print("Evaluating {} on {}-trained network: {} with var {}".format(result[1], result[0], result[2], result[3]))



if __name__ == "__main__":
    main()