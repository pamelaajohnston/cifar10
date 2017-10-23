import random
import os
import shlex, subprocess
import numpy as np
import time

# You've got loads of individually compressed datasets and you want to munge them all together to make one big one
# but also split the big one suitably into small batch files.

#datastats
numberLabels = 10
height = 32
width = 32
channels = 3
labelSize = 1
recordDataSize = channels * width * height
recordSize = recordDataSize + labelSize



datasrcdir = '/Users/pam/Documents/data/CIFAR-10_munge'
dataDirs = ['cifar-10-batches-bin_yuv', 'cifar-10-batches-bin_q10_f0', 'cifar-10-batches-bin_q25_f0', 'cifar-10-batches-bin_q37_f0', 'cifar-10-batches-bin_q41_f0', 'cifar-10-batches-bin_q46_f0', 'cifar-10-batches-bin_q50_f0']
qps = [0, 10, 25, 37, 41, 46, 50]
datadstdir = '/Users/pam/Documents/data/CIFAR-10_munge/cifar-10-batches-bin_all2'
mapFileName = 'map_CIFAR-10_munge_all2.txt'
readMap = True
trainFileNames = ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin','data_batch_4.bin','data_batch_5.bin']
#trainFileNames = ['data_batch_1.bin',]
testFileNames = ['test_batch.bin', ]

recordsPerFile = 10000 # 10k images per batch file in CIFAR-10
numTrainFiles = len(dataDirs) * len(trainFileNames)
numTestFiles = len(dataDirs) * len(testFileNames)
totalTrainRecords = recordsPerFile * numTrainFiles
totalTestRecords = recordsPerFile * numTestFiles



import errno
def main(argv=None):

    print("{} train files, {} records per train file, {} bytes per record".format(numTrainFiles, recordsPerFile, recordSize))
    # adding one to recordSize because I'm adding an extra label to show quant
    #bigFatTrainArray = np.empty((totalTrainRecords, recordSize+1))
    bigFatTrainArray = np.empty((totalTrainRecords, recordSize))

    for dataDirIdx, dataDir in enumerate(dataDirs):
        for trainFileNameIdx, trainFileName in enumerate(trainFileNames):
            data_folder = os.path.join(datasrcdir, dataDir, trainFileName)
            with open(data_folder, "rb") as f:
                fileData = np.fromfile(f, 'u1')
                num_images = fileData.shape[0] / recordSize
                fileData = fileData.reshape(num_images, recordSize)
                startIdx = ((dataDirIdx * len(trainFileNames)) + trainFileNameIdx) * recordsPerFile
                endIdx = startIdx + num_images
                print("startIdx = {} * {} * {}".format(dataDirIdx, trainFileNameIdx, recordsPerFile))
                print("Putting the data in from index {} to {}".format(startIdx, endIdx))
                aSecondLabel = np.full((num_images), qps[dataDirIdx])
                print(fileData[0, 0:28])
                #fileData = np.insert(fileData, 0, aSecondLabel, axis=1)
                print(fileData[0, 0:28])
                bigFatTrainArray[startIdx:endIdx, :] = fileData

    np.random.shuffle(bigFatTrainArray)
    print("Shape of train array: {}".format(bigFatTrainArray.shape))

    # now take the records 10k at a time and put them in files
    for i in range(numTrainFiles):
        startIdx = recordsPerFile * i
        endIdx = startIdx + recordsPerFile
        littleArray = bigFatTrainArray[startIdx:endIdx, :]
        littleArray = np.asarray(littleArray, 'u1')
        littleArray = bytearray(littleArray[:, :])

        mungedTrainFileName = "data_batch_{}.bin".format((i+1))
        mungedTrainFileName = os.path.join(datadstdir, mungedTrainFileName)
        print("Putting it {} from {} to {}".format(mungedTrainFileName, startIdx, endIdx))

        with open(mungedTrainFileName, 'wb') as f:
            f.write(littleArray)

    print("{} test files, {} records per train file, {} bytes per record".format(numTestFiles, recordsPerFile, recordSize))
    # adding one to recordSize because I'm adding an extra label to show quant
    #bigFatTestArray = np.empty((totalTestRecords, recordSize+1))
    bigFatTestArray = np.empty((totalTestRecords, recordSize))

    for dataDirIdx, dataDir in enumerate(dataDirs):
        for testFileNameIdx, testFileName in enumerate(testFileNames):
            data_folder = os.path.join(datasrcdir, dataDir, testFileName)
            with open(data_folder, "rb") as f:
                fileData = np.fromfile(f, 'u1')
                num_images = fileData.shape[0] / recordSize
                fileData = fileData.reshape(num_images, recordSize)
                startIdx = ((dataDirIdx * len(testFileNames)) + testFileNameIdx) * recordsPerFile
                endIdx = startIdx + num_images
                print("startIdx = {} * {} * {}".format(dataDirIdx, testFileNameIdx, recordsPerFile))
                print("Putting the data in from index {} to {}".format(startIdx, endIdx))
                aSecondLabel = np.full((num_images), qps[dataDirIdx])
                #fileData = np.insert(fileData, 0, aSecondLabel, axis=1)
                bigFatTestArray[startIdx:endIdx, :] = fileData

    np.random.shuffle(bigFatTestArray)

    # now take the records 10k at a time and put them in files
    for i in range(numTestFiles):
        startIdx = recordsPerFile * i
        endIdx = startIdx + recordsPerFile
        littleArray = bigFatTestArray[startIdx:endIdx, :]
        littleArray = np.asarray(littleArray, 'u1')
        littleArray = bytearray(littleArray[:, :])
        mungedTestFileName = "test_batch_{}.bin".format((i+1))
        mungedTestFileName = os.path.join(datadstdir, mungedTestFileName)
        print("Putting it {} from {} to {}".format(mungedTestFileName, startIdx, endIdx))

        with open(mungedTestFileName, 'wb') as f:
            f.write(littleArray)


if __name__ == "__main__":
    main()