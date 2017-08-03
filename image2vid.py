from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import socket
import readConfig
import functions
import pickle
import time
#import re
#from cStringIO import StringIO
#import shutil as shutil


def processSingleImage(data, w, h, x264, saveFrames, quants, logFile, srcFile, label, isRGB=True):
    if isRGB:
        datayuv = functions.planarRGB_2_planarYUV(data, w, h)
    else:
        datayuv = data.copy()
    dataQyuv = datayuv.copy()
    dataQyuv = functions.quantiseUV(dataQyuv, w, h)
    
    dataSQyuv = datayuv.copy()
    dataSQyuv = functions.YUV444_2_YUV420(dataSQyuv, w, h)
    dataSQyuv = functions.YUV420_2_YUV444(dataSQyuv, w, h)
    
    dataInterlaced = datayuv.copy()
    dataInterlaced = functions.interlace(dataInterlaced, w, h)
    
    frame_dict = {
        'yuv'   : datayuv,
        'y_quv' : dataQyuv,
        'y_squv': dataSQyuv,
        'interlaced': dataInterlaced
    }
    getVideoCompressedImages(datayuv, w, h, frame_dict, x264, saveFrames, quants, logFile, srcFile, label)

    return frame_dict


def getVideoCompressedImages(data, w, h, frame_dict, x264, saveFrames, quants, log, srcFile, label):
    numframes = 7
    offset = 8
    width = w
    height = h
    dstw = width + (2*offset)
    dsth = height + (2*offset)
    genyuv = "sequence.yuv"
    gen264 = "seq.264"
    decompyuv = "sequout.yuv"
    #print "Width {} Height {} dstw {} dsth {}".format(width, height, dstw, dsth)
    functions.createVideoFromFrame(data, genyuv, numframes, width, height, offset = 8)
    compareData = data.reshape(3, w, h)
    frametype = ['I', 'b', 'b', 'P', 'b', 'b', 'P']
    
    for quant in quants:
        #(app, yuvfilename, w, h, qp, outcomp, outdecomp, verbose = False)
        #print "Quant {}".format(quant)
        isize, psize, bsize = functions.compressFile(x264, genyuv, dstw, dsth, quant, gen264, decompyuv)
        frames = functions.cropImagesFromYUVfile(decompyuv, dstw, dsth, saveFrames)
        for idx, frame in enumerate(frames):
            mylen = len(frame)
            datayuv420 = np.asarray(frame)
            #print "The shape of the frame {}".format(frame.shape)
            datayuv = functions.YUV420_2_YUV444(datayuv420, dstw, dstw)
            datayuv = functions.cropROIfromYUV444(datayuv, 3, dstw, dstw, w, h, offset, offset)
            frameName = "q{}_f{}".format(quant, saveFrames[idx])
            #print frameName
            frame_dict[frameName] = datayuv
            m,s = functions.comparisonMetrics_yuvBuffers(compareData, datayuv, width, height)
            #print("Quant: {} PSNR: {} and SSIM: {}".format(quant, m, s))
            log.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(srcFile, label, quant, saveFrames[idx], frametype[saveFrames[idx]], m, s, isize, psize, bsize))

    #testrgb = planarYUV_2_planarRGB(datayuv, width=32, height=32)
    #display_image_rgb(testrgb, 32, 32)


datadir = '/Users/pam/Documents/data/CIFAR-10/cifar-10-batches-bin/'
srcdatasetdir = '/Users/pam/Documents/data/CIFAR-10/cifar-10-batches-bin'
dstdatasetdir = '/Users/pam/Documents/data/CIFAR-10/dataset'
x264 = "../x264/x264"


def generateDatasets(machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles, saveFrames, quants, dataSet="CIFAR10"):

    datadir = srcdatasetdir + "/"

    #if machine == '':
        #print "No entry for this machine"
        #quit()

    if not os.path.exists(dstdatasetdir):
        os.makedirs(dstdatasetdir)

    #label = 'A'
    #if dataSet=="CIFAR10":
    dw = 48
    dh = 48
    numframes = 7
    width = 32
    height = 32
    channels = 3
    pixel_depth = 8

    if dataSet == "MNIST":
        dw = 48
        dh = 48
        width = 28
        height = 28
        channels = 1

    if dataSet == "STL-10":
        dw = 112
        dh = 112
        width = 96
        height = 96
        channels = 3

    readApickle = False

    logfile = "{}/log.txt".format(dstdatasetdir)
    log = open(logfile, 'w')
    log.write("machine: {}".format(machine))
    log.write("src: {}".format(srcdatasetdir))
    log.write("dst: {}".format(dstdatasetdir))
    log.write("x264: {}".format(x264))
    log.write("batchfiles: {} \n".format(batchfiles))
    log.write("filename, quant, framenumber, frametype, PSNR, SSIM, isize, psize, bsize \n")

    #data_folders = [datadir+'data_batch_1']
    #data_folders = [datadir+'test_batch']
    if readApickle:
        data_folders = [datadir+'data_batch_1', datadir+'data_batch_2', datadir+'data_batch_3', datadir+'data_batch_4', datadir+'data_batch_5']
        label_folder = datadir+'batches.meta'

        data_dicts = []

        labels_dict = functions.unpickle(label_folder)
        #print("labels: ", labels_dict)

        num_cases_per_batch = labels_dict['num_cases_per_batch']
        label_names = labels_dict['label_names']

        print ("Number of cases in each batch: %d" %num_cases_per_batch)
        print ("Label Names: " + str(label_names))
    else:
        #data_folders = os.listdir(datadir)
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        data_folders = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and (".bin" in f)]

        if dataSet == "MNIST":
            label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if dataSet == "STL-10":
            label_names = ['zero', 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

        for idx, data_folder in enumerate(data_folders):
            data_folders[idx] = datadir + data_folder

    for data_folder in data_folders:
        if readApickle:
            data_dict = functions.unpickle(data_folder)
            data_dicts.append(data_dict)
            
            data_array = data_dict['data']
            data_labels = data_dict['labels']
            data_batch_label = data_dict['batch_label']
            data_filenames = data_dict['filenames']
        else:
            print data_folder
            f = open(data_folder, "rb")
            allTheData = np.fromfile(f, 'u1')
            recordSize = (width * height * channels) + 1
            if dataSet == "MNIST":
                # the MNIST file has a bunch of stuff in the first 4 32-bit ints (16 bytes)
                allTheData = allTheData[16:]
                recordSize = (width * height * channels)
            if dataSet == "STL-10":
                # STL-10 has a the labels in a different file.
                recordSize = (width * height * channels)
            num_cases_per_batch = allTheData.shape[0] / recordSize
            print("Numcases: {} recordSize: {} length of data {}".format(num_cases_per_batch, recordSize, len(allTheData)))
            
            allTheData = allTheData.reshape(num_cases_per_batch, recordSize)
            if dataSet == "CIFAR10":
                # from URL, 1 byte label, 3072 bytes rgb data in planar order
                data_labels = allTheData[:, 0].copy()
                data_array = allTheData[:, 1:].copy()
            if dataSet == "MNIST":
                data_label_filename = data_folder.replace(".bin", '')
                f = open(data_label_filename, "rb")
                data_labels = np.fromfile(f, 'u1')
                data_labels = data_labels[8:]
                # Add the UV channels
                data_array = allTheData.copy()
                uv_128s = np.full((num_cases_per_batch, recordSize), 128)
                data_array = np.concatenate((data_array, uv_128s), axis=1)
                data_array = np.concatenate((data_array, uv_128s), axis=1)
                data_array = data_array.flatten()
                recordSize = recordSize*3
                data_array = data_array.reshape(num_cases_per_batch, recordSize)
                # And because we've upped the number of channels...
                channels = 3
            if dataSet == "STL-10":
                data_label_filename = data_folder.replace(".bin", '')
                f = open(data_label_filename, "rb")
                data_labels = np.fromfile(f, 'u1')
                print data_labels
                data_array = allTheData.copy()
                data_array = data_array.reshape(num_cases_per_batch, channels, height, width)
                data_array = np.swapaxes(data_array, 2, 3)
                data_array = data_array.reshape(num_cases_per_batch, recordSize)

            data_batch_label = os.path.basename(data_folder)

            # from URL, 1 byte label, 3072 bytes rgb data in planar order

        
            print("The shape of allTheData {}".format(allTheData.shape))
            print("The shape of data_labels {}".format(data_labels.shape))
            print("The shape of data_array {}".format(data_array.shape))
            
        datasetNames = ["yuv", "y_quv", "y_squv", "interlaced"]
        for quant in quants:
            for idx, frame in enumerate(saveFrames):
                name = "q{}_f{}".format(quant, saveFrames[idx])
                datasetNames.append(name)
        #datasetNames = ["yuv"]
        
        

        #print datasetNames
        #print "The length of datasetNames {}".format(datasetNames)


        dataset = np.ndarray(shape=(len(datasetNames), num_cases_per_batch, (channels*width*height)), dtype=np.float32)
        # if doubling image
        #dataset = np.ndarray(shape=(len(datasetNames), num_cases_per_batch, (channels*width*2*height*2)), dtype=np.float32)

        isRGB = True
        if dataSet == "MNIST":
            isRGB = False

        #The image loop
        for idx, data in enumerate(data_array):
            #if idx > 2:
            #    break

            #print data_labels[idx]
            label = label_names[data_labels[idx]]
            #print "image {} label is {}".format(idx, label)
            if (idx % 1000 == 0):
                print "image {} label is {}".format(idx, label)
                localtime = time.localtime(time.time())
                print "Local current time :", localtime

            w = width
            h = height


            # Here, scale the image up...
            #data = functions.doubleImage(data, w, h)
            #w = w*2
            #h = h*2




            frame_dict = processSingleImage(data, w, h, x264, saveFrames, quants, log, data_batch_label, label, isRGB=isRGB)
            #print "Here are the keys: "
            #print frame_dict.keys()
            # sort the frames into their datasets
            for nameIdx, name in enumerate(datasetNames):
                frame = np.asarray(frame_dict[name])
                frame = frame.reshape((channels*w*h), )
                #normalise
                frame = (frame.astype(float) - pixel_depth / 2) / pixel_depth
                #print "name {} the frame shape: {}".format(name, frame.shape)
                dataset[nameIdx, idx, :]= frame

        """#pickling the new batch files
        for nameIdx, name in enumerate(datasetNames):
            pickleFileName = data_folder + "_" + name + "_gen"
            print "The pickleFileName {}".format(pickleFileName)
            myArray = dataset[nameIdx, :, :].copy()
            myDict = {'labels': data_labels, 'data': myArray, 'batch_label':data_batch_label, 'filenames': data_filenames}
            #print "The shape of the data {}".format(myArray.shape)
            if os.path.exists(pickleFileName) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            try:
                with open(pickleFileName, 'wb') as f:
                    pickle.dump(myDict, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickleFileName, ':', e)"""

        for nameIdx, name in enumerate(datasetNames):
            head, tail = os.path.split(data_folder)
            head, tail2 = os.path.split(head)
            #binFileName = data_folder + "_" + name + "_gen.bin"
            binPathName = dstdatasetdir + tail2 + "_" + name + "/"
            if not os.path.exists(binPathName):
                os.makedirs(binPathName)
            binFileName = binPathName + tail
            #print "The binFileName {}".format(binFileName)
            myArray = dataset[nameIdx, :, :].copy()
            labels = np.asarray(data_labels)
            labels = labels.reshape(labels.shape[0], 1)
            #print "The shape of the data {}".format(myArray.shape)
            #print "The shape of the labels {}".format(labels.shape)
            #myDict = {'labels': data_labels, 'data': myArray, 'batch_label':data_batch_label, 'filenames': data_filenames}
            myArray = np.concatenate((labels, myArray), axis=1)
            myArray = functions.convertToBytes(myArray)
            myByteArray = bytearray(myArray)
            mylen = len(myByteArray)
            #print "The shape of the data now {}".format(myArray.shape)
            #print "The length of the data now {}".format(mylen)
            force = False
            if os.path.exists(binFileName) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping bin saving.' % binFileName)
            try:
                with open(binFileName, 'wb') as f:
                    f.write(myByteArray)
            except Exception as e:
                print('Unable to save data to', binFileName, ':', e)
    return datasetNames

def main_0(argv=None):
    print "Start"
    machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles = readConfig.readConfigFile("config.txt")
    saveFrames = (0, 2, 3, 6)
    quants = (10, 25, 37, 41, 46, 50)
    bitrates = (200000, 100000, 50000, 35000, 20000, 10000)
    
    generateDatasets(machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles, saveFrames, bitrates)
    #quit()


def main(argv=None):
    print "Start"
    machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles = readConfig.readConfigFile("config.txt")
    saveFrames = (0, 2, 3, 6)
    #saveFrames = (0,)
    quants = (10, 25, 37, 41, 46, 50)
    #quants = (10, )
    bitrates = (200000, 100000, 50000, 35000, 20000, 10000)

    generateDatasets(machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles, saveFrames, quants, dataSet="STL-10")
    # quit()

if __name__ == "__main__":
    main()
