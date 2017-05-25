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


def processSingleImage(data, w, h, x264, saveFrames, quants):
    datayuv = functions.planarRGB_2_planarYUV(data, w, h)
    dataQyuv = datayuv.copy()
    dataQyuv = functions.quantiseUV(dataQyuv, w, h)
    
    frame_dict = {
                    'yuv'   : datayuv,
                    'y_quv' : dataQyuv,
    }
    getVideoCompressedImages(datayuv, w, h, frame_dict, x264, saveFrames, quants)
    return frame_dict


def getVideoCompressedImages(data, w, h, frame_dict, x264, saveFrames, quants):
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

    for quant in quants:
        #(app, yuvfilename, w, h, qp, outcomp, outdecomp, verbose = False)
        #print "Quant {}".format(quant)
        functions.compressFile(x264, genyuv, dstw, dsth, quant, gen264, decompyuv)
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
            #testrgb = planarYUV_2_planarRGB(datayuv, width=32, height=32)
            #display_image_rgb(testrgb, 32, 32)




datadir = '/Users/pam/Documents/data/CIFAR-10/cifar-10-batches-py/'
srcdatasetdir = '/Users/pam/Documents/data/CIFAR-10/cifar-10-batches-py/'
dstdatasetdir = '/Users/pam/Documents/data/CIFAR-10/'
x264 = "../x264/x264"


print "Start"

machine, srcdatasetdir, dstdatasetdir, copytodir, x264, batchfiles = readConfig.readConfigFile("config.txt")

if machine == '':
    print "No entry for this machine"
    quit()

if not os.path.exists(dstdatasetdir):
    os.makedirs(dstdatasetdir)

saveFrames = (0,2,3,6)
quants = (10, 25, 50)

#label = 'A'
dw = 48
dh = 48
numframes = 7
width = 32
height = 32
channels = 3
pixel_depth = 8

logfile = "{}/log.txt".format(dstdatasetdir)
log = open(logfile, 'w')
log.write("machine: {}".format(machine))
log.write("src: {}".format(srcdatasetdir))
log.write("dst: {}".format(dstdatasetdir))
log.write("x264: {}".format(x264))
log.write("batchfiles: {} \n".format(batchfiles))
log.write("filename, quant, frametype, label, PSNR, SSIM, filesize, numframes \n")

#data_folders = [datadir+'data_batch_1']
#data_folders = [datadir+'test_batch']
data_folders = [datadir+'data_batch_1', datadir+'data_batch_2', datadir+'data_batch_3', datadir+'data_batch_4', datadir+'data_batch_5']
label_folder = datadir+'batches.meta'

data_dicts = []

labels_dict = functions.unpickle(label_folder)
#print("labels: ", labels_dict)

num_cases_per_batch = labels_dict['num_cases_per_batch']
label_names = labels_dict['label_names']

print ("Number of cases in each batch: %d" %num_cases_per_batch)
print ("Label Names: " + str(label_names))


for data_folder in data_folders:
    data_dict = functions.unpickle(data_folder)
    data_dicts.append(data_dict)

    data_array = data_dict['data']
    data_labels = data_dict['labels']
    data_batch_label = data_dict['batch_label']
    data_filenames = data_dict['filenames']




    datasetNames = ["yuv", "y_quv"]
    for quant in quants:
        for idx, frame in enumerate(saveFrames):
            name = "q{}_f{}".format(quant, saveFrames[idx])
            datasetNames.append(name)

    print datasetNames
    print "The length of datasetNames {}".format(datasetNames)

    dataset = np.ndarray(shape=(len(datasetNames), num_cases_per_batch, (channels*width*height)), dtype=np.float32)

    #The image loop
    for idx, data in enumerate(data_array):
        #if idx > 2:
        #    break

        label = label_names[data_labels[idx]]
        #print "image {} label is {}".format(idx, label)
        if (idx % 1000 == 0):
            print "image {} label is {}".format(idx, label)
            localtime = time.localtime(time.time())
            print "Local current time :", localtime
        frame_dict = processSingleImage(data, width, height, x264, saveFrames, quants)
        #print "Here are the keys: "
        #print frame_dict.keys()
        # sort the frames into their datasets
        for nameIdx, name in enumerate(datasetNames):
            frame = np.asarray(frame_dict[name])
            frame = frame.reshape((channels*width*height), )
            #normalise
            frame = (frame.astype(float) - pixel_depth / 2) / pixel_depth
            #print "name {} the frame shape: {}".format(name, frame.shape)
            dataset[nameIdx, idx, :]= frame

    #pickling the new batch files
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
            print('Unable to save data to', pickleFileName, ':', e)



quit()