import random
import os
import shlex, subprocess
import numpy as np
import time

# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

datadir = '/Users/pam/Documents/data/ILSVRC/ILSVRC_new/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'
baseFileName = 'ILSVRC2017_train_00012000'
x264 = "x264"
ldecod = "/Users/pam/Documents/dev/JM/bin/ldecod.exe"
ldecod = "ldecod"
ffmpeg = "ffmpeg"

displayFrameNo = 0

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

def main(argv=None):
    fileName = datadir
    print(fileName)
    fileName = os.path.join(datadir, videoFilesBase, baseFileName + ".mp4")
    print("The filename is {}".format(fileName))


    # First demux mp4 to AnnexB (.264)
    app = ffmpeg
    appargs = "-i {} -codec copy -bsf:v h264_mp4toannexb out.264".format(fileName)
    exe = app + " " + appargs
    args = shlex.split(exe)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #out, err = proc.communicate()

    #print("output is:")
    #print(out)
    #print("error is:")
    #print(err)

    # Next decode and get the qp stuff
    app = ldecod
    appargs = "-p InputFile=out.264"
    exe = app + " " + appargs
    print("cmd>> {}".format(exe))
    args = shlex.split(exe)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()
    #print("output is: {}".format(out))
    out = out.split('\n')
    #print(out)

    #qpVals = [line for line in out if 'MB no' in line]

    #print(qpVals)

    mbNos = []
    allqps = []
    qps = []
    frameNos = []
    #frameMbNos = []
    frameQps = []
    for line in out:
        if 'MB no' in line:
            words = line.split(' ')
            mbNos.append(int(words[2]))
            qps.append(int(words[4]))
            allqps.append(int(words[4]))
        else:
            words = line.split(' ')
            #print("{}".format(words[0]))
            if "FrameData" in words[0]:
                picnum = int(words[1])
                numMbs = len(qps)
                frameNos.append(picnum)
                frameNos.append(qps)
                frameQps.append(qps)
                mbNos = []
                qps = []

    # re-order them for frames sake (stoopid b-frames!)
    entryLength = numMbs + 1
    print("Each line is {} long".format(entryLength))
    data = []
    for sublist in frameNos:
        #print("The sublist is: {}".format(sublist))
        try:
            for item in sublist:
                data.append(item)
                #print("The item is {}".format(item))
        except:
            #print("The sublist is {}".format(sublist))
            data.append(sublist)


    #print(data)
    data = np.array(data)
    data = data.reshape((-1, entryLength))
    frameNos = data[: , 0]
    qps = data[:, 1:]
    print("The unordered framenos: {}".format(frameNos))

    print("Shape of the qps {}".format(qps.shape))

    print("Data before {}".format(data[1]))
    i = np.argsort(frameNos)
    data = data[i, :]
    print("Data after  {}".format(data[1]))

    qps = data[:, 1:]
    #print("The frame qps  {}".format(qps[displayFrameNo]))
    #print("The qps: {}".format(qps[0]))

    #Now read the annotation
    fileName = '/{num:06d}.xml'.format(num=displayFrameNo)
    #fileName = "/000001.xml"
    annotFileName = os.path.join(datadir, annotFilesBase, baseFileName + fileName)

    etree = ET.parse(annotFileName)
    myval = etree.find('size')
    size = ([int(it.text) for it in myval])
    width, height = size
    #print("The dimensions: {} by {}".format(width, height))

    objects = etree.findall("object")
    for object_iter in objects:
        bndbox = object_iter.find("bndbox")
        dims = ([int(it.text) for it in bndbox])
        xmax, xmin, ymax, ymin = dims
    #print("The bounding box: {}, {}, {}, {}".format(xmax, xmin, ymax, ymin))

    #Bounding box in macroblocks
    xminMB = int(xmin/16)
    xmaxMB = int((xmax+15)/16)
    yminMB = int(ymin/16)
    ymaxMB = int((ymax+15)/16)

    mb_width = int((width+15)/16)
    mb_height = int((height+15)/16)

    qps = qps.reshape((qps.shape[0], mb_height, mb_width))
    #print("The shape of the data: {}".format(qps.shape))
    #print("The frame qps  {}".format(qps[displayFrameNo]))
    #print("Bounding box: ({},{}) to ({},{})".format(xminMB, yminMB, xmaxMB, ymaxMB))

    frameMBs = mb_width * mb_height

    objectqp = []
    frameNo = displayFrameNo
    firstMB = (yminMB * mb_width) +xminMB
    lastMB = (ymaxMB * mb_width) + xmaxMB
    for y in range(yminMB, ymaxMB):
        for x in range(xminMB, xmaxMB):
            objectqp.append(qps[frameNo, y, x])
            #print("the qp of ({},{}): {}".format(x, y, qps[frameNo, y, x]))

    objectqp = np.asarray(objectqp)

    #Walking through the frames - annotation xml frames
    list_of_files = {}
    if False:
        annotDir = os.path.join(datadir, annotFilesBase, baseFileName)
        for (dirpath, dirnames, filenames) in os.walk(annotDir):
            for filename in filenames:
                if filename.endswith('.xml'):
                    frameNo = int(filename.replace(".xml", ''))
                    print(frameNo)


    qps = qps.flatten()
    qps = np.ndarray.tolist(qps)
    average = np.mean(qps)
    variance = np.var(qps)
    print("The average qp: {}, with variance: {}".format(average, variance))
    average = np.mean(objectqp)
    variance = np.var(objectqp)
    print("The average object qp: {}, with variance: {}".format(average, variance))

    plt.hist(qps, normed=True, label="all qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    plt.hist(objectqp, normed=True, label="object qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of qp values")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()