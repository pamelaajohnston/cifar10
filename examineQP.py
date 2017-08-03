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

datadir = '/Volumes/LaCie/ILSVRC/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2015_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'

x264 = "x264"
ldecod = "/Users/pam/Documents/dev/JM/bin/ldecod.exe"
ffmpeg = "ffmpeg"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

def main(argv=None):
    baseFileName = 'ILSVRC2017_train_00000000'
    fileName = datadir
    print(fileName)
    fileName = os.path.join(datadir, videoFilesBase, baseFileName + ".mp4")
    videoBaseDir = os.path.join(datadir, videoFilesBase)
    print("The filename is {}".format(fileName))
    print("The videoBaseDir is {}".format(videoBaseDir))
    qps = []

    index = 0
    for (dirpath0, dirnames0, filenames0) in os.walk(videoBaseDir):
        for filename0 in filenames0:
            if filename0.endswith('.mp4'):
                #index = index + 1
                #if index > 3:
                #    break

                fileName = os.path.join(datadir, videoFilesBase, filename0)
                baseFileName, ext = os.path.splitext(filename0)
                print("The filename is {} baseFileName {}".format(fileName, baseFileName))

                # First demux mp4 to AnnexB (.264)
                app = ffmpeg
                appargs = "-i {} -codec copy -bsf:v h264_mp4toannexb out.264".format(fileName)
                exe = app + " " + appargs
                args = shlex.split(exe)
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                #out, err = proc.communicate()

                # Next decode and get the qp stuff
                app = ldecod
                appargs = "-p InputFile=out.264"
                exe = app + " " + appargs
                args = shlex.split(exe)
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out, err = proc.communicate()
                #print("output is:")
                out = out.split('\n')
                #print(out)

                #qpVals = [line for line in out if 'MB no' in line]

                #print(qpVals)

                mbNos = []
                qps = []
                for line in out:
                    if 'MB no' in line:
                        words = line.split(' ')
                        mbNos.append(int(words[2]))
                        qps.append(int(words[4]))

                #print(qps)

                qps = np.asarray(qps)

                #Now read the annotation
                annotFileName = os.path.join(datadir, annotFilesBase, baseFileName + "/000000.xml")
                #Walking through the frames - annotation xml frames
                annotDir = os.path.join(datadir, annotFilesBase, baseFileName)
                for (dirpath, dirnames, filenames) in os.walk(annotDir):
                    for filename in filenames:
                        if filename.endswith('.xml'):
                            frameNo = int(filename.replace(".xml", ''))
                            #print(frameNo)
                            annotFileName = os.path.join(datadir, annotFilesBase, baseFileName, filename)

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

                            frameMBs = mb_width * mb_height

                            objectqp = []
                            frameNo = 0
                            for y in range(yminMB, ymaxMB):
                                for x in range(xminMB, xmaxMB):
                                    mb = (y * mb_width) + x
                                    idx = (frameNo * frameMBs) + mb
                                    objectqp.append(qps[idx])

                            objectqp = np.asarray(objectqp)



    average = np.mean(qps)
    variance = np.var(qps)
    print("The average qp: {}, with variance: {}".format(average, variance))
    average = np.mean(objectqp)
    variance = np.var(objectqp)
    print("The average object qp: {}, with variance: {}".format(average, variance))

    plt.figure()
    (n, bins, patches) = plt.hist(qps, normed=True, label="all qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    (n_obj, bins_obj, patches_obj) = plt.hist(objectqp, normed=True, label="object qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of QP in ILSVRC2017 values")
    plt.legend(fontsize = 'small', framealpha = 0.5)
    #plt.show()
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    plt.savefig('QP_of_ILSVRC2017_train0000.png', bbox_inches='tight')

    print(bins)
    print(n)
    print(n_obj)

if __name__ == "__main__":
    main()