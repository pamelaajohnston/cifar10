import random
import os
import shlex, subprocess
import numpy as np
import time
from PIL import Image
import yuvview

# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

saveFolder = "qpFiles/"
decodedBmps = "dump"

datadir = '/Users/pam/Documents/data/ILSVRC/ILSVRC_new/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'

datadir = '/Volumes/LaCie/data/ILSVRC/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2015_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'

x264 = "x264"
#ldecod = "/Users/pam/Documents/dev/JM/bin/ldecod.exe"
#ldecod = "ldecod"
ffmpeg = "ffmpeg"

logfilename = "ILSVRC_PSNR_log.txt"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

# returns all the files
ilsvrcBase = '/Volumes/LaCie/data/ILSVRC/'
def allTheFiles():
    allFiles = [] # eventually we want this to contain: [video file, jpeg folder, annotation file]
    videoFilesBranch = 'Data/VID/snippets/'
    jpegFilesBranch = 'Data/VID/'
    annotFilesBranch = 'Annotations/VID'

    dataPartitions = ["train", "test", "val" ]
    dataPartitions = ["train",]

    for dataPartition in dataPartitions:
        videoFolder = os.path.join(ilsvrcBase, videoFilesBranch, dataPartition)
        for dirName, subdirList, fileList in os.walk(videoFolder):
            #print('Found directory: %s' % dirName)
            for fname in fileList:
                #print('\t%s' % fname)
                if fname.endswith('.mp4'):
                    vfileName = os.path.join(ilsvrcBase, videoFilesBranch, dataPartition, dirName, fname)
                    baseFileName, ext = os.path.splitext(fname)
                    subdirName = os.path.basename(dirName)
                    jfileName = os.path.join(ilsvrcBase, jpegFilesBranch, dataPartition, subdirName, fname)
                    jfileName = jfileName.replace(".mp4", "/")
                    afileName = os.path.join(ilsvrcBase, annotFilesBranch, dataPartition, subdirName, fname)
                    afileName = afileName.replace(".mp4", "/")

                    triple = [vfileName, jfileName, afileName]
                    allFiles.append(triple)
    return allFiles

def compareImages_PSNR(fname1, fname2):
    verbose = False
    psnr = 0
    img1 = Image.open(fname1)
    arr1 = np.array(img1)  # 640x480x4 array
    if verbose: print("The shape of image 1: {}".format(arr1.shape))
    img2 = Image.open(fname2)
    arr2 = np.array(img2)  # 640x480x4 array
    if verbose: print("The shape of image 2: {}".format(arr2.shape))
    psnr = yuvview.psnr(arr1, arr2)
    return psnr

def getImageFileNames(frameNo, folder1, folder2):
    verbose = False
    fname1 = "%06d.bmp" % (frameNo + 1)
    fname1 = "%06d.jpg" % (frameNo + 1)
    fname1 = os.path.join(folder1, fname1)
    if not os.path.isfile(fname1):
        fname1 = "null"
    if verbose: print("The first filename {}".format(fname1))
    fname2 = "%06d.jpeg" % frameNo
    fname2 = os.path.join(folder2, fname2)
    if not os.path.isfile(fname2):
        fname2 = "null"
    if verbose: print("The second filename {}".format(fname2))
    return (fname1, fname2)


def main(argv=None):
    logFile = open("ILSVRC_PSNR_log.txt", "w")
    counter = 0
    fileList = [["/Volumes/LaCie/data/ILSVRC/Data/VID/snippets/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000.mp4",
                 "/Volumes/LaCie/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/",
                 "/Volumes/LaCie/data/ILSVRC/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/"
                ]]
    #fileList = allTheFiles()
    for entry in fileList:
        counter = counter +1
        #if counter > 1:
        #    break
        (vfileName, jfileName, afileName) = entry
        print(vfileName)
        print(jfileName)
        print(afileName)

        exe = "{} -i {} -f image2 -pix_fmt bgr24 {}/%6d.bmp".format(ffmpeg, vfileName, decodedBmps)
        exe = "{} -i {} -f image2 -qscale:v 5 {}/%6d.jpg".format(ffmpeg, vfileName, decodedBmps)
        args = shlex.split(exe)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print("Started subproc for ldecod")
        #proc.wait()
        #print("Finished subproc for ldecod")
        out, err = proc.communicate()
        print(out)

        firstAnnotation = os.path.join(afileName, "000000.xml")
        etree = ET.parse(firstAnnotation)
        myval = etree.find('size')
        size = ([int(it.text) for it in myval])
        width, height = size

        # Getting the psnr between the video file and the jpeg.
        frameNumber = 0
        (f1, f2) = getImageFileNames(frameNumber, decodedBmps, jfileName)
        while f1 != "null" and f2 != "null":
            psnr = compareImages_PSNR(f1, f2)

            b = os.path.getsize(f2) * 8
            pels = width * height
            bpp = float(b)/float(pels)

            print("Seq: {} frameNumber: {} PSNR: {} jpeg_bpp: {}".format(vfileName, frameNumber, psnr, bpp))
            logFile.write("Seq {} frameNumber {} PSNR {} jpeg_bpp {} \n".format(vfileName, frameNumber, psnr, bpp))

            frameNumber = frameNumber + 1
            (f1, f2) = getImageFileNames(frameNumber, decodedBmps, jfileName)

        b = os.path.getsize(vfileName) * 8
        totalPixels = frameNumber*width*height
        bitsPerPixel = float(b)/float(totalPixels)
        logFile.write("Seq {} width {} height {} frames {} bits_per_pixel {} bytes_in_file {}".format(vfileName, width, height, frameNumber, bitsPerPixel, b))
        print("Seq: {} width: {} height: {} frames: {} bits_per_pixel: {} bytes_in_file: {} \n".format(vfileName, width, height, frameNumber, bitsPerPixel, b))

    logFile.close()



def main_2(argv=None):
    baseFileName = 'ILSVRC2017_train_00000000'
    fileName = datadir
    print(fileName)
    fileName = os.path.join(datadir, videoFilesBase, baseFileName + ".mp4")
    videoBaseDir = os.path.join(datadir, videoFilesBase)
    print("The filename is {}".format(fileName))
    print("The videoBaseDir is {}".format(videoBaseDir))
    totalqps = []
    totalobjqps = []
    intraqps = []
    interqps = []

    index = 0
    fileList = allTheFiles()
    counter = 0
    for entry in fileList:

        (fileName, foob, foob2) = entry
        # First demux mp4 to AnnexB (.264)
        os.remove("out.264")
        app = ffmpeg
        appargs = "-i {} -codec copy -bsf:v h264_mp4toannexb out.264".format(fileName)
        exe = app + " " + appargs
        args = shlex.split(exe)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print("Started subproc for ffmpeg")
        proc.wait()
        #print("Finished subproc for ffmpeg")

        #out, err = proc.communicate()

        # Next decode and get the qp stuff
        app = ldecod
        appargs = "-p InputFile=out.264"
        exe = app + " " + appargs
        args = shlex.split(exe)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print("Started subproc for ldecod")
        #proc.wait()
        #print("Finished subproc for ldecod")
        out, err = proc.communicate()
        #print("output is:")

        #saving the file:
        saveFileName = "{}_decoderOP.txt".format(baseFileName)
        saveFileName = os.path.join(saveFolder, saveFileName);
        with open(saveFileName, "w") as saveF:
            saveF.write(out)

        out = out.split('\n')
        #print(out)

        #qpVals = [line for line in out if 'MB no' in line]

        #print(qpVals)

        mbNos = []
        qps = []
        frameNos = []
        for line in out:
            if 'MB no' in line:
                words = line.split(' ')
                mbNos.append(int(words[2]))
                qps.append(int(words[4]))
                totalqps.append(int(words[4]))
            else:
                words = line.split(' ')
                # print("{}".format(words[0]))
                if "I" in line:
                    intraqps.extend(qps)
                else:
                    interqps.extend(qps)
                if "FrameData" in words[0]:
                    picnum = int(words[1])
                    numMbs = len(qps)
                    frameNos.append(picnum)
                    frameNos.append(qps)
                    mbNos = []
                    qps = []

        # re-order them for frames sake (stoopid b-frames!)
        entryLength = numMbs + 1
        print("Each line is {} long".format(entryLength))
        data = []
        for sublist in frameNos:
            # print("The sublist is: {}".format(sublist))
            try:
                for item in sublist:
                    data.append(item)
                    # print("The item is {}".format(item))
            except:
                # print("The sublist is {}".format(sublist))
                data.append(sublist)

        # print(data)
        data = np.array(data)
        data = data.reshape((-1, entryLength))
        frameNos = data[:, 0]
        # qps = data[:, 1:]
        # print("The unordered framenos: {}".format(frameNos))


        # print("Data before {}".format(data[1]))
        i = np.argsort(frameNos)
        data = data[i, :]
        # print("Data after  {}".format(data[1]))

        qps = data[:, 1:]
        #print("Shape of the qps {}".format(qps.shape))

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

                    xmin = 0
                    xmax = width - 1
                    ymin = 0
                    ymax = height - 1

                    objectInFrame = False
                    objects = etree.findall("object")
                    for object_iter in objects:
                        bndbox = object_iter.find("bndbox")
                        dims = ([int(it.text) for it in bndbox])
                        xmax, xmin, ymax, ymin = dims
                        objectInFrame = True
                    #print("The bounding box: {}, {}, {}, {}".format(xmax, xmin, ymax, ymin))

                    #Bounding box in macroblocks
                    xminMB = int(xmin/16)
                    xmaxMB = int((xmax+15)/16)
                    yminMB = int(ymin/16)
                    ymaxMB = int((ymax+15)/16)

                    mb_width = int((width+15)/16)
                    mb_height = int((height+15)/16)
                    #print("mbw: {}, mbh: {}".format(mb_width, mb_height))

                    #frameMBs = mb_width * mb_height
                    qps = qps.reshape((qps.shape[0], mb_height, mb_width))

                    if objectInFrame:
                        objectqp = []
                        #frameNo = 0
                        for y in range(yminMB, ymaxMB):
                            for x in range(xminMB, xmaxMB):
                                objectqp.append(qps[frameNo, y, x])
                        totalobjqps.extend(objectqp)
                        objectqp = np.asarray(objectqp)



    #qps = qps.flatten()
    #qps = np.ndarray.tolist(qps)
    average = np.mean(totalqps)
    variance = np.var(totalqps)
    max = np.max(totalqps)
    min = np.min(totalqps)
    print("The average qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))
    average = np.mean(totalobjqps)
    variance = np.var(totalobjqps)
    max = np.max(totalobjqps)
    min = np.min(totalobjqps)
    print("The average object qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))
    average = np.mean(intraqps)
    variance = np.var(intraqps)
    max = np.max(intraqps)
    min = np.min(intraqps)
    print("The average intra qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))
    average = np.mean(interqps)
    variance = np.var(interqps)
    max = np.max(interqps)
    min = np.min(interqps)
    print("The average inter qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))

    plt.figure()
    (n, bins, patches) = plt.hist(totalqps, normed=True, label="all qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    (n_obj, bins_obj, patches_obj) = plt.hist(totalobjqps, normed=True, label="object qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
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