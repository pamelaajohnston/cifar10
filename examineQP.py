import random
import os
import shlex, subprocess
import numpy as np
import time

# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

saveFolder = "qpFiles/"

datadir = '/Users/pam/Documents/data/ILSVRC/ILSVRC_new/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'

datadir = '/Volumes/LaCie/data/ILSVRC/'
videoFilesBase = 'Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'
annotFilesBase = 'Annotations/VID/train/ILSVRC2015_VID_train_0000/'
baseFileName = 'ILSVRC2017_train_00000000'
fileType = '.mp4'

datadir = '/Volumes/LaCie/data/yuv_quant_noDeblock_train/'
videoFilesBase = ''
annotFilesBase = ''
baseFileName = ''
fileType = '.h264'


x264 = "x264"
#ldecod = "/Users/pam/Documents/dev/JM/bin/ldecod.exe"
ldecod = "ldecod"
ffmpeg = "ffmpeg"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

fileSizes = [
    ['qcif', 176, 144],
    ['512x384', 512, 384],
    ['384x512', 384, 512],
    ['cif', 352, 288],
    ['sif', 352, 240],
    ['720p', 1280, 720],
    ['1080p', 1920, 1080]
]


def createFileList(myDir, takeAll = False, format='.yuv'):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(myDir):
        for filename in filenames:
            if filename.endswith(format):
                if takeAll or '_cif' in filename:
                    fileName = os.path.join(myDir, dirName, filename)
                    baseFileName, ext = os.path.splitext(filename)
                    print("The filename is {} baseFileName {}".format(fileName, baseFileName))

                    if format=='.yuv':
                        for fileSize in fileSizes:
                            if fileSize[0] in fileName:
                                tuple = [fileName, fileSize[1], fileSize[2]]
                                fileList.append(tuple)
                                break
                    elif format=='.tif':
                        tuple = [fileName, -1, -1]
                        fileList.append(tuple)
                    else:
                        tuple = [fileName, -1, -1]
                        fileList.append(tuple)


    #hacky hack
    #fileList = [['/Volumes/LaCie/data/yuv_quant_noDeblock/quant_0/mobile_cif_q0.yuv', 352, 288],]
    #random.shuffle(fileList)
    #print(fileList)
    return fileList

def demuxFromMPEG4toH264AnnexB(inFilename, outFilename):
    baseFileName, ext = os.path.splitext(inFilename)
    print("The filename is {} baseFileName {}".format(inFilename, baseFileName))

    # First demux mp4 to AnnexB (.264)
    try:
        os.remove(outFilename)
    except Exception as e:
        print("The file wasn't there?: {}".format(e))
    app = ffmpeg
    appargs = "-i {} -codec copy -bsf:v h264_mp4toannexb {}".format(inFilename, outFilename)
    exe = app + " " + appargs
    args = shlex.split(exe)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print("Started subproc for ffmpeg")
    proc.wait()
    # print("Finished subproc for ffmpeg")
    # out, err = proc.communicate()

def main(argv=None):
    baseFileName = 'ILSVRC2017_train_00000000'
    fileName = datadir
    print(fileName)
    #fileName = os.path.join(datadir, videoFilesBase, baseFileName + ".mp4")
    videoBaseDir = os.path.join(datadir, videoFilesBase)
    #print("The filename is {}".format(fileName))
    print("The videoBaseDir is {}".format(videoBaseDir))
    totalqps = []
    totalobjqps = []
    intraFrameQps = []
    interFrameQps = []
    interFrameIntraMBs = []

    index = 0
    #for (dirpath0, dirnames0, filenames0) in os.walk(videoBaseDir):
    filenames0 = createFileList(videoBaseDir, takeAll=True, format=fileType)
    # hacky hack, filenames is a list
    filenames0 = filenames0[:2]
    print(filenames0)

    for myfile in filenames0:
        filename0 = myfile[0]
        h264FileName = myfile[0]
        baseFileName, ext = os.path.splitext(myfile[0])
        if filename0.endswith('.mp4'):
            h264FileName = "out.264"
            demuxFromMPEG4toH264AnnexB(filename0, h264FileName)


        # Next decode and get the qp stuff
        app = ldecod
        appargs = "-p InputFile={}".format(h264FileName)
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
        saveFileName = os.path.join(saveFolder, saveFileName)
        with open(saveFileName, "w") as saveF:
            saveF.write(out)

        out = out.split('\n')
        #print(out)

        #qpVals = [line for line in out if 'MB no' in line]

        #print(qpVals)

        #example line:
        # MB no: 3325 qp: 32 intra: 0
        # 0  1   2    3   4  5      6

        mbNos = []
        qps = []
        intras = []
        frameNos = []
        for line in out:
            if 'MB no' in line:
                words = line.split(' ')
                mbNos.append(int(words[2]))
                qps.append(int(words[4]))
                totalqps.append(int(words[4]))
                intras.append(int(words[6]))
            else:
                words = line.split(' ')
                # print("{}".format(words[0]))
                if "I" in line:
                    intraFrameQps.extend(qps)
                else:
                    interFrameQps.extend(qps)
                    interFrameIntraMBs.extend(intras)

                # FrameData: 3  00003( P ) 6 1 4 4:2:0   48821
                # 0          1  2      3   4 5 6 7       8
                if "FrameData" in words[0]:
                    picnum = int(words[1])
                    numMbs = len(qps)
                    frameNos.append(picnum)
                    frameNos.append(qps)
                    frameNos.append(intras)
                    mbNos = []
                    qps = []
                    intras = []

        # re-order them for frames sake (stoopid b-frames!)
        entryLength = numMbs + numMbs + 1 # frameNumber + qps (for each mb) + intras (for each mb)
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

        qps = data[:, 1:(numMbs+1)]
        intras = data[:, (numMbs+1):]
        #print("Shape of the qps {}".format(qps.shape))

        doAnnotation = False
        if (doAnnotation):
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
    if doAnnotation:
        average = np.mean(totalobjqps)
        variance = np.var(totalobjqps)
        max = np.max(totalobjqps)
        min = np.min(totalobjqps)
    print("The average object qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))
    average = np.mean(intraFrameQps)
    variance = np.var(intraFrameQps)
    max = np.max(intraFrameQps)
    min = np.min(intraFrameQps)
    print("The average intra qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))
    average = np.mean(interFrameQps)
    variance = np.var(interFrameQps)
    max = np.max(interFrameQps)
    min = np.min(interFrameQps)
    print("The average inter qp: {}, with variance: {}, max: {}, min: {}".format(average, variance, max, min))

    average = np.mean(interFrameIntraMBs)*100
    print("The average percentage intra mbs: {} in an inter frame".format(average))

    #plt.figure()
    #(n, bins, patches) = plt.hist(totalqps, normed=True, label="all qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    #(n_obj, bins_obj, patches_obj) = plt.hist(totalobjqps, normed=True, label="object qps", alpha=0.7, bins=range(0, 52, 1))  # plt.hist passes it's arguments to np.histogram
    #plt.title("Histogram of QP in ILSVRC2017 values")
    #plt.legend(fontsize = 'small', framealpha = 0.5)
    #plt.show()
    #frame = plt.gca()
    #frame.axes.get_yaxis().set_ticks([])
    #plt.savefig('QP_of_ILSVRC2017_train0000.png', bbox_inches='tight')

    #print(bins)
    #print(n)
    #print(n_obj)

    #Now for the turning into patches....

if __name__ == "__main__":
    main()