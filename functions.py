from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import socket
import readConfig


def createVideoFromFrame(data, filename, numframes, width, height, offset = 8):
    #print("length of data is: " + str(len(data)))
    #shape = datayuv444.shape
    #print(str(shape))

    pic = data.reshape(3, width, height)
    shape = (3, (width+(offset*2)), (height+(offset*2)))
    #print(str(shape))

    bg_pic = np.zeros(shape)

    #add the stripes
    for channel in range(0, shape[0]):
        for row in range(0, shape[1]):
            bg_pic[channel, row,:] = random.randint(0,255)

    #datargb = planarYUV_2_planarRGB(bg_pic, width=40, height=40)
    #display_image_rgb(datargb, 40, 40)


    # remove any existing file
    if os.path.exists(filename):
        os.remove(filename)

    for x in range(0, numframes):
        #print(str(x))
        comb1 = np.array(bg_pic[:,(x*2):,:].copy())
        comb2 = np.array(bg_pic[:,0:(x*2),:].copy())
        
        #print("The shape of the background is: "+str(bg_pic.shape))
        #print("The shape of the comb1 is: "+str(comb1.shape))
        #print("The shape of the comb2 is: "+str(comb2.shape))
        comb = np.concatenate((comb1, comb2), axis=1)
        #print("The shape of the comb is: "+str(comb.shape))
        
        # Now slap the picture in the middle
        comb_pic = comb.copy()
        comb_pic[:, offset:(offset+width), offset:(offset+height)] = pic[:,:,:]
        
        #datargb = planarYUV_2_planarRGB(comb_pic, width=(width+(offset*2)), height=(height+(offset*2)))
        #display_image_rgb(datargb, (width+(offset*2)), (height+(offset*2)))
        
        datayuv = YUV444_2_YUV420(comb_pic, width=(width+(offset*2)), height=(height+(offset*2)))
        appendToFile(datayuv, filename)



def compressFile(app, yuvfilename, w, h, qp, outcomp, outdecomp, verbose = False):
    if verbose:
        print("************Compressing the yuv************")
    inputres = '{}x{}'.format(w,h)
    #app = "../x264/x264"
    #if sys.platform == 'win32':
        #app = "..\\x264\\x264.exe"
    appargs = '-o {} -q {} --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(outcomp, qp, inputres, outdecomp, yuvfilename)
    # IBBP: 2 b-frames
    appargs += ' -b 2 --b-adapt 0'
    
    #print appargs

    exe = app + " " + appargs
    #print exe

    if sys.platform == 'win32':
        args = exe
    else:
        args = shlex.split(exe)

    #subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()

    if verbose:
        print err

def comparisonMetrics(imageA, imageB, verbose = False):
    if verbose:
        print("************Comparison Metrics************")
    dataA = np.array(list(imageA.getdata()))
    dataB = np.array(list(imageB.getdata()))
    #m = mse(dataA, dataB)
    m = yuvview.psnr(dataA, dataB)
    s = yuvview.ssim(imageA, imageB)
    return (m, s)

def cropImageFromYuvFile(yuvFileName, yuvW, yuvH, bmpFileName, bmpW, bmpH, framenum, yuvformat='i420', bmpformat='L', verbose = False):
    if verbose:
        print("************Crop image from yuv************")
    offsetx = (yuvW - bmpW)/2
    offsety = (yuvH - bmpH)/2
    #decompbmp ="{}_{}x{}.bmp".format('decomp', yuvW, yuvH)
    
    decompImg = yuvview.yuvFileTobmpFile (yuvFileName=decompfilename, width=yuvW, height=yuvH, framenum=framenum, format=yuvformat, bmpFileName="")
    w, h = decompImg.size
    #print "decompImg dimensions {} by {}".format(w,h)
    #print "offsetx {} offsety {} sw {} sh {}".format(offsetx, offsety, sw, sh)
    image_out = decompImg.crop((offsetx, offsety, offsetx+bmpW, offsety+bmpH))
    image_out = image_out.convert(bmpformat)
    #print(image_in.format, image_in.size, image_in.mode)
    #print(image_out.format, image_out.size, image_out.mode)
    #img.rotate(45, expand=True)
    image_out.save(bmpFileName)
    return image_out

##########################################################################################
## These come from the CIFAR python notebook that I've been messing about with
##########################################################################################
import cPickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def thresholdAndRound(y):
    maxn = 255
    minn = 0
    y[y > maxn] = maxn
    y[y < minn] = minn
    y = np.around(y,0)
    return y

def convertToBytes(y):
    y = np.asarray(y, 'u1')
    return y


#Note that this function takes as input a planar RGB image
# It returns planar YUV4:4:4 (it's not common but it can be downsampled to 4:2:0)
def planarRGB_2_planarYUV(data, width, height):
    #print("in planarRGB_2_planarYUV")
    delta = 128.0
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    r = np.array(picture[0])
    g = np.array(picture[1])
    b = np.array(picture[2])
    #print("orig R:" + str(r[0]))
    #print("orig G:" + str(g[0]))
    #print("orig B:" + str(b[0]))
    
    y = np.array(0.299*r + 0.587*g + 0.114*b)
    y = thresholdAndRound(y)
    u = ((b-y)*0.564) + delta
    v = ((r-y)*0.713) + delta
    
    #print("orig Y:" + str(y[0]))
    #print("orig U:" + str(u[0]))
    #print("orig V:" + str(v[0]))
    
    y = thresholdAndRound(y)
    u = thresholdAndRound(u)
    v = thresholdAndRound(v)
    y = convertToBytes(y)
    u = convertToBytes(u)
    v = convertToBytes(v)
    
    yuv = np.concatenate((y,u,v), axis = 0)
    yuv = yuv.reshape((width*height*3), )
    
    #print(y)
    #print(v)
    
    return yuv

def YUV444_2_YUV420(data, width, height):
    from scipy import signal
    #print("YUV444_2_YUV420")
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    #shape = u.shape
    #print("The old shape of u: "+ str(shape))
    #print(u)
    
    kernel = np.array([[1,1,0],
                       [1,1,0],
                       [0,0,0]])
        
    # Perform 2D convolution with input data and kernel
    u = signal.convolve2d(u, kernel, mode='same')/kernel.sum()
    u = u[::2, ::2].copy()
    v = signal.convolve2d(v, kernel, mode='same')/kernel.sum()
    v = v[::2, ::2].copy()

    y = y.flatten()
    u = u.flatten()
    v = v.flatten()

    #shape = u.shape
    #print("The new shape of u: "+ str(shape))
    yuv = np.concatenate([y,u,v])
    return yuv

def YUV420_2_YUV444(data, width, height):
    from scipy import signal
    #print("YUV420_2_YUV444")
    picture = np.array(data)
    picSize = width*height
    #picture = pic_planar.reshape(3, width, height)
    y = np.array(picture[0:picSize])
    
    u = np.array(picture[picSize:(picSize*5/4)])
    u = u.reshape((width/2), (height/2))
    #print("The old shape of u: "+ str(u.shape))
    #print(u)
    u = np.repeat(u, 2, axis=0)
    #print("The new shape of u: "+ str(u.shape))
    #print(u)
    u = np.repeat(u, 2, axis=1)
    #print("The new shape of u: "+ str(u.shape))
    #print(u)
    
    v = np.array(picture[(picSize*5/4):])
    v = v.reshape((width/2), (height/2))
    #print("The old shape of v: "+ str(v.shape))
    #print(v)
    v = np.repeat(v, 2, axis=0)
    #print("The new shape of v: "+ str(v.shape))
    #print(u)
    v = np.repeat(v, 2, axis=1)
    #print("The new shape of v: "+ str(v.shape))
    #print(v)
    
    
    y = y.flatten()
    u = u.flatten()
    v = v.flatten()
    
    #shape = u.shape
    #print("The new shape of u: "+ str(shape))
    yuv = np.concatenate([y,u,v])
    return yuv





# planar YUV 4:4:4 to rgb
def planarYUV_2_planarRGB(data, width, height):
    #print("in planarYUV_2_planarRGB")
    maxn = 255
    minn = 0
    delta = 128.0
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    #print("recon Y:" + str(y[0]))
    #print("recon U:" + str(u[0]))
    #print("recon V:" + str(v[0]))
    
    
    r = y + 1.403 * (v-delta)
    g = y - (0.714 * (v-delta)) - (0.344 * (u-delta))
    b = y + 1.773 * (u-delta)
    
    #r = y + 1.13983 * v
    #g = y - (0.58060 * v) - (0.39465 * u)
    #b = y + (2.03211 * u)
    
    
    r = thresholdAndRound(r)
    r = convertToBytes(r)
    g = thresholdAndRound(g)
    g = convertToBytes(g)
    b = thresholdAndRound(b)
    b = convertToBytes(b)
    #print("Reconstructed r:" + str(r[0]))
    #print("Reconstructed g:" + str(g[0]))
    #print("Reconstructed b:" + str(b[0]))
    
    rgb = np.concatenate((r,g,b), axis = 0)
    rgb = rgb.reshape((width*height*3), )
    return rgb

def quantiseUV(data, width, height):
    numLevels = 16
    q = 256/numLevels
    x = np.linspace(0, 10, 1000)
    
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    u = q * np.round(u/q)
    v = q * np.round(v/q)
    
    yuv = np.concatenate((y,u,v), axis = 0)
    yuv = yuv.reshape((width*height*3), )
    return yuv

def saveToFile(data, filename):
    datayuv = np.asarray(data, 'u1')
    yuvByteArray = bytearray(datayuv)
    mylen = len(yuvByteArray)
    yuvFile = open(filename, "wb")
    yuvFile.write(yuvByteArray)
    yuvFile.close()

def appendToFile(data, filename):
    datayuv = np.asarray(data, 'u1')
    yuvByteArray = bytearray(datayuv)
    mylen = len(yuvByteArray)
    #print("Adding bytes to file: "+str(mylen))
    yuvFile = open(filename, "ab")
    yuvFile.write(yuvByteArray)
    yuvFile.close()

def cropImagesFromYUVfile(filename, width, height, frameNos):
    yuvFile = open(filename, "rb")
    data = np.fromfile(yuvFile, dtype=np.uint8)
    #data = yuvFile.read()
    #mylen = len(data)
    #print("Length of data: "+ str(mylen))
    # ASSUME YUV420 TODO: YUV444
    frameSize = width*height*3/2
    frames = []
    for frameNo in frameNos:
        start = frameSize*frameNo
        finish = (frameSize*(frameNo+1))
        #print("Start: "+ str(start)+ " Finish: " + str(finish))
        frame = data[start:finish]
        frames.append(frame)
    yuvFile.close()
    return frames

def cropROIfromYUV444(data, c, wsrc, hsrc, w, h, x, y):
    dst = np.zeros((c,w,h), dtype=np.uint)
    data = np.array(data)
    data = data.reshape(c,wsrc,hsrc)
    dst[:,:,:] = data[:,x:(x+w), y:(y+h)].copy()
    return dst
