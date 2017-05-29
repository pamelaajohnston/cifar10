from PIL import Image
import random
import os
import shlex, subprocess
#import yuvview
import numpy as np
import sys
import socket

def readConfigFile (filename):
    numArgs = 5
    empty = '', '', '', '', '', ''
    try:
        configFile = open(filename, "r")
    except:
        print("uh oh, no config file")
    print "getting hostname"
    hostname = socket.gethostname()
    # ignore any domain names for convenience since they can change
    hostname = hostname[: hostname.find('.')]
    print hostname
    print "reading file"
    line = configFile.readline()


    while line.startswith( '#' ) or line[: line.find(' ')] != hostname:
        line = configFile.readline()
        if line == '':
            print("uh oh, no entry for this machine here")
            return empty

    print "******************"
    mylist = [x.strip() for x in line.split('#', numArgs)]

    print mylist
    print "******************"
    machine = mylist[0]
    print "machine: {}".format(machine)
    srcdir = mylist[1]
    print "src: {}".format(srcdir)
    dstdirlocal = mylist[2]
    print "dstlocal: {}".format(dstdirlocal)
    dstdirnetwork = mylist[3]
    print "dstnetwork: {}".format(dstdirnetwork)
    x264 = mylist[4]
    print "x264: {}".format(x264)
    batchfiles = [x.strip() for x in mylist[numArgs].split('#')]
    print "batchfiles: {}".format(batchfiles)

    return machine, srcdir, dstdirlocal, dstdirnetwork, x264, batchfiles

def collate(srcdir, dstdir):
    copytree(srcdir, dstdir)

def copytree(src, dst, symlinks=False, ignore=None):
    import os
    from shutil import copy2, copystat, Error
    
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()
    
    try:
        os.makedirs(dst)
    except OSError, exc:
        # XXX - this is pretty ugly
        if "file already exists" in exc[1]:  # Windows
            pass
        elif "File exists" in exc[1]:        # Linux
            pass
        else:
            raise

    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                copy2(srcname, dstname)
            # XXX What about devices, sockets etc.?
        except (IOError, os.error), why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error, err:
            errors.extend(err.args[0])
    try:
        copystat(src, dst)
    except WindowsError:
        # can't copy file access times on Windows
        pass
    except OSError, why:
            errors.extend((src, dst, str(why)))
    if errors:
        raise Error, errors


