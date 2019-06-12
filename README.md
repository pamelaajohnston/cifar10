# cifar10
An investigation into cifar10, related to the paper [Pamela Johnston, Eyad Elyan, and Chrisina Jayne. “Spatial Effects of Video
Compression on Classification in Convolutional Neural Network” in 2018 International
Joint Conference on Neural Networks (IJCNN), pages 1370 - 1377. IEEE,
2018.](https://doi.org/10.1109/IJCNN.2018.8489370)

You'll need Python, Tensorflow and x264, and ffmpeg.

`ILSVRC_PSNR.py` - a way of comparing the JPEG images and the actual video in ImageNet - the actual video is slightly higher quality whereas the JPEG 'frames' have been recompressed and you can eyeball the difference.

`averageResults.py` for accumulating results of the neural network (average accuracy over several runs). Needs the stdout of the 'cifar10' neural network.

`cifar10....py` - the actual neural network  (and all the selectable architectures)

`classify_image_with_inceptionNet.py` altered to work with YUV files...works along with `editLabels.py` to do STL10 (only the labelled images) with inception.

`examineQP.py` and `examineQP_lonelyFrame.py` needs a modified JM decoder which outputs the QP for each macroblock, averages them and plots them (specifically to see what the average QP of the ImageNet 2015/2017 video dataset was)

`image2vid.py` takes a single image (from CIFAR-10 or STL-10), adds a "moving horizontal lines" border (think ZX Spectrum loading screen) to turn it into a short video.

`inspect_checkpoint.py` lets you get a visualisation of the first layer filters.

`functions.py`, `yuvview.py` libraries of useful functions for YUV colourspace.
 
 `refactorData.py`, `refactorData-CIFAR10_munge.py` arrange the small images of the dataset in a different way.
 
 Some of the other files are the starting point for files that ended up in the IPnet repository.
