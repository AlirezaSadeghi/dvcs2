Deep Video Captioner using Stochastic Scenes

This repository contains parts of the main source code of my MS.c thesis (Defended in January of 2018), called Deep Video Captioning using Deep Recurrent Neural Networks. The document can be provided to anyone on request.

This work & source code is heavily influenced by the work of the authors of the CVPR 2017 submission #601.  

The very high-level idea of this work is to try to detect scene-changes across frames in a video, and incorporate that information in the generation of the final video embedding. This is achieved using a stochastic step function defined in our model that detects sudden background changes.

This model is better or at worst-case on-par with the state of the art Video Captioning models, released before 2018, and the original model itself.

Feel free to contact me at alirezasadeghi71@gmail.com.

### Requirements
* [Theano](https://github.com/Theano/Theano) 0.9.0
* [Keras](https://github.com/fchollet/keras) 1.1.0, configured for using Theano as backend 

   Note: Be sure to have ```"image_dim_ordering": "th"``` and ```"backend": "theano"``` in your keras.json file.

### Dataset setup
This code comes with support to the Montreal Video Annotation Dataset (M-VAD) and to the MPII Movie Description dataset (MPII-MD). 

Before doing anything, follow the instructions for the the dataset of choice, since they're steps differ.

#### M-VAD
Request access and download the dataset from the [MILA website](https://mila.umontreal.ca/en/publications/public-datasets/m-vad/). Then create a folder ```datasets/M-VAD``` in the root of the project, and prepare three subfolders inside it:
* ```datasets/M-VAD/videos```. Put here all the videos, organized by movie as in the [repository from MILA](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/) (for instance, you should have ```datasets/M-VAD/videos/21_JUMP_STREET/video/21_JUMP_STREET_DVS20.avi```).
* ```datasets/M-VAD/annotations```. Create three subfolders here: ```train```, ```test```, ```val```, and put in each of them the .srt files corresponding to training ([download](https://drive.google.com/open?id=0ByiZnh0L1I6VNFFLWWZUYjI0bE0)), test ([download](https://drive.google.com/open?id=0ByiZnh0L1I6VdW1OQzU1MXFwanM)) and validation ([download](https://drive.google.com/open?id=0ByiZnh0L1I6VWVM3TWo4enAzUkk)) respectively.
* ```datasets/M-VAD/features```. Leave this folder empty.

Then, compute C3D and ResNet features by typing in a Python console:

	from datasets import MVAD
	dataset = MVAD()
    dataset.compute_c3d_descriptors()
    dataset.compute_resnet_descriptors()


#### MPII-MD
Request access and download the dataset from the [MPI website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/mpii-movie-description-dataset/). Then create a folder ```datasets/MPII-MD``` in the root of the project, and prepare three subfolders inside it:
* ```datasets/MPII-MD/jpgAllFrames```. Unpack here the package with the jpeg frames as provided by MPI. For instance, you should have ```datasets/MPII-MD/jpgAllFrames/0001_American_Beauty/0001_American_Beauty_00.00.51.926-00.00.54.129/0001.jpg```.
* ```datasets/MPII-MD/annotations```. Put here ```annotations-someone.csv```, ```dataSplit.txt``` and ```uniqueTestIds.txt```.
* ```datasets/MPII-MD/features```. Leave this folder empty.

Then, compute C3D and ResNet features by typing in a Python console:

	from datasets import MPII_MD
	dataset = MPII_MD()
    dataset.compute_c3d_descriptors()
    dataset.compute_resnet_descriptors()
    
   
Model evaluation is done using Pycoco, for which the source codes pycocoevalcap and pycocotools are included in this project. I'm sure there's a better way of incorporating them, but they do "good enough" for a one-man purely academic project. 