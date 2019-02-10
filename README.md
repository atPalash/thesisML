# thesisML
This repository contains packages for object segmentation and object detection for the thesis.
The following packages are developed and utilised:

* helpers: Two helper Classes is defined here:
    * helpers.py : For retrieving object contours from an image
    * FrameTransformation.py : For converting object location from 
    camera frame to robot base frame.

* objectSelection : consists of the following:
    * images : Folder for storing training and test images of objects
    under experiment.
    * lib_realsense_examples: Folder containing example codes for using intel-realsense camera
    * models : Folder containing CNN weight files are stored in HDF5. 
    * pyimagesearch: Folder consists of CNN network is use lenet.py
    * objectSelectorOOP.py: Class RealSenseCamera is defined here responsible
    for object segmentation from image of workspace.
    * objectIdentifier.py : Class ObjectIdentifier uses models stored in models folder
    to predict the type of object present in workspace.
    * train_network.py : code for training CNN with the images of objects.
    * test_network.py : test accuracy of model created.

* reebGraph : Class ReebGraph finds the optimum grasping location
for a two fingered gripper. Main code for gripping algorithm.

* vision : consists of the following:
    * generateTrainingData.py : 