# Video Segmentation with SLIC Superpixels and SVM Classification

This is a project done as part of my Computer Vision class. It is a novel (as far as I know) video segmentation framework where the frames are over-segmented into superpixels (groups of pixels) using the SLIC algorithm, visual features are extracted from the superpixels and classification between foreground and background is done via support vector machines.

The implementation is done completely in C++ with the OpenCV library. It is performant enough to run real-time from a webcam stream.
