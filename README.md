# Structure_From_Motion

# Overview

Structure from motion is a photogrammetric range imaging technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals.

Main objective : To implement the research paper Shape and motion from image streams under orthography: a factorization method

https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf

Different tracking and feature detection were used :
1. KLT tracker with goodfeaturestotrack
2. KLT tracker with SIFT/ORB .

Kindly run this post_tracking file in face_track with the required arguments, in matplotlib after rotating the desired output could be observed.

# Face_Reconstruction:
To track facial landmarks, dlib library was used and tomasi-kanade factorization was applied on created measurement matrix.

# Team Members
1. Abhay Khandelwal
2. Kalpit Jangid

