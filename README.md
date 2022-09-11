# DeepFake-Detection-Contrastive-Learning
DeepFake Detection Contrastive Learning

# Phase 1
## 1. Extracting Faces
There are two methods for extraction faces from frames
1. Using Dlib library

    1.1 HOG + Linear SVM

    1.2 MMOD CNN

* The HOG + Linear SVM face detector will be faster than the MMOD CNN face detector but will also be less accurate as HOG + Linear SVM does not tolerate changes in the viewing angle rotation.
* For more robust face detection, use dlib’s MMOD CNN face detector. This model requires significantly more computation (and is thus slower) but is much more accurate and robust to changes in face rotation and viewing angle.

* Furthermore, if you have access to a GPU, you can run dlib’s MMOD CNN face detector on it, resulting in real-time face detection speed. The MMOD CNN face detector combined with a GPU is a match made in heaven — you get both the accuracy of a deep neural network along with the speed of a less computationally expensive model. For more detailes you can see this [link](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)

2. Multi-task Cascade Convolutional Neural Networks (MTCNN)

pytorch implementation of inference stage of face detection algorithm described in

[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

### Conclusion

MTCNN is better than Dlib library in aspects of Accuracy and Inference Time.

```
python extract_face.py
```


## Activate the env
```
conda activate ./venv
```
## Install New Lib
```
python -m pip install <name>
```
## Installing Dlib
```
conda install -c conda-forge dlib
```

