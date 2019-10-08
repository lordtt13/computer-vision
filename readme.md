# Computer Vision Models
## Description:
> All models (optimized,pre-built and trained) for my work are listed here

## Pre-Built
- detect_gender          - CLI for gender detection on static 2-D image
- gender_detection_model - Pre-Built Model 
- gender_detect_webccam  - CLI for gender detection on video frames
- smallervggnet          - Small VGG Net Model Architecture
- train                  - Builder Script

## Facial Recognition
- data_generator        - Script which generates pictures using burst mode through camera
- Training              - Training for similarity using LBPH Recognizer
- main                  - Main App which gives similarity metric

## Trained
> Built on easily availaible datasets and images generated from OpenVINO Toolkit Models

- build_imfdb            - Build Database Script
- opt_model.h5           - Optimized Model
- SMALL_VGG_NET          - Initial Model Build
- smallvggnet            - optimized Model Build
- train_model            - Training Scipt for Model with multiple outputs
- init                   - Model,Graph Initializer
- convertor              - .svg to .png convertor
- deploy.prototxt        - caffemodel specs
- gen                    - qrcode generator
- integrator             - Model(s) integrator
- new_face_detection     - Script for GUI Build and interface
- res10_300x300_ssd      - CaffeModel
- new_model              - Model with OpenVINO Trainers
- xception               - Xception Net with IMFDB Trainers
- Age_Gender_Continous   - Age Gender Module (Densenet121) with Continous Age 
- Age_Gender_Categorical - Age Gender Module (Densenet121) with Softmax Ages

## Human Detector
> Built on imageai library with RetinaNet,YoloV3

- hum_detector              - Human Detector on pictures
- hum_detector_pic_resnet50 - Same as above but with Model Loader Changed 
- hum_detector_self         - Detector on vid frames
