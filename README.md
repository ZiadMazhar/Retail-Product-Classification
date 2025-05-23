
MVP: Visual Comparator and Segmentor for SKU Identification

This project implements a Minimum Viable Product (MVP) that leverages a Siamese Network and Segment Anything Model (SAM)
to identify and verify instances of a particular Stock Keeping Unit (SKU), with a focus on Kellogg’s Coco Pops.
It includes custom dataset decomposition and inference logic for visual similarity and segmentation tasks.

![Figure_1](https://github.com/user-attachments/assets/d52ea21e-171e-480f-b826-95313b32ccf2)


Key Components
--------------

1. Segment Anything Model (SAM)
   - Used to segment relevant objects from images.

2. Siamese Network
   - Deep learning model designed to compute similarity between pairs of images.

3. Dataset Structure
   - Comparator Dataset (Xc): For Siamese training.
   - Segmentor Dataset (Xs): For segmentation testing.

File Overview
-------------
- sam_helper.py: loads and initiate the sam model 
- siamese_data_pipeline.py: load and augument dataset for training
- siamese.py: load embeding model (mobileNetv2) and build and train the custom siamese model (gradual unfreezing) 
- cropper.py: to crop the images
- sam.py: middleware between the sam_helper.py and main.py
- main.py: Loads and tests a Siamese model on sample image pairs.

Requirements
------------
- check Requirments.txt

Run Instructions
----------------
1. Run the application:
   python main.py

License
-------
MIT License
