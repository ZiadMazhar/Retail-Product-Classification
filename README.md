
MVP: Visual Comparator and Segmentor for SKU Identification

This project implements a Minimum Viable Product (MVP) that leverages a Siamese Network and Segment Anything Model (SAM)
to identify and verify instances of a particular Stock Keeping Unit (SKU), with a focus on Kellogg’s Coco Pops.
It includes custom dataset decomposition and inference logic for visual similarity and segmentation tasks.
![Screenshot 2025-05-07 114527](https://github.com/user-attachments/assets/fddb7744-d9b1-4969-b793-75c8bd943f01)



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
- main.py: Loads and tests a Siamese model on sample image pairs.
- 04_A_Deconstruction_of_the_MVP.pdf: Design document describing model architecture and dataset formulation.

Requirements
------------
- Python 3.8+
- TensorFlow, NumPy, Matplotlib

Run Instructions
----------------
1. Place the images and model weights in the correct directory.
2. Run the application:
   python main.py

Future Work
-----------
- SAM integration
- Dataset generation pipeline
- Multi-SKU support

License
-------
MIT License
