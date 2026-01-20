## About

This project serves as a simple comparison of three different implementations of basic
methods for automatic counting and determining the size distribution 
of elements in images (automatic statistics of elements in microscopic images)

## Method Overview

The processing pipeline follows these steps:

1. **Preprocessing**
   - grayscale image input
   - enhancing image features to improve results of segmentation

2. **Segmentation**
   - dividing image into object and background
   - after segmentation image is binary

3. **Labeling**
   - connected component labeling

4. **Calculation of statistics**
   - all statistics are in form of `ImageObjectsStatistics`