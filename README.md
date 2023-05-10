# CV-Classic
The repository contains python implementaions **from scrach** of some classic computer vision algorithms, including **harris detector** and the complete **SIFT detetor and descriptor**.

Only uses opencv-python libary to read images or rotate images, the completely algorithms are only builded on numpy. Even convolution and downsampling methods are implemented by myself.

# Warn
As known, python code's executing speed is too low. Combined with an image is often very large, such as (1200, 800, 3), the SIFT algorithm is very slow. It often takes 30 minutes or even more! So please test it of small size images, such as (250, 250, 3).
