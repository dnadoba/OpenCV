# Color Channels
import cv2
import numpy as np
import pathlib

car = cv2.imread("car.png")

# 1.2.a
b, g, r = cv2.split(car)

# 1.2.b
# b, g, r = car[:, :, 0], car[:, :, 1], car[:, :, 2]

image_shape = b.shape
zero_channel = np.zeros(image_shape, dtype="uint8")

# create new image with only one channel set and the are set to zero
# if we only create an image with one image, OpenCV will interpret it as a greyscale image
blue_channel = cv2.merge((b, zero_channel, zero_channel))
green_channel = cv2.merge((zero_channel, g, zero_channel))
red_channel = cv2.merge((zero_channel, zero_channel, r))

pathlib.Path('./output').mkdir(parents=False, exist_ok=True)

cv2.imwrite("output/blue_car.png", blue_channel)
cv2.imwrite("output/green_car.png", green_channel)
cv2.imwrite("output/red_car.png", red_channel)
