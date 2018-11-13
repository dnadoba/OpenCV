# Duplicate Car
import cv2
import numpy as np
import pathlib

car = cv2.imread("car.png")

# position of original car
left = 790
top = 170
right = 1060
bottom = 270

width = right - left
height = bottom - top

only_car = car[top:bottom, left:right, :]

# position of duplicated car
left_car_2 = 500
top_car_2 = 170
right_car_2 = left_car_2 + width
bottom_car_2 = top_car_2 + height

car[top_car_2:bottom_car_2, left_car_2:right_car_2, :] = only_car

pathlib.Path('./output').mkdir(parents=False, exist_ok=True)

cv2.imwrite("output/duplicated_car.png", car)
