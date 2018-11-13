import cv2
import numpy as np
import pathlib
import glob
from functools import reduce


def flatMap(f, items):
    return reduce(lambda x, y: x + y, map(f, items))


square_size = 0.02423
image_size = (360, 640)
pattern_size = (9, 6)
image_paths = glob.glob("./calib_images/*.jpg")
image_objects = list(flatMap(lambda y:
              list(map(lambda x:
                   np.array([
                      x * square_size,
                      y * square_size,
                      0
                  ]),
                  range(0, pattern_size[0]))),  # width
                        range(0, pattern_size[1])))      # height
image_objects = np.array(image_objects, dtype='float32')

all_object_points = []
all_image_points = []
all_images = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    (foundCorners, corners) = cv2.findChessboardCorners(image, pattern_size)

    # cv2.drawChessboardCorners(image, pattern_size, corners, foundCorners)
    if foundCorners:
        all_object_points.append(image_objects)
        all_image_points.append(corners)
        all_images.append(image)

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(all_object_points, all_image_points, image_size, None, None)

print(cameraMatrix)

def projectPointsAndShowOnImage():
    for (image, object_points, rvec, tvec) in zip(all_images, all_object_points, rvecs, tvecs):
        image_points, jacobian = cv2.projectPoints(np.array(object_points, dtype="float32"), rvec, tvec, cameraMatrix, distCoeffs)
        cv2.drawChessboardCorners(image, pattern_size, image_points, True)
        cv2.imshow("Projected Points", image)
        cv2.waitKey(0)

projectPointsAndShowOnImage()

cameraMatrix[0, 0] = input("Wert für fx:")

cameraMatrix[0, 2] = input("Wert für cx:")

projectPointsAndShowOnImage()
