import numpy as np
import cv2


def to_homogeneous(x):
    return np.hstack((x, np.ones(1, dtype=x.dtype)))


def from_homogeneous(x):
    return x[:-1]/x[-1]


fx = 460.
fy = 460.
cx = 320.
cy = 240.

X1 = np.array([10, 10, 100], dtype=float)
X2 = np.array([33, 22, 111], dtype=float)
X3 = np.array([100, 100, 1000], dtype=float)
X4 = np.array([20, -100, 100], dtype=float)

hX1 = to_homogeneous(X1).reshape((4, 1))
hX2 = to_homogeneous(X2).reshape((4, 1))
hX3 = to_homogeneous(X3).reshape((4, 1))
hX4 = to_homogeneous(X4).reshape((4, 1))

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])

P = np.hstack((K, np.zeros((3, 1), dtype=float)))

# Aufgabe 1.1
Y1 = from_homogeneous(np.matmul(P, hX1)).reshape(2)
Y2 = from_homogeneous(np.matmul(P, hX2)).reshape(2)
Y3 = from_homogeneous(np.matmul(P, hX3)).reshape(2)
Y4 = from_homogeneous(np.matmul(P, hX4)).reshape(2)

# Aufgabe 1.2
imagePoints, jacobian = cv2.projectPoints(np.array([X1, X2, X3, X4]), np.eye(3, dtype=float),
                                          np.zeros(3, dtype=float), K, np.array([]))

Z1, Z2, Z3, Z4 = imagePoints.reshape(4, 2)
print("X1", X1, Y1, Z1, Y1 == Z1)
print("X2", X2, Y2, Z2, Y2 == Z2)
print("X3", X3, Y3, Z3, Y3 == Z3)
print("X4", X4, Y4, Z4, Y4 == Z4)
