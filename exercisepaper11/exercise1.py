import pathlib
import numpy as np
import cv2

show_img = False
fx = 363.58
fy = 363.53
cx = 250.32
cy = 212.55

scale = 5000

K = np.array([
    [fx,  0, cx],
    [0,  fy, cy],
    [0,   0,  1],
], dtype=float)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d 
property float x
property float y
property float z
end_header
'''


def to_homogeneous(x):
    return np.hstack((x, np.ones(1, dtype=x.dtype)))


def from_homogeneous(x):
    return x[:-1]/x[-1]


def to_pointcloud(depth):
    result = []
    height = depth.shape[0]
    max = depth.max() - 0.1
    for y in range(0, depth.shape[0]):
        for x in range(0, depth.shape[1]):
            z = depth[y][x]
            Y = height - y
            if z < max:
                X = ((x - cx) * z) / fx
                Y = ((Y - cy) * z) / fy
                result.append([X, Y, -z])
    return result

def write_ply(file, pointcloud):

    with open(file, 'w') as f:

        f.write(ply_header % dict(vert_num=len(pointcloud)))
        np.savetxt(f, pointcloud, '%f %f %f')

def project_pointcloud_to_image(colored_pointcloud, image_shape, K, R, t):
    P = np.matmul(K, np.hstack((R, t)))
    print("P", P)
    image = np.zeros(image_shape, dtype=np.uint8)
    for colored_point in colored_pointcloud:
        point3d = np.array(colored_point[0:3])
        #print("point3d", point3d)
        color_rgb = np.array(colored_point[3:6])
        color_bgr = color_rgb[::-1]
        Xh = to_homogeneous(point3d)
        #print("Xh", Xh)
        xh = np.matmul(P, Xh)
        #print("xh", xh)
        x = from_homogeneous(xh)
        # print("x",x)
        if 0 <= x[1] <= image_shape[0] and 0 <= x[0] <= image_shape[1]:
            image[int(x[1])][int(x[0])] = color_bgr
    return image


output_dir = 'output'
pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True)
one_channel_depth_img = cv2.imread('images/CoRBS_E1.png', flags=cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Depth img", one_channel_depth_img)
# print(one_channel_depth_img.min(), one_channel_depth_img.max())
#
# depth_img = cv2.applyColorMap((one_channel_depth_img * (1 / one_channel_depth_img.max() * 255)).astype(np.uint8), cv2.COLORMAP_JET)
# cv2.imshow("Depth img colered", depth_img)
# cv2.waitKey()

pointcloud = to_pointcloud(one_channel_depth_img)

write_ply(f'{output_dir}/pointcloud.ply', pointcloud)










