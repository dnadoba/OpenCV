import pathlib
import numpy as np
import cv2

show_img = False
f = 721.5
fx = f
fy = f
cx = 690.5
cy = 172.8

K = np.array([
    [fx,  0, cx],
    [0,  fy, cy],
    [0,   0,  1],
], dtype=float)

tx = baseline = 0.54  # meter

R = np.eye(3, dtype=float)
t = np.array([tx, 0, 0], dtype=float).reshape(3, 1)


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d 
property float x
property float y
property float z
property uchar red  
property uchar green  
property uchar blue  
end_header
'''


def to_homogeneous(x):
    return np.hstack((x, np.ones(1, dtype=x.dtype)))


def from_homogeneous(x):
    return x[:-1]/x[-1]


def to_pointcloud_with_color(img, depth):
    result = []
    height = img.shape[0]
    max = depth.max() - 0.1
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            color = img[y][x]
            z = depth[y][x]
            Y = height - y
            if z < max:
                X = ((x - cx) * z) / fx
                Y = ((Y - cy) * z) / fy
                result.append([X, Y, -z, color[2], color[1], color[0]])
    return result

def write_ply(file, colored_pointcloud):

    with open(file, 'w') as f:

        f.write(ply_header % dict(vert_num=len(colored_pointcloud)))
        np.savetxt(f, colored_pointcloud, '%f %f %f %d %d %d')

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
img1 = cv2.imread('images/KITTI14_31_left.png')
img2 = cv2.imread('images/KITTI14_31_right.png')
image_shape = img1.shape
pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True)

# Exercise 1

y = 5
block_size = 12
min_disp = 1 # 0-10 Gibt minimale Disparität an
num_disp = 16 * y # Gibt maxinale Disparität an; immer vielfaches von 16

stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size,
                               speckleWindowSize=100, speckleRange=1)
disparity = stereo.compute(img1, img2).astype(np.float32) / 16

disparity_img = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(f'{output_dir}/reduced noise disparity y({y}) block_size({block_size}).png', disparity_img)

ftx = np.full(disparity.shape, f * tx, dtype=np.float32)
# ftx/disparity but set to zero where division by zero would occur
depth = np.divide(ftx, disparity, out=np.zeros_like(ftx), where=disparity != 0)

print(depth.min(), depth.max())

depth_img = cv2.applyColorMap((depth * (1 / depth.max() * 255)).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(f'{output_dir}/reduced noise depth y({y}) block_size({block_size}).png', depth_img)

colored_pointcloud = to_pointcloud_with_color(img1, depth)

write_ply(f'{output_dir}/pointcloud.ply', colored_pointcloud)

translations = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.3, 0.3, 0.3]),
    np.array([0.6, 0.6, 0.6]),
    np.array([0.9, 0.9, 0.9]),
    np.array([1.2, 1.2, 1.2]),
]

for t in translations:
    print(t)
    def translate(point_and_color):
        point = np.array(point_and_color[0:3], dtype=float)
        color = np.array(point_and_color[3:])
        point += t
        return np.hstack((point, color))


    colored_pointcloud = list(map(translate, colored_pointcloud))
    # write_ply(f'{output_dir}/pointcloud t({t}).ply', colored_pointcloud)
    image = project_pointcloud_to_image(colored_pointcloud,
                                        image_shape,
                                        K,
                                        np.eye(3, dtype=float),
                                        np.array([tx, 0, 0], dtype=float).reshape(3, -1))
    print(image)
    print(image.shape)
    print(image.min())
    print(image.max())
    cv2.imwrite(f'{output_dir}/projected image t({t}).png', image)
    if show_img:
        cv2.imshow('projected image', image)
        cv2.waitKey(0)










