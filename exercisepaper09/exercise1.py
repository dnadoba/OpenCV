import sys
import glob
import pathlib
import numpy as np
sys.path.append('../shared')
import siftdetector
from once import once
import cv2


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


def write_ply(file, img, depth):

    with open(file, 'w') as f:

        result = []
        for (y, (img_row, depth_row)) in enumerate(zip(img, depth)):
            for (x, (color, z)) in enumerate(zip(img_row, depth_row)):
                if z != 0:
                    result.append([x, y, z, color[2], color[1], color[0]])

        f.write(ply_header % dict(vert_num=len(result)))
        np.savetxt(f, result, '%f %f %f %d %d %d')


output_dir = 'output'
img1 = cv2.imread('images/KITTI14_31_left.png')
img2 = cv2.imread('images/KITTI14_31_right.png')
pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True)

f = 721.5
fx = f
fy = f
cx = 690.5
cy = 172.8

tx = baseline = 0.54  # meter

# Exercise 1

for y in [4]:  # [4, 5, 6]:
    for block_size in [12]:  # [4, 8, 12, 16]:
        min_disp = 1 # 0-10 Gibt minimale Disparität an
        num_disp = 16 * y # Gibt maxinale Disparität an; immer vielfaches von 16
        print(f"y({y}) block_size({block_size})")
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size)
        disparity = stereo.compute(img1, img2).astype(np.float32)/y

        disparity_img = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/disparity y({y}) block_size({block_size}).png', disparity_img)

        # Exercise 2
        ftx = np.full(disparity.shape, f*tx, dtype=np.float32)
        # ftx/disparity but set to zero where division by zero would occur
        depth = np.divide(ftx, disparity, out=np.zeros_like(ftx), where=disparity != 0)

        depth_img = cv2.applyColorMap((depth * (1/depth.max()*255)).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/depth y({y}) block_size({block_size}).png', depth_img)

        write_ply(f'{output_dir}/pointcloud y({y}) block_size({block_size}).ply', img1, depth)

        # Exercise 3

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size,
                                       speckleWindowSize=100, speckleRange=1)
        disparity = stereo.compute(img1, img2).astype(np.float32) / y

        disparity_img = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/reduced noise disparity y({y}) block_size({block_size}).png', disparity_img)

        ftx = np.full(disparity.shape, f * tx, dtype=np.float32)
        # ftx/disparity but set to zero where division by zero would occur
        depth = np.divide(ftx, disparity, out=np.zeros_like(ftx), where=disparity != 0)

        depth_img = cv2.applyColorMap((depth * (1 / depth.max() * 255)).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/reduced noise depth y({y}) block_size({block_size}).png', depth_img)

        write_ply(f'{output_dir}/reduced noise pointcloud y({y}) block_size({block_size}).ply', img1, depth)









