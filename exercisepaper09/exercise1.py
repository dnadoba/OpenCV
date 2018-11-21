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
        vert_num = img.shape[0] * img.shape[1]
        f.write(ply_header % dict(vert_num=vert_num))
        for (y, (img_row, depth_row)) in enumerate(zip(img, depth)):
            for (x, (color, z)) in enumerate(zip(img_row, depth_row)):
                f.write(f"{x} {y} {z} {color[2]} {color[1]} {color[0]}\n")


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


for y in [5]:
    for block_size in [12]:
        min_disp = 1 # 0-10 Gibt minimale Disparität an
        num_disp = 16 * y # Gibt maxinale Disparität an; immer vielfaches von 16

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size)
        disparity = stereo.compute(img1, img2).astype(np.float32)/(y + 1/16) + 1
        print(disparity.min(), disparity.max())
        disparity_img = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/disparity y({y}) block_size({block_size}).png', disparity_img)

        depth = np.full(disparity.shape, f*tx, dtype=np.float32)/disparity

        depth_img = cv2.applyColorMap((depth * (1/depth.max()*255)).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'{output_dir}/depth y({y}) block_size({block_size}).png', depth_img)

        write_ply(f'{output_dir}/pointcloud y({y}) block_size({block_size}).ply', img1, depth*8)











