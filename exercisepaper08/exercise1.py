import sys
import glob
import pathlib
import numpy as np
sys.path.append('../shared')
import siftdetector
from once import once
import cv2
from functools import reduce

def min_elm(smaller, itr):
    return reduce(lambda smallest, current: smallest if smaller(smallest, current) else current, itr)

def drawlines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    print(r, c)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        cv2.circle(img2, (int(pt1[0]), int(pt1[1])), 5, color, -1)
    return img1, img2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d 
property float x
property float y
property float z
end_header
'''
def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')


fx = 707
fy = 707
cx = 604
cy = 180
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1],
], dtype=float)

image_name_prefix = sys.argv[1] or "KITTI11"
show_images = False
output_dir = 'output'
image_pair_paths = glob.glob(f"images/{image_name_prefix}*.png")
threshold = 5

pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True)

detected_keypoints = []
descriptors = []
images = []

for filename in image_pair_paths:
    [cur_detected_keypoints, cur_descriptors] = once(siftdetector.detect_keypoints, filename, threshold)
    keypoints_cv2 = siftdetector.to_cv2_kplist(cur_detected_keypoints)
    descriptors_cv2 = siftdetector.to_cv2_di(cur_descriptors)
    detected_keypoints.append(keypoints_cv2)
    descriptors.append(descriptors_cv2)

    print(f"Found {len(keypoints_cv2)} keypoints in {filename}")
    img = cv2.imread(filename)
    images.append(img)
    image_with_keypoints = img.copy()
    cv2.drawKeypoints(img, keypoints_cv2, image_with_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_with_keypoints_filename = f"{output_dir}/{filename.replace('/', '_')}"
    cv2.imwrite(image_with_keypoints_filename, image_with_keypoints)
    if show_images:
        cv2.imshow("Keypoints", image_with_keypoints)
        cv2.waitKey(0)

threshold_matchings = [0.7, 0.8]
for threshold_matching in threshold_matchings:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors[0], descriptors[1], k=2)
    good = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < threshold_matching*n.distance:

            good.append([m])
            pts1.append(detected_keypoints[0][m.queryIdx])
            pts2.append(detected_keypoints[1][m.trainIdx])
            m.queryIdx = len(pts1) - 1
            m.trainIdx = len(pts2) - 1

    print(f"Found {len(good)} good matches with matching threshold of {threshold_matching}")


    image_knn_matches = np.hstack((images[0], images[1]))
    cv2.drawMatchesKnn(images[0], pts1, images[1], pts2, good, image_knn_matches)
    cv2.imwrite(f"{output_dir}/{image_name_prefix}_{threshold_matching}_knn_matches.png", image_knn_matches)
    if show_images:
        cv2.imshow("Matches", image_knn_matches)
        cv2.waitKey(0)

    # Exercise 1

    points1 = np.array(list(map(lambda keyPt: keyPt.pt, pts1)))
    points2 = np.array(list(map(lambda keyPt: keyPt.pt, pts2)))

    F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_LMEDS)
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]
    lines = cv2.computeCorrespondEpilines(points1, 2, F)

    lines = lines.reshape(-1, 3)
    img_with_lines1, img_with_lines2 = drawlines(images[0].copy(), images[1], lines, points1, points2)
    cv2.imwrite(f"{output_dir}/{image_name_prefix}_{threshold_matching}_epilines.png", img_with_lines1)
    if show_images:
        cv2.imshow("Image", img_with_lines1)
        cv2.waitKey(0)


    # Exercise 2

    E = K.T * np.mat(F) * K

    R1, R2, t = cv2.decomposeEssentialMat(E)
    combinations = [
        (R1,  t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]


    results = []
    for (R, t) in combinations:
        P0 = np.hstack((K, np.zeros((3,1), dtype=float)))
        P1 = np.matmul(K, np.hstack((R, t)))

        pointcloud_homo = cv2.triangulatePoints(P0, P1, points1.T, points2.T)
        pointcloud = cv2.convertPointsFromHomogeneous(pointcloud_homo.T).reshape(-1, 3)
        point_count_in_image = 0
        for point in pointcloud:
            z = point[2]
            if z >= 0:
                point_count_in_image += 1

        results.append((point_count_in_image, R, t, pointcloud))

    _, R, t, pointcloud = min_elm(lambda lhs, rhs: lhs[0] < rhs[0], results)

    write_ply(f"{output_dir}/{image_name_prefix}_{threshold_matching}_pointcloud.ply", pointcloud)









