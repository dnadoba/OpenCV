import sys
sys.path.append('../shared')
import siftdetector
from once import once
import glob
import pathlib
import numpy as np
import cv2

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

bf = cv2.BFMatcher()
matches = bf.match(descriptors[0], descriptors[1])

image_with_all_matches = np.hstack((images[0], images[1]))
cv2.drawMatches(images[0], detected_keypoints[0], images[1], detected_keypoints[1], matches, image_with_all_matches)
cv2.imwrite(f"{output_dir}/{image_name_prefix}_all_matches.png", image_with_all_matches)
if show_images:
    cv2.imshow("Matches", image_with_all_matches)
    cv2.waitKey(0)

matches = sorted(matches, key=lambda x: x.distance)
best_30_matches = matches[0:30]
pts1 = []
pts2 = []
for idx, match in enumerate(best_30_matches):
    pts1.append(detected_keypoints[0][match.queryIdx])
    pts2.append(detected_keypoints[1][match.trainIdx])
    match.queryIdx = idx
    match.trainIdx = idx

image_with_30_best_matches = np.hstack((images[0], images[1]))
cv2.drawMatches(images[0], pts1, images[1], pts2, best_30_matches, image_with_30_best_matches)
cv2.imwrite(f"{output_dir}/{image_name_prefix}_best_30_matches.png", image_with_30_best_matches)
if show_images:
    cv2.imshow("Matches", image_with_30_best_matches)
    cv2.waitKey(0)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors[0], descriptors[1], k=2)
good = []
pts1 = []
pts2 = []
theshold_matching = 0.7

for m, n in matches:
    if m.distance < theshold_matching*n.distance:

        good.append([m])
        pts1.append(detected_keypoints[0][m.queryIdx])
        pts2.append(detected_keypoints[1][m.trainIdx])
        m.queryIdx = len(pts1) - 1
        m.trainIdx = len(pts2) - 1


image_knn_matches = np.hstack((images[0], images[1]))
cv2.drawMatchesKnn(images[0], pts1, images[1], pts2, good, image_knn_matches)
cv2.imwrite(f"{output_dir}/{image_name_prefix}_knn_matches.png", image_knn_matches)
if show_images:
    cv2.imshow("Matches", image_knn_matches)
    cv2.waitKey(0)
