import math

import cv2
import numpy as np
m1 = np.array(
    [[-0.6412001365394222 ,-0.7673736931259935, 792.4207899599218],
     [0.7673736931259935 ,-0.6412001365394222 ,287.488899883018],
     [0, 0, 1]]
)
m2 = np.array(
    [[0.7499172658437602, 0.6615316276561682, -135.36668596147643],
     [-0.6615316276561682, 0.7499172658437602, 299.92112503628226],
     [0, 0, 1]]
)
m3 = np.array(
    [
        [1, 0, 69-(-193.)],
        [0, 1, -42-(-43.)],
        [0, 0, 1.]
    ]
)
#,)
total_matrix = np.dot(np.linalg.inv(m1),np.dot(m3,m2))
print(total_matrix)

image2 = cv2.imread("data/puzzles/70/piece-1.png")
m = np.array([
    [total_matrix[0][0],total_matrix[0][1],total_matrix[0][2]],
    [total_matrix[1][0],total_matrix[1][1],total_matrix[1][2]]
])


dst_transformed3 = cv2.warpAffine(image2, m, (2000, 2000))
cv2.imshow("2",dst_transformed3)


image1 = cv2.imread("data/puzzles/70/piece-0.png")
image1 = cv2.warpAffine(image1, np.array([
    [1.,0,0],
    [0,1,0]
]), (2000, 2000))



cv2.imshow("1",image1)
dst_transformed3+=image1
cv2.imshow("res",dst_transformed3)
cv2.waitKey()
cv2.destroyWindow()