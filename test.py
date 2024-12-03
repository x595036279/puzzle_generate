import math

import cv2
import numpy as np

# 读取groundtruth.txt文件
with open('./data/puzzles/3/groundtruth.txt', 'r') as f:
    lines = f.readlines()
# 存储每个碎片的平移距离和旋转角度
transforms = []
for line in lines:
    tx, ty, theta = map(float, line.split())
    transforms.append((tx, ty, theta))



# 计算变换矩阵的函数
def calculate_transform_matrix(tx1, ty1, theta1, tx2, ty2, theta2):
    theta1 = -theta1 / math.pi * 180
    theta2 = -theta2 / math.pi * 180
    # 构建第一张碎片的变换矩阵
    matrix1 = np.array([
        [np.cos(theta1), -np.sin(theta1), -tx1],
        [np.sin(theta1), np.cos(theta1), -ty1],
        [0, 0, 1]
    ])

    # 构建第二张碎片的变换矩阵
    matrix2 = np.array([
        [np.cos(theta2), -np.sin(theta2), -tx2],
        [np.sin(theta2), np.cos(theta2), -ty2],
        [0, 0, 1]
    ])

    # 计算两个变换矩阵的乘积，得到拼接的总变换矩阵
    total_matrix = np.dot(matrix2, matrix1)

    return total_matrix

m1 = np.array(
    [[-9.96613612e-01 , 8.22271731e-02 , 9.51450060e+02],
     [-8.22271731e-02 ,-9.96613612e-01 , 1.03318387e+03],
     [0,0,1]]
)
m2 = np.array(
    [[ 1.58541134e-01, -9.87352373e-01 , 9.08919186e+02],
 [ 9.87352373e-01 , 1.58541134e-01 ,-7.25090732e+01],
     [0,0,1]]
)
m3 = np.array(
    [
        [1,0,313],[0,1,0],[0,0,1]
    ]
)
total_matrix = np.dot(m2,m3)
print(total_matrix)
m1 = np.array([
    [m1[0][0],m1[0][1],m1[0][2]],
    [m1[1][0],m1[1][1],m1[1][2]],
    ])
m2 = np.array([
    [m2[0][0],m2[0][1],m2[0][2]],
    [m2[1][0],m2[1][1],m2[1][2]]
    ])
image1 = cv2.imread("data/puzzles/3/piece-0.png")
image2 = cv2.imread("data/puzzles/3/piece-1.png")
#cv2.imshow("2",dst_transformed2)
dst_transformed1 = cv2.warpAffine(image1, m1, (2000, 2000))
# cv2.circle(dst_transformed1, (322, 256), 5, (8, 248, 8), -1)
#cv2.imshow("1",dst_transformed1)


dst_transformed2 = cv2.warpAffine(image2, m2, (2000, 2000))
#cv2.circle(dst_transformed2, (9, 271), 5, (8, 248, 8), -1)
#cv2.imshow("2",dst_transformed2)
m3 = np.array([
    [1,0,313.],  # dx2-dx1
    [0,1,-15.],  #dy2-dy1
])

dst_transformed3 = cv2.warpAffine(dst_transformed2, m3, (2000, 2000))
dst_transformed3+=dst_transformed1
cv2.imshow("3",dst_transformed3)

cv2.waitKey()
cv2.destroyWindow()