import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def get_coutours(image, bg_color):
    '''
    Get the contours of the image
    '''
    # Set background color threshold
    lower_bg = np.array(bg_color)
    upper_bg = np.array(bg_color)

    # Create a background mask
    bg_mask = cv2.inRange(image, lower_bg, upper_bg)
    # Make the background part pure black and other parts white
    bg_mask = np.where(bg_mask == 255, 0, np.where(bg_mask == 0, 255, bg_mask))

    # Extract edge contours
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, bg_mask


if __name__ == '__main__':
    img_path = './demo.png'
    img = cv2.imread(img_path)
    bg_color = [8, 248, 8]
    contours, bg_mask = get_coutours(img, bg_color)
    # cv2.imshow("mask",bg_mask)
    # cv2.waitKey()
    # cv2.destroyWindow()
    # for item in contours[0]:
    #     cv2.circle(img, tuple(item[0]), 2, (99, 196, 250), -1)
    #     cv2.imshow("img", img)
    #     cv2.waitKey()
    #     print(item[0])
        # with open("contours.txt", "a+") as f:
        #     f.write(f"{item[0][0]} {item[0][1]}\n")
    points = []


    with open("contours.txt") as f:
        all_line = [line.rstrip() for line in f]
        for index, line in enumerate(all_line):
            m1, m2 = [t(s) for t, s in zip((
                int, int), line.split())]
            points.append([m1, m2])
            cv2.circle(img, tuple([m1, m2]), 2, (99, 196, 250), -1)
    x = []
    y = []
    # cv2.imshow("img", img)
    # cv2.waitKey()
    for item in points:
        x.append(item[0])
        y.append(item[1])

    plt.xlim(0, max(x) + 200 )
    plt.ylim(0, max(y) +200)
    x = np.array(x)
    y = np.array(y)
    plt.plot(x, y,  lw=1, label='origin')
    plt.legend()  # 显示label

    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴


    def func(x, a, b, c, d, e, f, g,h,i):
        return a * np.sin(b * x ** c + h) + d * np.cos(e * x ** f +i) \
               + g


    popt, pcov = curve_fit(func, x, y, maxfev=800000)  # p0 = 1是因为只有a一参数
    print(popt)  # 即参数a的最佳值
    print(pcov)

    y2 = func(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6],popt[7],popt[8])

    plt.plot(x, y2, c='r', label='拟合曲线')

    plt.show()
    # cv2.imshow("img", img)
    # cv2.waitKey()
