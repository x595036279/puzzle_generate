import os
import math
import json
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from glob import glob
from samples.read_puzzle import Puzzle
import cv2

from src import utils
from src.vector import Vector


class PuzzleGenerator:

    def __init__(self, img_path):

        self.img = cv2.imread(img_path)
        self.img_size = self.img.shape[:2]  # Height, Width, Channel
        self.aspect_ratio = self.img_size[0] / self.img_size[1]
        print(
            f"self.img_size[0]={self.img_size[0]},self.img_size[1]={self.img_size[1]},aspect_ratio={self.aspect_ratio}")
        self.raw_regions = 'train_data/raw_regions/'
        if not os.path.exists(self.raw_regions):
            os.mkdir(self.raw_regions)

        self.puzzle_folder = 'train_data/puzzles/'
        if not os.path.exists(self.puzzle_folder):
            os.mkdir(self.puzzle_folder)

    def get_smooth_curve(self, x_len, x_pt_n, x_offset, y_offset, x_step, smooth_flag=False, index=0, direction=0):
        # x_arr, y_arr = self.get_smooth_curve(self.img_size[1], self.w_n, offset_w, offset_h, piece_w, True)
        x_arr = []
        y_arr = []

        for i in range(x_pt_n + 1):

            if i == 0:
                x = 0
            elif i == x_pt_n:
                x = x_len - 1
            else:
                x = round(x_step * i + random.uniform(-x_offset, x_offset))
            y = round(random.uniform(-y_offset, y_offset))

            x_arr.append(x)
            y_arr.append(y)

        x_arr = list(set(x_arr))
        y_arr = y_arr[:len(x_arr)]
        x_arr.sort()

        # 自定义插值函数
        def custom_interpolation_function(x, i, direction):
            # [a, b, c, d, e, f, g] = [3.07827841e+01, 1.85495501e-01, 1.14060835e+00, - 4.19590444e+02,
            #  6.34090338e+02, - 9.63729545e-01, 4.91195476e+02]
            # return a * np.sin(b * x ** c) + d * np.cos(e * x ** f) + g
            # 设置随机种子，以确保每次运行得到相同的随机序列
            # 生成随机参数
            return np.sin(x)
            random_factor = np.random.uniform(0.5, 1.5)
            random_phase = np.random.uniform(0, 2 * np.pi)
            # 生成一个在[0, 1)范围内的随机数
            random_condition = np.random.rand()

            # return 10 * random_factor * np.cos(0.015 * x + random_phase) * (
            #         (-1) ** i) + i * random_factor * np.sin(0.02 * x + random_phase) + 1 / (x + 1)
            return 10 * np.cos(0.015 * x) * (
                    (-1) ** i) + i * np.sin(0.02 * x) + 1 / (x + 1)
            # if direction == 1:  # 垂直
            #     return 10 * np.cos(0.015 * x) * (
            #             (-1) ** i) + i * np.sin(0.02 * x) + 1 / (x + 1)
            #     if int(i + random_factor) % 3 != 0 and random_condition < 0.7:
            #         return 10 * random_factor * np.cos(0.015 * x + random_phase) * (
            #                 (-1) ** i) + i * random_factor * np.sin(0.02 * x + random_phase) + 1 / (x + 1)
            #         # 引入非线性变化，使曲线具有一些转角
            #     elif random_condition >= 0.7:
            #         return 10 * random_factor * np.cos(0.055 * x + random_phase) * np.sin(0.01 * x) * (
            #                 (-1) ** i) + i * random_factor * np.sin(0.002 * x + random_phase) * np.cos(0.02 * x) * (
            #                        (-1) ** i)
            #     else:
            #         return 20 * random_factor * np.cos(0.02 * x + random_phase) * ((-1) ** i) + np.exp(
            #             -random_factor * 2 * x) * random_phase + 1 / (x + 1)
            # else:  # 水平
            #     return 10 * random_factor * np.sin(0.02 * x + random_phase) * ((-1) ** i) + np.exp(-random_factor * x)
            #     if int(i + random_factor) % 5 == 0 and random_condition < 0.5:
            #         return 10 * random_factor * np.sin(0.04 * x + random_phase) * (
            #                 (-1) ** i) + i * random_factor * np.cos(0.02 * x + random_phase) * ((-1) ** i)
            #     elif random_condition >= 0.5:
            #         return 10 * random_factor * np.cos(0.015 * x + random_phase) * (
            #                 (-1) ** i) + i * random_factor * np.sin(0.03 * x + random_phase) * ((-1) ** i)
            #     return 10 * random_factor * np.sin(0.02 * x + random_phase) * ((-1) ** i) + np.exp(-random_factor * x)

        if smooth_flag:
            # if len(x_arr) >= 4:
            #     smooth_func = interpolate.interp1d(x_arr, y_arr, kind='cubic')
            # elif len(x_arr) == 3:
            #     smooth_func = interpolate.interp1d(x_arr, y_arr, kind='quadratic')
            # elif len(x_arr) == 2:
            #     smooth_func = interpolate.interp1d(x_arr, y_arr, kind='slinear')
            # else:
            #     raise ValueError("The length of cutting points in x_arr must be larger than 0.")
            x_arr = np.arange(0, x_len, dtype=np.int32)

            # 使用自定义插值函数进行变换
            y_arr = custom_interpolation_function(x_arr, index, direction).astype(np.int32)

        # plt.plot(x_arr, y_arr, 'r')
        # plt.plot(x_arr_s, y_arr_s, 'b')
        # plt.show()

        return x_arr, y_arr

    def get_mask(self, offset_rate_h, offset_rate_w):

        piece_h = self.img_size[0] / self.h_n
        print(f"piece_h = {piece_h}")
        piece_w = self.img_size[1] / self.w_n
        print(f"piece_w = {piece_w}")
        offset_h = piece_h * offset_rate_h
        offset_w = piece_w * offset_rate_w
        self.mask = utils.new_array(self.img_size, 0)

        # Vertical cuts
        for i in range(1, self.w_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[0], self.h_n, offset_h, offset_w, piece_h, True, index=i,
                                                 direction=1)
            len_of_array = len(y_arr)
            num_large_intervals = self.h_n
            # print(f" =======   num_large_intervals = {num_large_intervals}")
            num_small_intervals = 20
            # 划分y_arr为num_large_intervals个小区间，每个小区间长度相等
            large_interval_size = len_of_array // num_large_intervals
            d = 0
            for j in range(num_large_intervals):

                start = j * large_interval_size
                end = (j + 1) * large_interval_size if j < num_large_intervals - 1 else len_of_array
                # 在每个小区间内再随机划分成num_small_intervals个更小的区间
                sub_interval_size = (end - start) // num_small_intervals
                for j in range(num_small_intervals):
                    sub_start = start + j * sub_interval_size
                    sub_end = start + (j + 1) * sub_interval_size if j < num_small_intervals - 1 else end

                    # 随机选择递增、递减或不变
                    change_type = random.choice(['increase', 'decrease'])
                    # print(f"start = {start} end = {end} sub_start = {sub_start} sub_end = {sub_end} change_type {change_type}")
                    # 随机选择变换的幅度
                    magnitude = random.uniform(0.5, 1)
                    # 在每个更小的区间内进行不同幅度的递增、递减和不变的变换
                    y_arr[sub_start] = d
                    for k in range(sub_start + 1, sub_end):
                        if change_type == 'increase':
                            if y_arr[k] + d <= piece_w / 4:
                                y_arr[k] += d
                                d += magnitude
                            else:
                                y_arr[k] = d
                        elif change_type == 'decrease':
                            if y_arr[k] + d >= -piece_w / 4:
                                y_arr[k] += d
                                d -= magnitude
                            else:
                                y_arr[k] = d
            y_arr = y_arr + round(i * piece_w)
            y_arr = np.clip(y_arr, 0, self.img_size[1] - 1)

            for j in range(self.img_size[0]):
                self.mask[x_arr[j]][y_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[x_arr[j]][k] = 255
            # cv2.imwrite('./tmp/mask_init.png', np.array(self.mask, dtype=np.uint8))

        # Horizontal cuts
        for i in range(1, self.h_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[1], self.w_n, offset_w, offset_h, piece_w, True, index=i,
                                                 direction=0)

            len_of_array = len(y_arr)
            num_large_intervals = self.w_n
            num_small_intervals = 20
            # print(f" =======   num_large_intervals = {num_large_intervals}")
            # 划分y_arr为num_large_intervals个小区间，每个小区间长度相等
            large_interval_size = len_of_array // num_large_intervals
            d = 0
            for j in range(num_large_intervals):

                start = j * large_interval_size
                end = (j + 1) * large_interval_size if j < num_large_intervals - 1 else len_of_array
                # 在每个小区间内再随机划分成num_small_intervals个更小的区间
                sub_interval_size = (end - start) // num_small_intervals
                for j in range(num_small_intervals):
                    sub_start = start + j * sub_interval_size
                    sub_end = start + (j + 1) * sub_interval_size if j < num_small_intervals - 1 else end

                    # 随机选择递增、递减或不变
                    change_type = random.choice(['increase', 'decrease'])
                    # print(
                    #     f"start = {start} end = {end} sub_start = {sub_start} sub_end = {sub_end} change_type {change_type}")
                    # 随机选择变换的幅度
                    magnitude = random.uniform(0.3, 0.6)
                    # 在每个更小的区间内进行不同幅度的递增、递减和不变的变换
                    y_arr[sub_start] = d
                    for k in range(sub_start + 1, sub_end):
                        if change_type == 'increase':
                            if y_arr[k] + d <= piece_w / 4:
                                y_arr[k] += d
                                d += magnitude
                            else:
                                y_arr[k] = d
                        elif change_type == 'decrease':
                            if y_arr[k] + d >= -piece_w / 4:
                                y_arr[k] += d
                                d -= magnitude
                            else:
                                y_arr[k] = d

            y_arr = y_arr + round(i * piece_h)
            y_arr = np.clip(y_arr, 0, self.img_size[0] - 1)

            for j in range(self.img_size[1]):
                self.mask[y_arr[j]][x_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[k][x_arr[j]] = 255

        cv2.imwrite('./tmp/mask_init.png', np.array(self.mask, dtype=np.uint8))
        # cv2.imshow('mask', self.mask)
        # cv2.waitKey()

    def get_regions(self, small_region_area_ratio):

        dirs = [Vector(0, -1), Vector(0, 1), Vector(-1, 0), Vector(1, 0)]  # (x, y)
        small_region_area_limit = small_region_area_ratio * \
                                  self.img_size[0] * self.img_size[1] / (self.w_n * self.h_n)

        mask = np.invert(np.array(self.mask, dtype=np.uint8))

        self.region_cnt, self.region_mat, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
        stats = stats.tolist()

        # Remap region idx
        region_idx_map = -1 * np.ones(self.region_cnt, dtype=np.int32)
        region_new_cnt = 0

        for i in range(1, self.region_cnt):
            if stats[i][4] < small_region_area_limit:
                region_idx_map[i] = -1
            else:
                region_idx_map[i] = region_new_cnt
                region_new_cnt += 1

        self.region_mat = region_idx_map[self.region_mat]

        print('\tRegion cnt final (raw): %d (%d)' % (region_new_cnt, self.region_cnt - 1))
        self.region_cnt = region_new_cnt

        # Expand valid region to fill out the canvas
        bg_pts = np.transpose(np.nonzero(self.region_mat == -1)).tolist()
        self.region_mat = self.region_mat.tolist()
        que = []

        for bg_pt in bg_pts:
            cur_p = Vector(bg_pt[1], bg_pt[0])
            for dir in dirs:
                next_p = cur_p + dir
                if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_mat[next_p.y][next_p.x] == -1:
                    continue
                que.append(next_p)

        while len(que) > 0:
            cur_p = que.pop(0)
            for dir in dirs:
                next_p = cur_p + dir
                if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_mat[next_p.y][next_p.x] != -1:
                    continue
                self.region_mat[next_p.y][next_p.x] = self.region_mat[cur_p.y][cur_p.x]
                que.append(next_p)

        # Check the region mat
        unlabel_pts = np.transpose(np.nonzero(np.ma.masked_equal(self.region_mat, -1).mask))
        assert (unlabel_pts.size == 0)

        # for i in range(self.region_cnt):
        #     mask = np.ma.masked_equal(self.region_mat, i).mask.astype(np.uint8)
        #     mask = mask * 255
        #     cv2.imwrite('tmp/' + str(i) + '.png', mask)
        #     cv2.imshow('tmp', mask)
        #     cv2.waitKey(0)

    def save_raw_regions(self, iter):

        file_path = os.path.join(self.raw_regions, '%d.npy' % iter)
        np.save(file_path, np.array(self.region_mat, dtype=np.int32))

        f = open(file_path[:-3] + 'txt', 'w')
        f.write(str(self.region_cnt))
        f.close()
        print('\tSave to %s & %d.txt' % (file_path, iter))

    def save_puzzle(self, iter, bg_color):

        bg_mat = np.full(self.img.shape, bg_color, np.uint8)
        region_mat_np = np.array(self.region_mat, np.uint32)

        region_rgbs = []
        w_max = 0
        h_max = 0
        groundtruth = []

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        os.mkdir(puzzle_path)

        # Compute maximum boundary
        for i in range(self.region_cnt):
            region_map = region_mat_np == i
            region_map3 = np.repeat(region_map, 3).reshape(self.img.shape)
            rgb = np.where(region_map3, self.img, bg_mat)

            coords = np.argwhere(region_map)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            region_rgb = rgb[y0:y1, x0:x1]

            region_rgbs.append(region_rgb)
            groundtruth.append({
                'id': i + 1,
                'dx': int(x0),
                'dy': int(y0)
            })

            w_max = max(w_max, x1 - x0)
            h_max = max(h_max, y1 - y0)

        r = int(math.sqrt(w_max ** 2 + h_max ** 2) + 5)

        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.txt')
        # outfile = open(groundtruth_path, 'w')

        # Compute groundtruth
        # Save groundtruth in txt
        for i in range(self.region_cnt):
            pad_top = (r - region_rgbs[i].shape[0]) // 2
            pad_left = (r - region_rgbs[i].shape[1]) // 2
            pad_bottom = r - region_rgbs[i].shape[0] - pad_top
            pad_right = r - region_rgbs[i].shape[1] - pad_left

            region_pad = cv2.copyMakeBorder(region_rgbs[i],
                                            pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                            value=bg_color)
            # cv2.imshow("region_pad", region_pad)
            cv2.imwrite("region_pad.png", region_pad)

            degree = random.uniform(-self.rot_range, self.rot_range)
            # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)

            # 这个问题可能是由于旋转后图像边界的插值方式引起的。
            # 在这段代码中，cv2.warpAffine函数的参数borderMode=cv2.BORDER_CONSTANT指定了边界填充方式为常数填充，
            # 而borderValue=bg_color指定了填充颜色为bg_color。
            rotation_mat = cv2.getRotationMatrix2D((region_pad.shape[1] / 2, region_pad.shape[0] / 2), degree, 1)
            region_rot = cv2.warpAffine(region_pad, rotation_mat, (region_pad.shape[1], region_pad.shape[0]),
                                        flags=cv2.INTER_NEAREST, borderValue=bg_color)
            # region_rot = cv2.warpAffine(region_pad, rotation_mat, (region_pad.shape[1], region_pad.shape[0]),
            #                             borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            # cv2.imshow("region_rot", region_rot)
            cv2.imwrite("region_rot.png", region_rot)

            cv2.imwrite(os.path.join(puzzle_path, f'fragment_{i + 1:04d}.png'), region_rot)

            groundtruth[i]['dx'] -= pad_left
            groundtruth[i]['dy'] -= pad_top
            groundtruth[i]['rotation'] = degree / 180 * math.pi

            COS = math.cos(groundtruth[i]['rotation'])
            SIN = math.sin(groundtruth[i]['rotation'])

            # outfile.write('%d %d %.3f\n' % (groundtruth[i]['dx'], groundtruth[i]['dy'], groundtruth[i]['rotation']))
            # outfile.write('%d\n%f %f %f %f %f %f 0 0 1\n' % (i+1,COS,SIN,groundtruth[i]['dx'],-SIN,-COS,groundtruth[i]['dy']))
            # rgb = np.ma.masked_equal(self.region_mat == i, self.img)
            # cv2.imshow('region_rgb', region_rgbs[i])
            # cv2.imshow('region_pad', region_pad)
            # cv2.imshow('region_rot', region_rot)
            # cv2.waitKey()
            # print(rgb)
            # break

        # outfile.close()

        # Save groundtruth in json
        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.json')
        outfile = open(groundtruth_path, 'w')
        json.dump(groundtruth, outfile)
        outfile.close()

        # Save config file
        config_path = os.path.join(puzzle_path, 'config.txt')
        outfile = open(config_path, 'w')

        outfile.write('fragment_\n')  # Prefix
        outfile.write('%d\n' % self.region_cnt)  # Piece number
        outfile.write('%d %d %d\n' % (bg_color[0], bg_color[1], bg_color[2]))  # bg color in BGR

        outfile.close()

        bg_color_path = os.path.join(puzzle_path, 'bg_color.txt')
        outfile = open(bg_color_path, 'w')
        outfile.write('%d %d %d\n' % (bg_color[0], bg_color[1], bg_color[2]))
        outfile.close()

        folder = os.path.split(puzzle_path)[-1]

        puzzle = Puzzle(puzzle_path)
        puzzle.read_groundtruth()
        for i in range(len(puzzle.pieces)):
            puzzle.apply_groundtruth(i, folder, display=False)

    def save_zip(self, iter):

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        zip_path = os.path.join(puzzle_path, 'puzzle-%d.zip' % iter)

        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        for i in range(self.region_cnt):
            piece_name = 'piece-%d.png' % i
            zipf.write(os.path.join(puzzle_path, piece_name), piece_name)
        zipf.write(os.path.join(puzzle_path, 'config.txt'), 'config.txt')
        zipf.close()

    def run(self, piece_n, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=0):

        self.rot_range = rot_range
        self.piece_n = piece_n
        # self.w_n = math.floor(math.sqrt(piece_n / self.aspect_ratio * 2))
        # self.h_n = math.floor(self.w_n * self.aspect_ratio / 2)

        self.w_n = math.floor(math.sqrt(3 * self.img_size[1] * piece_n / (4 * self.img_size[0])))
        self.h_n = math.floor(math.sqrt(4 * self.img_size[0] * piece_n / (3 * self.img_size[1])))
        print(self.w_n)
        print(self.h_n)
        print("=====")
        # 调整块数，使得乘积接近指定的块数
        while self.w_n * self.h_n < piece_n:
            if self.img_size[0] < self.img_size[1]:
                self.h_n += 1
            else:
                self.w_n += 1

        print('\tInitial block in hori: %d, in vert: %d' % (self.w_n, self.h_n))
        print('\tOffset rate h: %.2f, w: %.2f, small region: %.2f, rot: %.2f' %
              (offset_rate_h, offset_rate_w, small_region_area_ratio, rot_range))

        self.get_mask(offset_rate_h, offset_rate_w)
        self.get_regions(small_region_area_ratio)

    def save(self, bg_color=(8, 248, 8)):

        exist_data_len = len(glob(os.path.join(self.raw_regions, '*.npy')))
        # print(f"exist_data_len = {exist_data_len} ============")
        self.save_raw_regions(exist_data_len + 1)
        self.save_puzzle(exist_data_len + 1, bg_color)
        # self.save_zip(exist_data_len)
