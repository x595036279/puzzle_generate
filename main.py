import os
import sys
import platform
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QDialog, QVBoxLayout, \
    QLabel, QComboBox, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap

import argparse

from src.generator import PuzzleGenerator


def generate_puzzle(args):
    generator = PuzzleGenerator(args.img_path)
    for i in range(args.sample_n):
        generator.run(args.piece_n, args.offset_h, args.offset_w, args.small_region, args.rotate)
        generator.save(args.bg_color)


class NumberInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("选择碎片数量")

        layout = QVBoxLayout()

        label = QLabel("请选择要分割的碎片数量：", self)
        layout.addWidget(label)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["9 张", "36 张", "66 张", "100 张", "120 张", "150 张", "200 张", "自定义"])
        self.combo_box.currentIndexChanged.connect(self.on_combo_box_changed)
        layout.addWidget(self.combo_box)

        self.custom_input = QLineEdit(self)
        self.custom_input.setPlaceholderText("输入自定义数量")
        self.custom_input.setEnabled(False)
        layout.addWidget(self.custom_input)

        self.btn_confirm = QPushButton("确定", self)
        self.btn_confirm.clicked.connect(self.accept)
        layout.addWidget(self.btn_confirm)

        self.setLayout(layout)

        # self.map = {
        #     9: 12,
        #     36: 45,
        #     66: 70,
        #     100: 110,
        #     120: 130,
        #     150: 165,
        #     200: 220
        # }

    def on_combo_box_changed(self, index):
        if index == self.combo_box.count() - 1:
            self.custom_input.setEnabled(True)
        else:
            self.custom_input.setEnabled(False)

    def get_selected_number(self):
        selected_text = self.combo_box.currentText()
        if selected_text == "自定义":
            custom_text = self.custom_input.text()
            number = int(custom_text) if custom_text.isdigit() else 0
        else:
            number = int(selected_text.split()[0])
            # number = self.map[number]
        # print("number = ", number)
        return number


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("碎片分割")
        self.setGeometry(50, 50, 1000, 1000)

        self.btn_upload = QPushButton("上传图片", self)
        self.btn_upload.setGeometry(0, 0, 200, 60)
        self.btn_upload.clicked.connect(self.upload_image)

        self.btn_process = QPushButton("分割图片", self)
        self.btn_process.setGeometry(202, 0, 200, 60)
        self.btn_process.clicked.connect(self.process_image)

        self.btn_open_folder = QPushButton("打开文件夹", self)
        self.btn_open_folder.setGeometry(404, 0, 200, 60)
        self.btn_open_folder.clicked.connect(self.open_folder)

        self.image_path = None

        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 70, 800, 800)  # 设置图片显示区域

        self.uploaded_images = []
        self.current_image_index = 0

        self.btn_prev_image = QPushButton("<", self)
        self.btn_prev_image.setGeometry(0, 61, 30, 30)
        self.btn_prev_image.clicked.connect(self.show_previous_image)

        self.btn_next_image = QPushButton(">", self)
        self.btn_next_image.setGeometry(31, 61, 30, 30)
        self.btn_next_image.clicked.connect(self.show_next_image)

    def open_folder(self):
        folder_path = "train_data/puzzles"
        if folder_path:
            if platform.system() == "Windows":  # 判断当前操作系统是 Windows
                os.system("start {}".format(folder_path))  # 在Windows下打开文件夹
            elif platform.system() == "Darwin":  # 判断当前操作系统是 macOS
                os.system("open {}".format(folder_path))  # 在 macOS 下打开文件夹
            elif platform.system() == "Linux":  # 判断当前操作系统是 Linux
                os.system("xdg-open {}".format(folder_path))  # 在 Linux 下打开文件夹

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio))

    def show_previous_image(self):
        if self.uploaded_images:
            self.current_image_index -= 1
            if self.current_image_index < 0:
                self.current_image_index = len(self.uploaded_images) - 1
            self.show_image(self.uploaded_images[self.current_image_index])

    def show_next_image(self):
        if self.uploaded_images:
            self.current_image_index += 1
            if self.current_image_index >= len(self.uploaded_images):
                self.current_image_index = 0
            self.show_image(self.uploaded_images[self.current_image_index])

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        # print(files)
        if files:
            self.image_path = files
            self.uploaded_images = files
            self.current_image_index = 0
            self.show_image(self.uploaded_images[self.current_image_index])
            QMessageBox.information(self, "上传成功", "当前已上传 {} 张图片".format(len(self.image_path)), QMessageBox.Ok)

    def process_image(self):
        if self.image_path:
            dialog = NumberInputDialog(self)
            if dialog.exec_():
                number_of_fragments = int(dialog.get_selected_number())
                print("选择的碎片数量：", number_of_fragments)

            for pic in self.image_path:
                self.parser = argparse.ArgumentParser(description='A tool for generating puzzles.')
                self.parser.add_argument(
                    '-t', '--sample-n', default=1, type=int,
                    help='Number of puzzle you want to generate from the input image. Default is 1.')
                self.parser.add_argument('--offset-h', default=1, type=float,
                                         help='Provide the horizontal offset rate when chopping the image. Default is 1. \
                                                            The offset is the rate of the initial rigid piece height. If the value is less than \
                                                            0.5, no interaction will happen.')
                self.parser.add_argument('--offset-w', default=1, type=float,
                                         help='Provide the vertical offset rate when chopping the image. Default is 1. \
                                                            The offset is the rate of the initial piece width. If the value is less than \
                                                            0.5, no interaction will happen.')
                self.parser.add_argument('-s', '--small-region', default=0.25, type=float,
                                         help='A threshold controls the minimum area of a region with respect to initial rigid \
                                                            piece area. Default is 0.25.')
                self.parser.add_argument('-r', '--rotate', default=180, type=float,
                                         help='A range of random rotation (in degree) applied on puzzle pieces. Default is 180. \
                                                            The value should be in [0, 180]. Each piece randomly select a rotation degree in [-r, r]')
                self.parser.add_argument('--bg_color', default=[8, 248, 8], type=int, nargs=3,
                                         help='Background color to fill the empty area. Default is [0, 0, 0]. The type is three uint8 \
                                                            numbers in BGR OpenCV format.')
                self.parser.add_argument('-n', '--piece-n', default=number_of_fragments, type=int,
                                         help='Number of puzzle pieces. Default is 100. The actual number of puzzle pieces may be different.')
                self.parser.add_argument('-i', '--img-path', default=f'{pic}', type=str,
                                         help='Path to the input image.')
                args = self.parser.parse_args()
                args.bg_color = tuple(args.bg_color)
                generate_puzzle(args)
            QMessageBox.information(self, "分割完成", "已成功分割 {} 张图片".format(len(self.image_path)), QMessageBox.Ok)
            # self.show_image("tmp/mask_init.png")
        else:
            QMessageBox.information(self, "错误", "请先上传图片！", QMessageBox.Ok)
            # print("请先上传图片！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
