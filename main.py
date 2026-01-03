# -*- coding: utf-8 -*-
"""
Created on Oct  12 2023

@author: https://github.com/LyyyyRan
"""

# division:
from __future__ import division

# some pkg:
import sys
import cv2

# for PyQt:
from PyQt5 import QtCore, QtGui, QtWidgets

# for yolo && MobileNetV3:
from utils.utils import drawLandmark_multiple
from predict import Pipeline

# For ignore warnings:
import warnings

# ignore warnings:
warnings.filterwarnings("ignore")


# MainClass:
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        # super(Ui_MainWindow, self).__init__(self)

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0

        # Detect:
        self.pipeline = Pipeline()  # YOLOv5 + MobileNetV2
        self.detect_flag = False  # default

        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数

    # 界面布局:
    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局

        # buttons:
        self.button_open_camera = QtWidgets.QPushButton('Open Camera')
        self.button_close = QtWidgets.QPushButton('Exit')
        self.button_detect = QtWidgets.QPushButton('Detect')

        # size of buttons:
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_detect.setMinimumHeight(50)

        self.button_close.move(10, 100)  # 移动按键

        # show image:
        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(640, 480)

        # 把按键加入到按键布局中
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.button_detect)

        # 把某些控件加入到总布局中
        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        # 显示所有控件
        self.setLayout(self.__layout_main)

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_detect.clicked.connect(self.detect_LyNet)  # 回调函数指针 指向神经网络模型函数
        self.button_close.clicked.connect(self.close)  # close() 由 QtWidgets.QWidget自带的

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                msg = QtWidgets.QMessageBox.warning(
                    self, 'warning', "Check connection of the camera", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(5)  # 定时器开始计时5ms，结果是每过5ms从摄像头中取一帧显示
                self.button_open_camera.setText('Close Camera')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('Open Camera')

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取

        if self.detect_flag:
            results = self.pipeline.predict(self.image)

            for item in results:
                print('lyyy:', 'face_time(s):', item['face_time'],
                      'face_score:', item['face_score'],
                      'mouth_time(s):', item['mouth_time'],
                      'mouth_score:', (item['mouth_score'], item['mouth_success'])
                      )

                # if Open Mouth:
                open_mouth_flag = (item['mouth_success'], item['mouth_score'])

                # DrawLandMark:
                self.image = drawLandmark_multiple(
                    self.image, item['face_bbox'], item['mouth_points'], flag=open_mouth_flag)

        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(
            show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    # 是否启动模型识别: (单击反转)
    def detect_LyNet(self):
        self.detect_flag = not self.detect_flag


# Run:
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
