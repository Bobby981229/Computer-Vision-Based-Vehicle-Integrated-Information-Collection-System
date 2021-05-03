import tkinter as tk
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras

from CNN import cnn_predict
from Unet import unet_predict
from core import locate_and_correct, erode
from modelColour_identify import car_model
from openCV_functions import license_image
from openCV_functions import license_spilt
from openCV_functions import plt_show_raw
from openCV_functions import template_matching
from CarCounter import CarCounter


class Form:
    def __init__(self, form, length, width):
        """
        
        :param form:
        :param length:
        :param width:
        """
        # Form Control Setting
        self.form = form
        self.length = length
        self.width = width
        # Set the initial position of form
        self.form.geometry("%dx%d+%d+%d" % (length, width, 480, 90))
        self.form.title(
            "Computer Vision-Based Vehicle Integrated Information Collection System")
        self.imgOrg_path = None

        # LabelFrame Control Setting
        self.frame_buttons = tk.LabelFrame(self.form, text="Button  Controls", font=('TIMES NEW ROMAN', '14', 'bold'),
                                           width=273, height=410, labelanchor="nw").place(x=575, y=45)
        self.frame_canvas = tk.LabelFrame(self.form, text="Canvas  Controls", font=('TIMES NEW ROMAN', '14', 'bold'),
                                          width=750, height=200, labelanchor="nw").place(x=35, y=570)

        # Labels Control Setting
        self.label_org = Label(self.form, text='Original Image:',
                               font=('TIMES NEW ROMAN', 14, 'bold')).place(x=35, y=15)
        self.label_locate = Label(
            self.form, text='License Plate Location:', font=('TIMES NEW ROMAN', 13, 'bold')).place(x=50, y=620)
        self.label_identify = Label(
            self.form, text='Identification Results:', font=('TIMES NEW ROMAN', 13, 'bold')).place(x=53, y=710)
        self.label_model = Label(self.form, text='Vehicle Model / Colour:', font=(
            'TIMES NEW ROMAN', 15, 'bold')).place(x=565, y=470)
        self.label_result = Label(self.form, text='Result...', font=(
            'TIMES NEW ROMAN', 14)).place(x=565, y=510)

        # Canvas Control Setting
        self.canvas_org = Canvas(self.form, width=512, height=512,
                                 bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.canvas_org.place(x=35, y=42)
        self.canvas_locate_1 = Canvas(self.form, width=245, height=85,
                                      bg='white', relief='solid', borderwidth=1)  # 车牌区域1画布
        self.canvas_locate_1.place(x=235, y=590)
        self.canvas_identify_1 = Canvas(self.form, width=245, height=65,
                                        bg='white', relief='solid', borderwidth=1)  # 车牌识别1画布
        self.canvas_identify_1.place(x=235, y=690)
        self.canvas_locate_2 = Canvas(self.form, width=245, height=85,
                                      bg='white', relief='solid', borderwidth=1)  # 车牌区域2画布
        self.canvas_locate_2.place(x=510, y=590)
        self.canvas_identify_2 = Canvas(self.form, width=245, height=65,
                                        bg='white', relief='solid', borderwidth=1)  # 车牌识别2画布
        self.canvas_identify_2.place(x=510, y=690)

        # Buttons Control Setting
        self.btn_imgPath = Button(self.form, text='Select Image', font=('TIMES NEW ROMAN', '12', 'bold'),
                                  width=12, height=2, command=self.load_show_img).place(x=585, y=70)  # 选择文件按钮
        self.btn_blueRecog = Button(self.form, text='Blue Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                    width=12, height=2, command=self.display).place(x=585, y=150)  # 识别车牌按钮
        self.btn_clear = Button(self.form, text='Clear ', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.clear).place(x=720, y=310)  # 清空所有按钮
        self.btn_whiteRecog = Button(self.form, text='White Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                     width=12, height=2, command=self.predict_white).place(x=720, y=150)  #
        self.btn_cvMethod = Button(self.form, text='OpenCV Match', font=('TIMES NEW ROMAN', '12', 'bold'),
                                   width=12, height=2, command=self.opevCV_matching).place(x=720, y=70)  #
        self.btn_greenRecogn = Button(self.form, text='Green Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                      width=12, height=2, command=self.predict_green).place(x=585, y=230)  #
        self.btn_exit = Button(self.form, text='Exit', font=('TIMES NEW ROMAN', '12', 'bold'),
                               width=12, height=2, command=self.quit_program).place(x=585, y=390)  #
        self.btn_model = Button(self.form, text='Model / Colour', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.model_colour).place(x=720, y=230)  #
        self.btn_video = Button(self.form, text='Detect in Video', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.counter_video).place(x=585, y=310)  #

        # Deep Learning Trained Models Setting
        self.unet_blue = keras.models.load_model('unet.h5')
        self.unet_white = keras.models.load_model('unet_white.h5')
        # self.unet_green20 = keras.models.load_model('unet_green20.h5')
        self.cnn = keras.models.load_model('cnn.h5')
        print('The program is loading... Wait for a moment, please...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])

    def opevCV_matching(self):
        # Step 1. Image Pre-processing
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        self.imgOrg_path = file_path
        img_open = Image.open(self.imgOrg_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.canvas_org.create_image(
            258, 258, image=self.img_Tk, anchor='center')
        original_image = cv.imread(file_path)

        image = original_image.copy()
        carLicense_image = license_image(image)

        # Step 2. Split the License Numbers and Characters
        image = carLicense_image.copy()
        word_images = license_spilt(image)

        # Step 3. Characters and Numbers Matching with Template
        word_images_ = word_images.copy()
        result = template_matching(word_images_)
        print(result)
        tk.messagebox.showinfo('Result', result)

        # Step 4. Image Rendering
        width, weight = original_image.shape[0:2]
        # 中文无法显示
        image = original_image.copy()
        cv.rectangle(image, (int(0.2 * weight), int(0.75 * width)),
                     (int(weight * 0.8), int(width * 0.95)), (0, 255, 0), 5)
        cv.putText(image, "".join(result), (int(0.2 * weight) + 30,
                                            int(0.75 * width) + 80), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 8)
        plt_show_raw(image)

        cv.imshow('License Plate', image)
        cv.waitKey(0)
        cv.destroyAllForms()
        self.clear()

    def model_colour(self):
        if self.imgOrg_path == None:  # 还没选择图片就进行预测
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            result = car_model(self.imgOrg_path)
            self.label_result = Label(self.form, text=result, font=(
                'TIMES NEW ROMAN', 16, 'bold'), fg="red").place(x=565, y=510)

    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.imgOrg_path = Entry(
            self.form, state='readonly', text=sv).get()  # 获取到所打开的图片
        img_open = Image.open(self.imgOrg_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.canvas_org.create_image(
            258, 258, image=self.img_Tk, anchor='center')

    def display(self):
        if self.imgOrg_path == None:  # 还没选择图片就进行预测
            self.canvas_identify_1.create_text(
                32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
        else:
            img_src = cv.imdecode(np.fromfile(
                self.imgOrg_path, dtype=np.uint8), -1)  # 从中文路径读取时用
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                lic = cv.resize(img_src, dsize=(240, 80), interpolation=cv.INTER_AREA)[
                    :, :, :3]  # 直接resize为(240,80)
                img_src_copy, Lic_img = img_src, [lic]
            else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                img_src, img_mask = unet_predict(
                    self.unet_blue, self.imgOrg_path)
                img_src_copy, Lic_img = locate_and_correct(img_src,
                                                           img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正

            # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
            Lic_pred = cnn_predict(self.cnn, Lic_img)

            if Lic_pred:
                # img_src_copy[:, :, ::-1]将BGR转为RGB
                img = Image.fromarray(img_src_copy[:, :, ::-1])
                self.img_Tk = ImageTk.PhotoImage(img)
                self.canvas_org.delete('all')  # 显示前,先清空画板
                self.canvas_org.create_image(258, 258, image=self.img_Tk,
                                             anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        self.lic_Tk1 = ImageTk.PhotoImage(
                            Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.canvas_locate_1.create_image(
                            5, 5, image=self.lic_Tk1, anchor='nw')
                        self.canvas_identify_1.create_text(
                            35, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                    elif i == 1:
                        self.lic_Tk2 = ImageTk.PhotoImage(
                            Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.canvas_locate_2.create_image(
                            5, 5, image=self.lic_Tk2, anchor='nw')
                        self.canvas_identify_2.create_text(
                            40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))

            else:  # Lic_pred为空说明未能识别
                self.canvas_identify_1.create_text(
                    47, 15, text='未能识别', anchor='nw', font=('黑体', 27))

                # img_src_copy[:, :, ::-1]将BGR转为RGB
                img = Image.fromarray(img_src_copy[:, :, ::-1])
                self.img_Tk = ImageTk.PhotoImage(img)
                self.canvas_org.delete('all')  # 显示前,先清空画板
                self.canvas_org.create_image(258, 258, image=self.img_Tk,
                                             anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上

    def predict_white(self):
        """
        Use the trained unet model to locate the white license plate area
        :return:
        """
        if self.imgOrg_path == None:  # 还没选择图片就进行预测
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            img_src, img_mask = unet_predict(
                self.unet_white, self.imgOrg_path)  # 测试图片
            img2gray = erode(img_mask)
            ret, mask = cv.threshold(img2gray, 200, 255, cv.THRESH_BINARY)
            cv.waitKey(0)
            gray2img = cv.cvtColor(img2gray, cv.COLOR_GRAY2BGR)
            img_src_copy, Lic_img = locate_and_correct(img_src, gray2img)
            # cv.imshow('show', img_mask)
            cv.imshow('License Plate', img_src_copy)
            cv.waitKey(0)
            cv.destroyAllForms()
            self.clear()

    def predict_green(self):
        """
        Use the trained unet model to locate the white license plate area
        :return:
        """
        if self.imgOrg_path == None:  # 还没选择图片就进行预测
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            img_src, img_mask = unet_predict(
                self.unet_green20, self.imgOrg_path)  # 测试图片
            img2gray = erode(img_mask)
            ret, mask = cv.threshold(img2gray, 200, 255, cv.THRESH_BINARY)
            cv.waitKey(0)
            gray2img = cv.cvtColor(img2gray, cv.COLOR_GRAY2BGR)
            img_src_copy, Lic_img = locate_and_correct(img_src, gray2img)
            # cv.imshow('show', img_mask)
            cv.imshow('License Plate', img_src_copy)
            cv.waitKey(0)
            cv.destroyAllForms()
            self.clear()

    def counter_video(self):
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename()
        DETECTION_ZONE = (20, 400, 890, 500)
        cc = CarCounter(video_adr=video_path, detection_zone=DETECTION_ZONE)
        cc.count_cars()

    def clear(self):
        self.canvas_org.delete('all')
        self.canvas_locate_1.delete('all')
        self.canvas_locate_2.delete('all')
        self.canvas_identify_1.delete('all')
        self.canvas_identify_2.delete()
        self.imgOrg_path = None
        # self.label_result['text'] = "Result..."

    def quit_program(self):
        quit = tk.messagebox.askokcancel(
            'Prompt', 'Confirm to exit the program?')
        if quit == True:
            self.form.destroy()

    def closeEvent():  # 关闭前清除session(),防止'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    form = Tk()
    length = 900  # 窗口宽设定1000
    width = 800  # 窗口高设定600
    Form(form, length, width)
    form.protocol("WM_DELETE_WINDOW", Form.closeEvent)
    form.mainloop()
