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
from U_Net import u_net_predict
from location_correct import locate_and_correct, erode
from modelColour_identify import car_model
from openCV_functions import license_image
from openCV_functions import license_spilt
from openCV_functions import plt_show_raw
from openCV_functions import template_matching
from CarCounter import CarCounter


class Form:
    def __init__(self, form, length, width):
        """
        Windows UI Design
        :param form: windows frame
        :param length: length - 900
        :param width: width - 800
        """
        # Form Control Setting
        self.form = form
        self.length = length
        self.width = width
        # Set the initial position of form
        self.form.geometry("%dx%d+%d+%d" % (length, width, 480, 90))
        self.form.title(
            "Computer Vision-Based Vehicle Integrated Information Collection System")
        self.img_path = None

        # LabelFrame Control Setting
        self.frame_buttons = tk.LabelFrame(self.form, text="Button  Controls", font=('TIMES NEW ROMAN', '14', 'bold'),
                                           width=273, height=410, labelanchor="nw").place(x=575, y=45)  # buttons frame
        self.frame_canvas = tk.LabelFrame(self.form, text="Canvas  Controls", font=('TIMES NEW ROMAN', '14', 'bold'),
                                          width=750, height=200, labelanchor="nw").place(x=35, y=570)  # canvas frame

        # Labels Control Setting
        self.label_org = Label(self.form, text='Original Image:',
                               font=('TIMES NEW ROMAN', 14, 'bold')).place(x=35, y=15)  # original label
        self.label_locate = Label(
            self.form, text='License Plate Location:', font=('TIMES NEW ROMAN', 13, 'bold')).place(x=50, y=620)  # LP location label
        self.label_identify = Label(
            self.form, text='Identification Results:', font=('TIMES NEW ROMAN', 13, 'bold')).place(x=53, y=710)  # result label
        self.label_model = Label(self.form, text='Vehicle Model / Colour:', font=(
            'TIMES NEW ROMAN', 15, 'bold')).place(x=565, y=470)   # vehicle model... label
        self.label_result = Label(self.form, text='Result...', font=(
            'TIMES NEW ROMAN', 14)).place(x=565, y=510)  # vehicle model... recognition result

        # Canvas Control Setting
        self.canvas_org = Canvas(self.form, width=512, height=512,
                                 bg='white', relief='solid', borderwidth=1)  # The original image canvas
        self.canvas_org.place(x=35, y=42)
        self.canvas_locate_1 = Canvas(self.form, width=245, height=85,
                                      bg='white', relief='solid', borderwidth=1)  # License plate area canvas 1
        self.canvas_locate_1.place(x=235, y=590)
        self.canvas_identify_1 = Canvas(self.form, width=245, height=65,
                                        bg='white', relief='solid', borderwidth=1)  # License plate recognition canvas 1
        self.canvas_identify_1.place(x=235, y=690)
        self.canvas_locate_2 = Canvas(self.form, width=245, height=85,
                                      bg='white', relief='solid', borderwidth=1)  # License plate area canvas 2
        self.canvas_locate_2.place(x=510, y=590)
        self.canvas_identify_2 = Canvas(self.form, width=245, height=65,
                                        bg='white', relief='solid', borderwidth=1)  # License plate recognition canvas 2
        self.canvas_identify_2.place(x=510, y=690)

        # Buttons Control Setting
        self.btn_imgPath = Button(self.form, text='Select Image', font=('TIMES NEW ROMAN', '12', 'bold'),
                                  width=12, height=2, command=self.load_show_img).place(x=585, y=70)  # Select file button
        self.btn_blueRecog = Button(self.form, text='Blue Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                    width=12, height=2, command=self.recog).place(x=585, y=150)  # blue plate recognition button
        self.btn_clear = Button(self.form, text='Clear ', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.clear).place(x=720, y=310)  # Clear button
        self.btn_whiteRecog = Button(self.form, text='White Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                     width=12, height=2, command=self.predict_white).place(x=720, y=150)  # white plate
        self.btn_cvMethod = Button(self.form, text='OpenCV Match', font=('TIMES NEW ROMAN', '12', 'bold'),
                                   width=12, height=2, command=self.opevCV_matching).place(x=720, y=70)  # OpenCV method
        self.btn_greenRecogn = Button(self.form, text='Green Plates', font=('TIMES NEW ROMAN', '12', 'bold'),
                                      width=12, height=2, command=self.predict_green).place(x=585, y=230)  # Green  plate
        self.btn_exit = Button(self.form, text='Exit', font=('TIMES NEW ROMAN', '12', 'bold'),
                               width=12, height=2, command=self.quit_program).place(x=585, y=390)  # exit system
        self.btn_model = Button(self.form, text='Model / Colour', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.model_colour).place(x=720, y=230)  # make, model, colour button
        self.btn_video = Button(self.form, text='Detect in Video', font=('TIMES NEW ROMAN', '12', 'bold'),
                                width=12, height=2, command=self.counter_video).place(x=585, y=310)  # counting in video

        # Deep Learning Trained Models Setting
        self.unet_blue = keras.models.load_model('../models/unet.h5')  # Blue plate u_net model
        self.unet_white = keras.models.load_model('../models/unet_white.h5')  # Whit plate u_net model
        # self.unet_green20 = keras.models.load_model('../models/unet_green20.h5')  # Green plate u_net model
        self.cnn = keras.models.load_model('../models/cnn.h5')  # blue plate cnn  model
        print('The program is loading... Wait for a moment, please...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])

    def opevCV_matching(self):
        """
        Select a vehicle image and using traditional method to recognize
        :return: recognition result
        """
        self.canvas_org.delete('all')  # Delete the image on the canvas
        # Step 1. Image Pre-processing
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        self.img_path = file_path
        img_open = Image.open(self.img_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.canvas_org.create_image(
            258, 258, image=self.img_Tk, anchor='center')
        original_image = cv.imread(file_path)

        image = original_image.copy()
        carLicense_image = license_image(image)

        # Step 2. Split the License Numbers and Characters
        try:
            image = carLicense_image.copy()
            word_images = license_spilt(image)
        except:
            tk.messagebox.showerror(
                'Error', 'Unable to segment license plate area!')
        else:
            # Step 3. Characters and Numbers Matching with Template
            word_images_ = word_images.copy()
            result = template_matching(word_images_)
            print('Recognition Result:', result)
            tk.messagebox.showinfo('Result', result)

            # Step 4. Image Rendering
            width, weight = original_image.shape[0:2]
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
        """
        Recognize vehicle make, model,colour and year
        :return: return recognition results
        """
        if self.img_path == None:  # Make predictions before you have selected an image
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            result = car_model(self.img_path)
            self.label_result = Label(self.form, text=result, font=(
                'TIMES NEW ROMAN', 16, 'bold'), fg="red").place(x=565, y=510)

    def load_show_img(self):
        """
        Load vehicle images
        :return:
        """
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        try:
            self.img_path = Entry(
                self.form, state='readonly', text=sv).get()  # Gets the open image
            img_open = Image.open(self.img_path)   # open image
            if img_open.size[0] * img_open.size[1] > 240 * 80:  # not licence plate
                img_open = img_open.resize((512, 512), Image.ANTIALIAS)
            self.img_Tk = ImageTk.PhotoImage(img_open)
            self.canvas_org.create_image(
                258, 258, image=self.img_Tk, anchor='center')
        except:
            tk.messagebox.showerror('Error', 'Failed to open a image!')

    def recog(self):
        """
        recognize and display the result
        :return:
        """
        if self.img_path == None:  # Not select image
            self.canvas_identify_1.create_text(
                32, 15, text='Please select image', anchor='nw', font=('TIMES NEW ROMAN', 14))
        else:
            img_origin = cv.imdecode(np.fromfile(
                self.img_path, dtype=np.uint8), -1)
            h, w = img_origin.shape[0], img_origin.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # A picture is a license plate, no location required
                lic = cv.resize(img_origin, dsize=(240, 80), interpolation=cv.INTER_AREA)[:, :, :3]  # resize to (240,80)
                img_origin_copy, Lic_img = img_origin, [lic]
            else:  # Prediction of the original image by U-Net, to get img_mask, to LP location and recognition
                img_origin, img_mask = u_net_predict(
                    self.unet_blue, self.img_path)
                img_origin_copy, Lic_img = locate_and_correct(img_origin, img_mask)  # License plate positioning and correction

            # CNN model is used to predict the license plate recognition
            Lic_pred = cnn_predict(self.cnn, Lic_img)

            if Lic_pred:
                # img_origin_copy[:, :, ::-1]  Converting BGR to RGB
                img = Image.fromarray(img_origin_copy[:, :, ::-1])
                self.img_Tk = ImageTk.PhotoImage(img)
                self.canvas_org.delete('all')  # Clear the drawing board before displaying
                # The outline of the positioned license plate is drawn and displayed on the drawing board
                self.canvas_org.create_image(258, 258, image=self.img_Tk, anchor='center')
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

            else:  # is null indicating that it is not recognized
                self.canvas_identify_1.create_text(
                    47, 15, text='Recognize Failed', anchor='nw', font=('TIMES NEW ROMAN', 22))

                # img_origin_copy[:, :, ::-1] Converting BGR to RGB
                img = Image.fromarray(img_origin_copy[:, :, ::-1])
                self.img_Tk = ImageTk.PhotoImage(img)
                self.canvas_org.delete('all')  # Clear the drawing board before displaying
                # The outline of the positioned license plate is drawn and displayed
                self.canvas_org.create_image(258, 258, image=self.img_Tk, anchor='center')

    def predict_white(self):
        """
        Use the trained unet model to locate the white license plate area
        :return:
        """
        if self.img_path == None:  # Make predictions before selected an image
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            img_origin, img_mask = u_net_predict(
                self.unet_white, self.img_path)
            img2gray = erode(img_mask)
            ret, mask = cv.threshold(img2gray, 200, 255, cv.THRESH_BINARY)
            cv.waitKey(0)
            gray2img = cv.cvtColor(img2gray, cv.COLOR_GRAY2BGR)
            img_origin_copy, Lic_img = locate_and_correct(img_origin, gray2img)
            # cv.imshow('show', img_mask)
            cv.imshow('License Plate', img_origin_copy)
            cv.waitKey(0)
            cv.destroyAllForms()
            self.clear()

    def predict_green(self):
        """
        Use the trained unet model to locate the white license plate area
        :return:
        """
        if self.img_path == None:  # Make predictions before selected an image
            tk.messagebox.showerror(
                'Error', 'Please select a license plate image!')
        else:
            img_origin, img_mask = u_net_predict(
                self.unet_green20, self.img_path)
            img2gray = erode(img_mask)
            ret, mask = cv.threshold(img2gray, 200, 255, cv.THRESH_BINARY)
            cv.waitKey(0)
            gray2img = cv.cvtColor(img2gray, cv.COLOR_GRAY2BGR)
            img_origin_copy, Lic_img = locate_and_correct(img_origin, gray2img)
            # cv.imshow('show', img_mask)
            cv.imshow('License Plate', img_origin_copy)
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
        """
        Clear all the data
        :return:
        """
        self.canvas_org.delete('all')
        self.canvas_locate_1.delete('all')
        self.canvas_locate_2.delete('all')
        self.canvas_identify_1.delete('all')
        self.canvas_identify_2.delete('all')
        self.img_path = None

    def quit_program(self):
        """
        Exit the system
        :return:
        """
        quit = tk.messagebox.askokcancel(
            'Prompt', 'Confirm to exit the program?')
        if quit == True:
            self.form.destroy()

    def closeEvent():  # Clear session() before closing
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    form = Tk()
    length = 900  # Window width set to 1000
    width = 800  # Window height set at 600
    Form(form, length, width)
    form.protocol("WM_DELETE_WINDOW", Form.closeEvent)
    form.mainloop()
