import cv2
import numpy
import matplotlib
from ObjectTracker import VehicleTracker


class CarCounter:

    def __init__(self, video_adr, detection_zone=(0, 0, 200, 200), rescale_pct=0.7):
        self.video_adr = video_adr
        self.rescale_pct = rescale_pct
        self.detection_zone = detection_zone

    def rescale_frame(self, frame, percent=0.9):
        """
        缩小原来大小的0.9倍
        :param frame:
        :param percent:
        :return:
        """
        width = int(frame.shape[1] * percent)
        height = int(frame.shape[0] * percent)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def filter_mask(self, img):
        """
        对视频进行预处理
        :param img:
        :return:
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel, iterations=1)
        retval, threshold = cv2.threshold(
            dilation, 155, 255, cv2.THRESH_BINARY)
        threshold = cv2.GaussianBlur(threshold, (5, 5), 1)

        return threshold

    def get_centroid(self, x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1

        return (cx, cy)

    def detect_vehicles(self, fg_mask, min_contour_width=20, min_contour_height=30):

        matches = []
        contours, hierarchy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)
            if not contour_valid:
                continue
            centroid = self.get_centroid(x, y, w, h)
            matches.append(((x, y, w, h), centroid))

        return matches

    def count_cars(self):
        try:
            vid_file = cv2.VideoCapture(self.video_adr)
        except:
            print("problem opening input stream")
            return

        if not vid_file.isOpened():
            print("capture stream not open")
            return

        back_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000.0)
        nFrames = int(vid_file.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid_file.get(cv2.CAP_PROP_FPS)

        ret, frame = vid_file.read()
        #
        ct = VehicleTracker(detection_zone=self.detection_zone)

        while ret:
            patch = self.rescale_frame(frame, percent=self.rescale_pct)
            fgmask = back_sub.apply(patch)
            filtered_img = self.filter_mask(fgmask)

            squares = self.detect_vehicles(filtered_img)
            cars = ct.update(squares)

            for key, car in cars.items():
                start_point = (car[0][0], car[0][1])
                end_point = (car[0][0]+car[0][2], car[0][1]+car[0][3])
                patch = cv2.rectangle(
                    patch, start_point, end_point, color=(100, 45, 255), thickness=2)
                patch = cv2.putText(
                    patch, f'carId:{key}', start_point, fontFace=4, fontScale=0.6, color=(0, 0, 0), thickness=1)

            patch = cv2.putText(patch, f'Vehicle Numbers:{ct.count}', (
                20, 20), fontFace=4, fontScale=0.6, color=(0, 0, 0), thickness=1)
            patch = cv2.rectangle(patch, (self.detection_zone[0], self.detection_zone[1]), (
                self.detection_zone[2], self.detection_zone[3]), color=(78, 252, 3), thickness=1)
            cv2.imshow("frameWindow", patch)
            cv2.imshow("fgmask", filtered_img)

            cv2.waitKey(int(600/fps))
            ret, frame = vid_file.read()

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
