import cv2 as cv
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
from openCV_functions import license_image
from openCV_functions import template_matching
from openCV_functions import plt_show_raw
from openCV_functions import license_spilt


def opevCV_matching():
    # Step 1. Image Pre-processing
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
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
    height, weight = original_image.shape[0:2]
    image = original_image.copy()
    cv.rectangle(image, (int(0.2 * weight), int(0.75 * height)), (int(weight * 0.8), int(height * 0.95)), (0, 255, 0),
                 5)
    cv.putText(image, "".join(result), (int(0.2 * weight) + 30, int(0.75 * height) + 80), cv.FONT_HERSHEY_COMPLEX, 2,
               (0, 255, 0), 8)
    plt_show_raw(image)


def main():
    opevCV_matching()


if __name__ == '__main__':
    main()
