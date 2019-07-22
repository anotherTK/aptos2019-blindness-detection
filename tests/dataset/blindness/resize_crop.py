
"""Preprocess Blindness Dataset
: resize the image to 1024
: filter the contentless regions
"""

import os
import cv2


def resize_image(imgpath, desired_size=1024, min_size=200, save_dir=None):
    img = cv2.imread(imgpath)
    # padding the original image
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # RGB->gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # find the whole contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the max contour
    contours = max(contours, key=cv2.contourArea)
    # detect the rectangel bounding box
    x, y, w, h = cv2.boundingRect(contours)

    resized_img = img.copy()
    if w > min_size and h > min_size:
        resized_img = img[y:y+h, x:x+w] 
    else:
        print("Bounding box not found for {}".format(imgpath))

    height, width, _ = resized_img.shape
    if max([height, width]) > desired_size:
        ratio = float(desired_size / max([height, width]))
        resized_img = cv2.resize(resized_img, tuple([int(width * ratio), int(height * ratio)]), interpolation=cv2.INTER_CUBIC)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        savepath = os.path.join(save_dir, imgpath.split('/')[-1].replace('.png', '.jpg'))
        cv2.imwrite(savepath, resized_img)


if __name__ == "__main__":

    resize_image("work_dirs/temp/0f23c3028206.png", save_dir="work_dirs/temp")
    
