import numpy as np
import cv2
import os


def get_file_name(filepath):
    name = os.listdir(filepath)
    name = [os.path.join(filepath, i) for i in name if os.path.splitext(i)[1] in ['.jpg', '.png', '.jpeg']]
    return len(name), name


def split_ext(filepath):
    a, b = os.path.split(filepath)
    return a, os.path.splitext(b)


def get_data_name(filepath):
    names = []
    for root, j, file in os.walk(filepath):
        for name in file:
            path = os.path.join(root, name)
            names.append(path)
    return len(names), names


def convert(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])


def rename_file(file_path):
    new_name = 0
    for root, j, file in os.walk(file_path):
        for name in file:
            path = os.path.join(root, name)
            try:
                os.rename(path, os.path.join(root, str(new_name) + ".jpg"))
            except:
                print('命名方式已存在')
            new_name += 1
    return "good"


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(filename, src):
    cv2.imencode('.tiff', src)[1].tofile(filename)


def correct_size(size):
    return ((size[0] // 256 + 1 if size[0] % 256 != 0 else size[0] // 256) * 256,
            (size[1] // 256 + 1 if size[1] % 256 != 0 else size[1] // 256) * 256)


def cv_plot(img):
    cv2.namedWindow("demo", 0)
    cv2.resizeWindow("demo", 640, 480)
    cv2.imshow('demo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reg_mask(mask):
    """
    image: array of the picture
    position: the red box's position in the picture
    """
    threshold1 = 200
    threshold2 = 50
    B, G, R = cv2.split(mask)
    position = []
    rr = np.zeros(mask.shape[:2], dtype="uint8")
    gg = np.zeros(mask.shape[:2], dtype="uint8")
    bb = np.zeros(mask.shape[:2], dtype="uint8")
    for (i, r), g, b in zip(enumerate(np.nditer(R)), np.nditer(G), np.nditer(B)):
        if (r >= threshold1 and g >= threshold1 and b >= threshold1):
            index = i // R.shape[1]
            colunmn = i % R.shape[1]
            rr[index][colunmn] = 1
            gg[index][colunmn] = 1
            bb[index][colunmn] = 1
    return cv2.merge([bb, gg, rr])


def image_with_mask(image, mask):
    return cv2.add(cv2.bitwise_and(image, cv2.bitwise_not(mask)), mask)


def dilate(mask):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    red_box = cv2.dilate(mask, element)
    return red_box
