# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4

import matplotlib.pyplot as plt # install matplotlib (pip install matplotlib)
import cv2 # install opencv (pip install opencv-python)
from scipy import ndimage


def generate_plot(img):
    plt.imshow(img)
    plt.show()

def generate_plots(color_img, gray_img):
    plt.subplot(211)
    plt.imshow(color_img)
    plt.subplot(212)
    plt.imshow(gray_img, cmap=plt.get_cmap('gray'))
    plt.show()

# Use a single number
def resize_img(img, to_size=(28, 28)):
    return cv2.resize(img, to_size)


def invert_colors(img):
    return ~img# equivalently cv2.bitwise_not(img)


def write_img(img, name, ext="jpeg"):
    # ccreate folder if not exists
    # save the processed images
    cv2.imwrite(f"processed_imgs/{name}.jpeg", img)


def black_and_white_denoise(img, threshold=140): 
    # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    # We are using Otsu's Binarization 
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


def scale_and_flatten(img):
    return img.flatten() / 255.0


def crop_margins(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:,0]) == 0:
        img = np.delete(img,0,1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:,-1]) == 0:
        img = np.delete(img,-1,1)


def _resize_img(img, size=(20, 20)):
    channels, rows, cols = img.shape
    to_rows, to_cols = size

    if rows > cols:
        factor = to_rows/rows
        rows = to_rows
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols,rows))
    else:
        factor = to_cols/cols
        cols = to_cols
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))

    return img

def pad(img, size=(28, 28)):
    rows, cols = img.shape
    to_rows, to_cols = size
    colsPadding = (int(math.ceil((to_cols-cols)/2.0)),int(math.floor((to_cols-cols)/2.0)))
    rowsPadding = (int(math.ceil((to_rows-rows)/2.0)),int(math.floor((to_rows-rows)/2.0)))
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
    
    return img

def _getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def _shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def shift(img):
    shiftx,shifty = getBestShift(img)
    shifted = shift(img,shiftx,shifty)
    return shifted


def run(img):
    generate_plot(img)

    t, img = black_and_white_denoise(img)   
    generate_plot(img)

    img = cv2.resize(255-img, (28, 28))
    generate_plot(img)

    img = crop_margins(img)
    generate_plot(img)

    img = _resize_img(img)
    generate_plot(img)

    img = pad(img)
    generate_plot(img)

    img = shift(img)
    generate_plot(img)


if __name__ == "__main__":
    filename = "no_flash/all_data/1_1"
    colors = cv2.imread(f"imgs/{filename}.jpg")
    gray = cv2.imread(f"imgs/{filename}.jpg", cv2.IMREAD_GRAYSCALE)
    generate_plots(colors, gray)
    run(gray)

# - [x] 1. Read Image
# - [x] 2. Scale 
# - [x] 3. Invert colors (binary threshold 0 - 128; 128 - 255)
# - [x] 3.1. Denoise 
# - [ ] 4. Fit the numbers into a 20 x 20 pixel box (remove every row and column at 
#    the sides of the image which are completely black)
# - [ ] 5. Compute the resize factor 
# 6. But in the end we need a 28 x 28 image so pad, the pixels around using np.lib.pad
# 7. Center the inner box to be centered according center of mass

