# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4

import matplotlib.pyplot as plt # install matplotlib (pip install matplotlib)
import cv2 # install opencv (pip install opencv-python)

filename = "all_numbers_4_people_with_flash"
colors = cv2.imread(f"imgs/{filename}.jpg")
gray = cv2.imread(f"imgs/{filename}.jpg", cv2.IMREAD_GRAYSCALE)


def generate_plot(img):
    plt.imshow(img)
    plt.show()

def generate_plots(color_img=colors, gray_img=gray):
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


def black_and_white_denoise(img, threshold=128): 
    # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    # We are using Otsu's Binarization 
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


def scale_and_flatten(img):
    return img.flatten() / 255.0


# ------------------- #
# Test - Original     #
# ------------------- #
generate_plots()

# ----------------- #
# Test - Resized    #
# ----------------- #
resized_colors = resize_img(colors)
resized_gray = resize_img(gray)

generate_plots(resized_colors, resized_gray)

# ------------------------- #
# Test - Inverted Colors    #
# ------------------------- #
inverted_colors = invert_colors(colors)
inverted_gray = invert_colors(gray)

generate_plots(inverted_colors, inverted_gray)


# ------------------------- #
# Test - Inverted Colors    #
# ------------------------- #
_, denoised_gray = black_and_white_denoise(gray)
generate_plot(denoised_gray)



# ------------------------------------ #
# Test - Other denoising algorithms    #
# ------------------------------------ #
img = cv2.medianBlur(gray,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



# - [x] 1. Read Image
# - [x] 2. Scale 
# - [x] 3. Invert colors (binary threshold 0 - 128; 128 - 255)
# - [x] 3.1. Denoise 
# - [ ] 4. Fit the numbers into a 20 x 20 pixel box (remove every row and column at 
#    the sides of the image which are completely black)
# - [ ] 5. Compute the resize factor 
# 6. But in the end we need a 28 x 28 image so pad, the pixels around using np.lib.pad
# 7. Center the inner box to be centered according center of mass

