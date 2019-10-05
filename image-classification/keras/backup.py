


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
