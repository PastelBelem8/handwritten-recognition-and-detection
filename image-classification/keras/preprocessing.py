https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
# 1. Read Image
# 2. Scale 
# 3. Invert colors (binary threshold 0 - 128; 128 - 255)

# 4. Fit the numbers into a 20 x 20 pixel box (remove every row and column at 
        the sides of the image which are completely black)
# 5. Compute the resize factor 
# 6. But in the end we need a 28 x 28 image so pad, the pixels around using np.lib.pad
# 7. Center the inner box to be centered according center of mass

