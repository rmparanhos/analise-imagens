import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import skimage.exposure

# load the image
false_color = skimage.io.imread("Trabalho-Final/Foto 2/processing/753-redim-2.tif")
band6 = skimage.io.imread("Trabalho-Final/Foto 2/processing/LO82270682021258CUB00_B6 - cópia-2.tif")

fig, ax = plt.subplots()
plt.title("false color")
plt.imshow(false_color)
plt.show()

fig, ax = plt.subplots()
plt.title("band6")
plt.imshow(band6)
plt.show()

# create a histogram of the blurred grayscale image
gray_image = skimage.color.rgb2gray(band6)
plt.title("band6 grey")
plt.imshow(gray_image, cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))

plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()


# Contrast stretching
p2, p98 = np.percentile(gray_image, (2, 98))
gray_image_eq = skimage.exposure.rescale_intensity(gray_image, in_range=(p2, p98))
plt.title("band6 grey constrast stretch")
plt.imshow(gray_image_eq, cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(gray_image_eq, bins=256, range=(0.0, 1.0))

plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale contrast stertch Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

# create a mask based on the threshold
t = 0.5
binary_mask = gray_image_eq > t

fig, ax = plt.subplots()
plt.title("binary mask t")
plt.imshow(binary_mask, cmap='gray')
plt.show()

# use the binary_mask to select the "interesting" part of the image
selection = np.zeros_like(false_color)
selection[binary_mask] = false_color[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask t applied on false color")
plt.imshow(selection)
plt.show()

selection = np.zeros_like(band6)
selection[binary_mask] = band6[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask t applied on band6")
plt.imshow(selection)
plt.show()


rootPixels = np.count_nonzero(binary_mask)
print(rootPixels)
#resoluacao da banda 6 do landsat 8 é 30m
print(rootPixels*0.0009)

w = binary_mask.shape[1]
h = binary_mask.shape[0]
areaPiexels = w * h
print(areaPiexels)
print(areaPiexels*0.0009)

density = rootPixels / areaPiexels
print(density)

#outros metodos de trhehodl
fig, ax = skimage.filters.try_all_threshold(gray_image_eq, figsize=(10, 8), verbose=False)
plt.show()

#Trheshold otsu method
thresh = skimage.filters.threshold_otsu(gray_image_eq)
print(thresh)
binary_mask = gray_image_eq > thresh

fig, ax = plt.subplots()
plt.title("binary mask otsu")
plt.imshow(binary_mask, cmap='gray')
plt.show()

# use the binary_mask to select the "interesting" part of the image
selection = np.zeros_like(false_color)
selection[binary_mask] = false_color[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask otsu applied on false color")
plt.imshow(selection)
plt.show()

selection = np.zeros_like(band6)
selection[binary_mask] = band6[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask otsu applied on band6")
plt.imshow(selection)
plt.show()


rootPixels = np.count_nonzero(binary_mask)
print(rootPixels)
#resoluacao da banda 6 do landsat 8 é 30m
print(rootPixels*0.0009)

w = binary_mask.shape[1]
h = binary_mask.shape[0]
areaPiexels = w * h
print(areaPiexels)
print(areaPiexels*0.0009)

density = rootPixels / areaPiexels
print(density)

#Trheshold li method
thresh = skimage.filters.threshold_li(gray_image_eq)
print(thresh)
binary_mask = gray_image_eq > thresh

fig, ax = plt.subplots()
plt.title("binary mask li")
plt.imshow(binary_mask, cmap='gray')
plt.show()

# use the binary_mask to select the "interesting" part of the image
selection = np.zeros_like(false_color)
selection[binary_mask] = false_color[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask li applied on false color")
plt.imshow(selection)
plt.show()

selection = np.zeros_like(band6)
selection[binary_mask] = band6[binary_mask]

fig, ax = plt.subplots()
plt.title("binary mask li applied on band6")
plt.imshow(selection)
plt.show()


rootPixels = np.count_nonzero(binary_mask)
print(rootPixels)
#resoluacao da banda 6 do landsat 8 é 30m
print(rootPixels*0.0009)

w = binary_mask.shape[1]
h = binary_mask.shape[0]
areaPiexels = w * h
print(areaPiexels)
print(areaPiexels*0.0009)

density = rootPixels / areaPiexels
print(density)

