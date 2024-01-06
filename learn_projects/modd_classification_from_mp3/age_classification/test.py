import cv2
import numpy as np
import matplotlib.pyplot as plt






image = cv2.imread("C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\goog_image.jpeg")
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

result_image = np.zeros(shape=(image.shape[0], image.shape[1], 3))
for px_i in range(image.shape[0]):
    for px_j in range(image.shape[1]):
        result_image[px_i, px_j] = np.dot(kernel, image[px_i, px_j])
        result_image[px_i, px_j] = np.dot(kernel.T, result_image[px_i, px_j])

plt.imshow(image)
plt.show()
plt.imshow(result_image[:, :, 0])
plt.show()
plt.imshow(result_image[:, :, 1])
plt.show()
plt.imshow(result_image[:, :, 2])
plt.show()
plt.imshow(result_image)
plt.show()