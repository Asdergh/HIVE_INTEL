import numpy as np
import matplotlib.pyplot as plt
import cv2
import random as rd



class ImageConvolution():
    
    def __init__(self, image_shape=(224, 224), filters_count=32, filters_size=(2, 2)) -> None:
        
        self.image_shape = image_shape
        self.filters_count = filters_count
        self.filters_size = filters_size

#kernel convolution
    def run_convolution(self, image):

        result_image = np.zeros(shape=self.image_shape)
        curent_filter_number = 0
        while curent_filter_number < self.filters_count:

            for px_i in range(self.image_shape[0]):
                for px_j in range(self.image_shape[1]):

                    conv_index_i = px_i + self.filters_size[0]
                    conv_index_j = px_j + self.filters_size[1]
                    
                    if ((conv_index_i == self.image_shape[0]) and 
                        (conv_index_j == self.image_shape[1])):
                        
                        break

                    if len(self.image_shape) == 3:

                        self.kernel = np.array([[1, 0, -1],
                                               [2, 0, -2],
                                               [1, 0, -1]])
                        try:
                            result_image[px_i: conv_index_i, px_j: conv_index_j, 0] = np.dot(self.kernel, image[px_i: conv_index_i, px_j: conv_index_j, 0])
                            result_image[px_i: conv_index_i, px_j: conv_index_j, 1] = np.dot(self.kernel, image[px_i: conv_index_i, px_j: conv_index_j, 1])
                            result_image[px_i: conv_index_i, px_j: conv_index_j, 2] = np.dot(self.kernel, image[px_i: conv_index_i, px_j: conv_index_j, 2])

                        except BaseException:
                            pass


                    else:

                        raise ValueError("image shape uncorrect!!!, must be (n, n, 3)")

                    curent_filter_number += 1

        return result_image

# summary convolution per one slice (n, n)
    def summary_convolution(self, image, start_random=True):

        # 3d image tensor mode

        if start_random == False:
            if len(image.shape) == 3:
                self.red_chanel = image[:, :, 0]
                self.green_chanel = image[:, :, 1]
                self.blue_chanale = image[:, :, 2]

                self.kernel_red = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1]))
                self.kernel_green = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1]))
                self.kernel_blue = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1]))

                result_filter = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1]))
                for pix_i in range(image.shape[0] // self.filters_size[0]):
                    for pix_j in range(image.shape[1] // self.filters_size[1]):

                        conv_index_i = pix_i + self.filters_size[0]
                        conv_index_j = pix_j + self.filters_size[1]

                        self.kernel_red[pix_i, pix_j] = np.sum(image[pix_i: conv_index_i, pix_j: conv_index_j, 0]) * 0.01
                        self.kernel_green[pix_i, pix_j] = np.sum(image[pix_i: conv_index_i, pix_j: conv_index_j, 1]) * 0.01
                        self.kernel_blue[pix_i, pix_j] = np.sum(image[pix_i: conv_index_i, pix_j: conv_index_j, 2]) * 0.01

                result_filter = self.kernel_red + self.kernel_green + self.kernel_blue
            #2d image tensor mode
            else:

                self.kernel = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1])) 
                result_filter =  result_filter = np.zeros(shape=(image.shape[0] // self.filters_size[0], image.shape[1] // self.filters_size[1]))
                for pix_i in range(image.shape[0] // self.filters_size[0]):
                    for pix_j in range(image.shape[1] // self.filters_size[1]):

                        conv_index_i = pix_i + self.filters_size[0]
                        conv_index_j = pix_j + self.filters_size[1]

                        self.kernel[pix_i, pix_j] = np.sum(image[pix_i: conv_index_i, pix_j: conv_index_j]) * 0.01
                
                result_filter = self.kernel
        
        else:

            random_start_point = rd.randint(0, image.shape[0])

            if len(image.shape) == 3:

                red_chanel = image[:, :, 0]
                green_chanel = image[:, :, 1]
                blue_chanel = image[:, :, 2]


                kernel_red = np.zeros(shape=(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0], 
                                             image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]))
                kernel_green = np.zeros(shape=(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0], 
                                               image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]))
                kernel_blue = np.zeros(shape=(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0], 
                                              image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]))
                print(kernel_red.shape)
                
                curent_kernel_index_i = 0
                curent_kernel_index_j = 0

                for px_i in range(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0]):
                    for px_j in range(image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]):
                        
                        conv_index_i = px_i + self.filters_size[0]
                        conv_index_j = px_j + self.filters_size[1]

                        
                        kernel_red[px_i, px_i] = np.sum(image[px_i: conv_index_i, px_j: conv_index_j, 0]) * 0.01
                        kernel_green[px_i, px_j] = np.sum(image[px_i: conv_index_i, px_j: conv_index_j, 1]) * 0.01
                        kernel_blue[px_i, px_j] = np.sum(image[px_i: conv_index_i, px_j: conv_index_j, 2]) * 0.01

                        
                        curent_kernel_index_i += 1
                    curent_kernel_index_j += 1

                result_filter = kernel_red + kernel_green + kernel_blue

            else:
                
                kernel = np.zeros(shape=(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0], 
                                             image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]))
                
                curent_kernel_index_i = 0
                curent_kernel_index_j = 0

                for px_i in range(image[random_start_point:, random_start_point:].shape[0] // self.filters_size[0]):
                    for px_j in range(image[random_start_point:, random_start_point:].shape[1] // self.filters_size[1]):

                        conv_index_i = px_i + self.filters_size[0]
                        conv_index_j = px_j + self.filters_size[1]

                        kernel[px_i, px_j] = np.sum(image[px_i: conv_index_i, px_j: conv_index_j]) * 0.01
                    
                        curent_kernel_index_i += 1
                    curent_kernel_index_j += 1
                
                result_filter = kernel
            
        print(result_filter.shape)
        return result_filter
    

    def canny_convolution(self, image):
        
        Gx_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])
        
        Gy_kernel = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])
        

        result_filter = np.zeros(shape=(image.shape[0], image.shape[1], 3))


        for px_i in range(image.shape[0]):
                for px_j in range(image.shape[1]):

                    result_filter[px_i, px_j] = np.dot(Gx_kernel, image[px_i, px_j])
                    result_filter[px_i, px_j] = np.dot(Gy_kernel, result_filter[px_i, px_j])
        
        return result_filter
                    
        


image_preprocessor = ImageConvolution(
    image_shape=(224, 224, 3),
    filters_count=100,
    filters_size=(3, 3)
)


img = cv2.imread("C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\elephanat_image.jpeg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (224, 224))
convolved_img = image_preprocessor.summary_convolution(img, start_random=True)
convolved_img1 = image_preprocessor.summary_convolution(convolved_img, start_random=True)
convolved_img2 = image_preprocessor.summary_convolution(convolved_img1, start_random=True)
convolved_img3 = image_preprocessor.summary_convolution(convolved_img2, start_random=True)

goog_image = cv2.imread("C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\octobus.jpg")
canny_conv = image_preprocessor.canny_convolution(goog_image)


plt.imshow(goog_image)
plt.show()
plt.imshow(canny_conv)
plt.show()
# plt.imshow(convolved_img, cmap="magma")
# plt.show()
# plt.imshow(convolved_img1, cmap="magma")
# plt.show()
# plt.imshow(convolved_img2, cmap="magma")
# plt.show()
# plt.imshow(convolved_img3, cmap="magma")
# plt.show()
# plt.imshow(img, cmap="magma")
# plt.show()


