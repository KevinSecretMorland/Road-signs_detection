import numpy as np
import cv2
import matplotlib.pyplot as plt

image = np.random.rand(800, 600, 3)
image = cv2.imread('/Users/marc/Documents/Dataset/rbg/image/image_00020.png')
# print(image.shape)
[h, w, d] = image.shape


i = np.array([w-40, h-40, 20])
# print(i)
#cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 1)

if not (i[0] > w or i[1] > h):

    x = i[0] - i[2]
    y = i[1] - i[2]

    rect_width = i[0] + i[2]
    rect_height = i[1] + i[2]

    #w_flag = False
    #h_flag = False
    if rect_width >= w:
        rect_width = w
        #w_flag = True
    if rect_height >= h:
        rect_height = h
        #h_flag = True

    # cv2.rectangle(image, (x, y), (rect_width, rect_height), (0, 255, 0), 1)

    ROI = image[y:rect_height, x:rect_width, :]

    ROI = cv2.resize(ROI, (2*i[2], 2*i[2]))

    plt.imshow(ROI)
    plt.show()


else:
    print(f'Wrong circle coordinates, x={i[0]} and y={i[1]}')
