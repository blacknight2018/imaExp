import cv2
import numpy as np

img = cv2.imread("d:/2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


# print(gauss(3,1))
jj = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
jj = gauss(3, 0.5)
jj = np.array(jj)
# print(np.sum(jj))
sp = img.shape
img2 = np.copy(img)
tmp = []
for x in range(sp[0]):
    # print(x)
    for y in range(sp[1]):
        # print(y)
        k = img2[x][y]
        if x - 1 >= 0 and y - 1 >= 0 and x + 1 <= sp[0] - 1 and y + 1 <= sp[1] - 1:
            cur = jj[0][0] * img2[x - 1][y - 1]
            cur += jj[0][1] * img2[x - 1][y]
            cur += jj[0][2] * img2[x - 1][y + 1]
            cur += jj[1][0] * img2[x][y - 1]
            cur += jj[1][1] * img2[x][y]
            cur += jj[1][2] * img2[x][y + 1]

            cur += jj[2][0] * img2[x + 1][y - 1]
            cur += jj[2][1] * img2[x + 1][y]
            cur += jj[2][2] * img2[x + 1][y + 1]
            img2[x][y] = cur

# R G B
cv2.imshow("img", img)
cv2.imshow("img2", img2)
cv2.waitKey(0)