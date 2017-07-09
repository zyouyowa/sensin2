"""
画像を線画化する処理
"""

import cv2
import numpy as np

def make_contour_image(path):
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite('gray.jpg', gray)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    #cv2.imwrite('dilated.jpg', dilated)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)
    #cv2.imwrite('diff.jpg', diff)

    # 白黒反転
    contour = 255 - diff

    def downer(elem, threthold):
        if elem < threthold:
            return 0
        else:
            return 255

    vdowner = np.vectorize(downer)
    contour = vdowner(contour, 0.9*255)

    return contour

if __name__ == '__main__':
    path = '/Users/iguchiyusuke/Pictures/caffeine.jpg'
    contour = make_contour_image(path=path)
    cv2.imwrite('/Users/iguchiyusuke/Pictures/caffeine_contour_test.jpg', contour)
