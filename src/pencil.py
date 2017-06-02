import cv2

def make_pencil_sketch_image(path):
    img = cv2.imread(path)
    gray, tmp = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.11, shade_factor=0.04)
    return gray

if __name__ == '__main__':
    path = "/Users/iguchiyusuke/Pictures/caffeine_contour_test.jpg"
    pencil_sketch = make_pencil_sketch_image(path=path)
    cv2.imwrite("/Users/iguchiyusuke/Pictures/caffeine_pencil_sketch_test.jpg", pencil_sketch)
