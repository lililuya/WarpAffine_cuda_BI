import cv2
import numpy as np
import warp_affine

input_image = cv2.imread("/mnt/hd1/cxh/liwen/AffineTransformProject/data/0.png")
if input_image is None:
    raise FileNotFoundError("Input image not found!")

dst_height, dst_width = 1152, 768

d2i = np.array([
    2, 0.0, 0.0,  # 水平缩放 0.5，水平平移 100
    0.0, 2, 0.0    # 垂直缩放 0.5，垂直平移 50
], dtype=np.float32)

import time
for i in range(1000000):
    s = time.time()
    output_image = warp_affine.warpaffine_gpu(input_image, dst_height, dst_width, d2i)
    e =time.time()
    t = e- s
    print(f"Cost {t} s")
    cv2.imwrite(f"./out_dir/output_{i}.jpg", output_image)
print("Output image saved as output.jpg")