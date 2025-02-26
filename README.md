# 使用cuda核函数实现warpAffine加速
## 1. 任务目标
- 使用在cuda核函数上利用双线性差值算法实现warpaffine
- 与cv2.warpaffine(img, dst, cv2.INTERLINE)
## 2. 使用方法
- 在linux上编译opencv
- 使用python安装模块作为extention
- 然后include路径到当前环境变量中即可使用，或者配置在全局的环境中

### 2.1 linux上编译opencv
```bash
# 下载opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.11.0.zip
unzip opencv.zip

# 下载contrib
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip
unzip opencv_contrib.zip

# 注意contrib的路径位置，确保没报错，ARCH_BIN 40系是8.9
mkdir build/
cd build/
# 生成cmake，进行configuration
cmake 
    -DCMAKE_BUILD_TYPE=RELEASE     \
    -DCMAKE_INSTALL_PREFIX=/usr/local     \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.11.0/modules ..     \
    -DWITH_CUDA=1    \
    -DCUDA_ARCH_BIN=8.9  \
    -DENABLE_FAST_MATH=1   \
    -DCUDA_FAST_MATH=1     \
    -DWITH_CUBLAS=1     \
    -DOPENCV_GENERATE_PKGCONFIG=1  \
    ..

# 编译, 根据机器的核心数
sudo make -j$(nproc)

#  安装
sudo make install 

# 安装完成后，查看版本和库
sudo vim /etc/profile
export PKG_CONFIG_PATH = /usr/local/lib/pkgconfig: $PKG_CONFIG_PATH
source /etc/profile  
pkg-config --modversion opencv4
pkg-config --libs opencv4

# 配置全局环境变量
sudo touch /etc/profile.d/pkgconfig.sh
echo '/usr/local/lib' > /etc/profile.d/pkgconfig.sh
sudo ldconfig

# 运行官方的case example
cd ~/opencv/samples/cpp/example_cmake
cmake .
make
./opencv_example
```

### 2.2 编译python包
```bash
python setup.py build_ext --inplace
# 编译完成之后会生成一个build以及在当前目录生成一个so文件
```

### 2.3 导入动态链接库
导入当前环境的torch/lib和声明一下LD_PATH
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aiart/.conda/envs/image/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### 2.3 python脚本测试
```python
import cv2
import numpy as np
import warp_affine  # 编译后的模块

# 读取输入图像
input_image = cv2.imread("/mnt/hd1/cxh/liwen/AffineTransform/cheer_first_frame.jpg")
if input_image is None:
    raise FileNotFoundError("Input image not found!")

# 定义目标图像的尺寸
dst_height, dst_width = 640, 640

# d2i = np.array([
#     1, 1.1, 0.0,  # 水平缩放 0.5，水平平移 100
#     1.1, 1, 0.0    # 垂直缩放 0.5，垂直平移 50
# ], dtype=np.float32)

angle = 45  # 旋转角度（度）
theta = np.radians(angle)  # 转换为弧度

# 创建旋转矩阵
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# 仿射变换矩阵（旋转45度）
rotation_matrix = np.array([
    [cos_theta, -sin_theta, 0],  # 第一行
    [sin_theta, cos_theta, 0]     # 第二行
], dtype=np.float32)


# 调用 GPU 仿射变换函数
import time
for i in range(1000000):
    s = time.time()
    output_image = warp_affine.warpaffine_gpu(input_image, dst_height, dst_width, rotation_matrix)
    e =time.time()
    t = e- s
    print(f"Cost {t} s")
    cv2.imwrite(f"./out_dir/output_{i}.jpg", output_image)
# 保存结果

print("Output image saved as output.jpg")
```

## 3. 参考
- [cuda核函数实现](https://github.com/Rex-LK/tensorrt_learning/tree/main) 
- [双线性差值算法](https://blog.csdn.net/weixin_42108183/article/details/124199939)
- [Opencv编译](https://blog.csdn.net/weixin_65269400/article/details/140178341)
