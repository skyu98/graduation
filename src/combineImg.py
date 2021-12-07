import numpy as np
from PIL import Image
import glob
import os

origin_dir = '../imgs/test1_input_imgs' #存放原始图片的文件夹路径
res_dir = '../imgs/test1_output_imgs' #存放检测结果图片的文件夹路径
output_dir = '../imgs/test1_combined_imgs' #存放拼接结果图片的文件夹路径

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def combineImg(imgName) :
    origin_img = Image.open(origin_dir + '/' + imgName) 
    res_img = Image.open(res_dir + '/' + imgName) 

    # print(res_img.size) # 宽，高
    assert origin_img.size == (3840, 2160)

    new_height = res_img.size[1]
    new_width = int(origin_img.size[0] * new_height / origin_img.size[1])

    origin_img = origin_img.resize((new_width, new_height))
    # print(origin_img.size)

    origin_arr = np.array(origin_img) # 转化为ndarray对象
    res_arr = np.array(res_img) 

    combined_arr = np.concatenate((origin_arr, res_arr), axis = 1) # 横向拼接
    # print(combined.size)

    combined_img = Image.fromarray(combined_arr)
    combined_img.save(output_dir + '/' + imgName)
    
imgs = glob.glob(os.path.join(origin_dir, '*.jpg'))
imgs += (glob.glob(os.path.join(origin_dir, '*.JPG')))
imgs.sort()

for img in imgs :
    imgName = img[len(origin_dir) + 1:]
    print(imgName)
    combineImg(imgName)