import glob
import os

dir = '../imgs/raw_imgs'#存放图片的文件夹路径
imgs = glob.glob(os.path.join(dir, '*.jpg'))
imgs += (glob.glob(os.path.join(dir, '*.JPG')))
imgs.sort()

for img in imgs :
    img = img[len(dir) + 1:]
    command = '../bin/main ' + img
    os.system(command)
