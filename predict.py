import copy
import os
import random


import numpy as np
from PIL import Image

from res_zp_deeplab import res_zp_Deeplabv3
# from deeplab import Deeplabv3

if __name__ == "__main__":
    
    class_colors = [[0,0,0],[0,255,0]]
    #------------------------------------------------------------------------------------------#
    #   Define the height and width of the input image, and the number of types
    #------------------------------------------------------------------------------------------#
    HEIGHT = 512
    WIDTH = 512
   
    NCLASSES = 2
    
    #---------------------------------------------#
    #   载入模型
    #---------------------------------------------#
    model = res_zp_Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,3))#,train_cl = False ,Xcep_weight_dir = None)
    
    model.load_weights("./log/last1.h5")

    
    imgs = os.listdir("./img/" )
    for jpg in imgs:
        #--------------------------------------------------#
        #   打开imgs文件夹里面的每一个图片
        #--------------------------------------------------#
        img = Image.open("./img/"+ jpg ).convert('RGB')
        
        old_img = copy.deepcopy(img)
        
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        #--------------------------------------------------#
        #   对输入进来的每一个图片进行Resize
        #   resize成[HEIGHT, WIDTH, 3]
        #--------------------------------------------------#
        img = img.resize((WIDTH,HEIGHT), Image.BICUBIC)
        img = np.array(img) / 255
        img = img.reshape(-1, HEIGHT, WIDTH, 3)

        #--------------------------------------------------#
        #   将图像输入到网络当中进行预测
        #--------------------------------------------------#
        pr = model.predict(img)[0]
        pr = pr.reshape((int(HEIGHT), int(WIDTH), NCLASSES)).argmax(axis=-1)
        pr = pr *255
        pr = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h))
        pr.save("./result/"+'test_result'+jpg)
        


