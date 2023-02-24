import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import time

model = LinearRegression()

class Filter:
    
    def Get_coeff(self,x,y):
        target = list()
        input = list()
        for i in range(x,x+self.gap):
            for j in range(y,y+self.gap):
                input.append([i-x,j-y])
                # print(f"input append {i-x}{j-y} -> {self.arr[i][j]}")
                target.append(self.arr[i][j][0]*0.299+self.arr[i][j][1]*0.587+self.arr[i][j][2]*0.587)
        input = np.array(input)
        target = np.array(target)
        # x = input[:,0]
        # y = input[:,1]
        # z = target
        model.fit(input,target)
        coefs = model.coef_
        intercept = model.intercept_
        # print(coefs)
        # print(intercept)
        self.pixel[int(x/self.gap)][int(y/self.gap)] = 0.299*coefs[0]+0.587*coefs[1]+0.114*intercept
        # xs = np.tile(np.arange(10), (10,1))
        # ys = np.tile(np.arange(10), (10,1)).T
        # zs = xs*coefs[0]+ys*coefs[1]+intercept
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.plot_surface(xs,ys,zs, alpha=0.5)
        # ax.scatter(x, y, z) 
        # ax.set_box_aspect((1, 1, 0.5))
        # plt.title('Axes3D Plot')
        # plt.show()       
        
    # x 是列 y 是行
    def __init__(self,arr):
        self.arr = arr
        self.gap = 4 #defind gap
        self.pixel = np.zeros((int(arr.shape[0]/self.gap),int(arr.shape[1]/self.gap)))
        for i in range(0,arr.shape[0],self.gap):
            for j in range(0,arr.shape[1],self.gap):
                self.Get_coeff(i,j)  
        cv2.imwrite('coeff_low.jpg', self.pixel*255*5)      

start_time = time.time()
color = cv2.imread('Input_hdr/output_origin.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# color = cv2.imread('Input_hdr/hotel_room_LDR.jpg')
# color = cv2.resize(color, (1024,512))
# color = color/255
RGBimage = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) #BGR to RGB
x = Filter(RGBimage)
elapsed_time = time.time() - start_time
print(f"Time spanning {elapsed_time}")
cv2.imshow('Result',x.pixel)
cv2.waitKey(0)
cv2.destroyAllWindows()