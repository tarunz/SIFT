
# coding: utf-8

# In[2]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi,floor,ceil


# In[3]:

def show_rgb(img):
    return plt.imshow(img)
def to_gray(color_img):
    return cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
def show_gray(img):
    return plt.imshow(img,cmap='gray')
def blur(img,sig):
    return cv2.GaussianBlur(img,(5,5),sig)
def difference(img1,img2):
    return img2-img1
def get_blurs(img,k):
    blurs = list()
    s = 1.6
    K = 0.714
    for i in range(k):
        s = K*s
        blurs.append(blur(img,s))
    return blurs
def get_diff_blur(blurs):
    diffs = list()
    for i in range(1,len(blurs)):
        diffs.append(difference(blurs[i],blurs[i-1]))
    return diffs
def maxomin(x,y,lay,diffs):
    p = diffs[lay][x][y]
    thresh = 3
    if p <= thresh:
        return False
    if p>diffs[lay+1][x][y]:
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if (i !=0 and j!=0 and k!=0) and diffs[lay+i][x+j][y+k] >= p-thresh:
                        return False
    else:
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if (i !=0 and j!=0 and k!=0) and diffs[lay+i][x+j][y+k] <= p+thresh:
                        return False
    return True
def get_keypoints(orig,diffs):
    res = np.array(orig)
    keypoints = list()
    for k in range(1,len(diffs)-1):
        for i in range(1,len(diffs[0])-1):
            for j in range(1,len(diffs[0][0])-1):
                if maxomin(i,j,k,diffs):
                    cv2.circle(res,(j,i), 1, (0,0,0), 1)
                    keypoints.append((i,j))
    return (res,keypoints)


# In[4]:

def getHist(img, px, py):
    histo = [0 for i in range(8)]
    global orig
    for i in range(4):
        for j in range(4):
            x = px + i
            y = py + j
            dx = orig[x][y+1]-orig[x][y-1]
            dy = orig[x+1][y]-orig[x-1][y]
            mag  = sqrt(dx*dx + dy*dy)
            if dx==0:
                theta = 90
            else:
                theta = np.arctan(dy/dx) * 360 / (2*pi)-1
            theta /= 45
            left = int(floor(theta))
            right = int(ceil(theta))
            left_val = mag * (theta-left)
            right_val = mag * (right-theta)
            histo[left] += left_val
            histo[right] += right_val
    return histo 

def getDescriptors(img, i, j):
    sift_descriptor = []
    for r in range(4):
        for c in range(4):
            x = r*4+i
            y = c*4+j
            sift_descriptor.append(getHist(img, x, y))
    return sift_descriptor


# In[6]:

orig1 = cv2.imread('scar.jpg',0)
original1 = orig1.astype(float)
orig = cv2.resize(orig1,(0,0),fx = 2,fy=2)
# cv2.imshow("ngvh",orig)
for i in [1,0.5,1.25]:
    orig = cv2.resize(orig1,(0,0),fx = i,fy=i)
    original = cv2.resize(original1,(0,0),fx = i,fy=i)
    blurs = get_blurs(original,8)
    diffs = get_diff_blur(blurs)
    (utech,keypoints) = get_keypoints(orig,diffs)
    plt.rcParams["figure.figsize"] = 10,10
    plt.axis("off")
    cv2.imwrite("Image"+str(i)+".jpg",utech)
    cv2.imshow("Image"+str(i),utech)
    descriptors = list()
    n,m = original.shape
    for x in keypoints:
        a,b = x
        if a-20>=0 and b-20>=0 and a <= n-20 and b <= m-20:
            descriptors.append(getDescriptors(original,a,b))
    print descriptors
    print
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[ ]:



