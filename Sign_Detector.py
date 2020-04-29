import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCHES = 15
#
sift = cv2.xfeatures2d.SURF_create()
#sift = cv2.ORB_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
flann = cv2.FlannBasedMatcher(index_params,{})      

trainImg1 = cv2.imread('lozange2.png',0) # trainImage 1
trainImg1 = cv2.resize(trainImg1, (400, 400)) 

trainImg2 = cv2.imread('lozange.png',0) # trainImage 2
trainImg2 = cv2.resize(trainImg2, (400, 400)) 

trainImg3 = cv2.imread('110.png',0) # trainImage 3
trainImg3 = cv2.resize(trainImg3, (400, 400)) 

kpImage1, desImage1 = sift.detectAndCompute(trainImg1, None)
kpImage2, desImage2 = sift.detectAndCompute(trainImg2, None)
kpImage3, desImage3 = sift.detectAndCompute(trainImg3, None)

img1 = cv2.drawKeypoints(trainImg1, kpImage1, trainImg1, color=(0,0,255), flags=2)
img2 = cv2.drawKeypoints(trainImg2, kpImage2, trainImg2, color=(0,0,255), flags=2)
img3 = cv2.drawKeypoints(trainImg3, kpImage3, trainImg3, color=(0,0,255), flags=2)

cv2.imshow("frame1",img1)
cv2.imshow("frame2",img2)
cv2.imshow("frame3",img3)

###################################################################################################

Image = cv2.imread('Road 1.jpg',0) # trainImage 1
PlanImage = cv2.resize(Image, (640, 480))
PlanImageBW = cv2.cvtColor(PlanImage, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(PlanImageBW,cv2.CV_64F,1,0,ksize=5)

kpPlan, desPlan = sift.detectAndCompute(PlanImage, None)

matches1 = flann.knnMatch(desPlan,desImage1,k=2)
matches2 = flann.knnMatch(desPlan,desImage2,k=2)
matches3 = flann.knnMatch(desPlan,desImage3,k=2)

good1 = []
good2 = []
good3 = []

for i,j in matches1:
    if i.distance < 0.75*j.distance:
        good1.append(i)
       
for k,l in matches2:
    if k.distance < 0.75*l.distance:
        good2.append(k)
        
for m,n in matches3:
    if m.distance < 0.75*n.distance:
        good3.append(m)
        


if(len(good1)>=MIN_MATCHES):
        
    src_pts1=[]
    dst_pts1=[]
    
# Assuming matches stores the matches found and 
# Returned by bf.match(des_model, des_frame)
# Differenciate between source points and destination points
    src_pts1 = np.float32([kpPlan[i.queryIdx].pt for i in good1]).reshape(-1, 1, 2)
    dst_pts1 = np.float32([kpImage1[i.trainIdx].pt for i in good1]).reshape(-1, 1, 2)
    
    M1, mask1 = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
    
    h, w = PlanImage.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    dst1 = cv2.perspectiveTransform(pts, M1)
    cv2.polylines(PlanImage, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA) 

if(len(good2)>=MIN_MATCHES):
    
    src_pts2=[]
    dst_pts2=[]
    
    src_pts2 = np.float32([kpPlan[k.queryIdx].pt for k in good2]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kpImage2[k.trainIdx].pt for k in good2]).reshape(-1, 1, 2)
    
    M2, mask2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    
    h, w = PlanImage.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    dst2 = cv2.perspectiveTransform(pts, M2)
    cv2.polylines(Image, [np.int32(dst2)], True, 255, 3, cv2.LINE_AA) 
    
#if(len(good3)>=MIN_MATCHES):
#    
#    src_pts3=[]
#    dst_pts3=[]
#     
#    src_pts3 = np.float32([kpPlan[m.queryIdx].pt for m in good3]).reshape(-1, 1, 2)
#    dst_pts3 = np.float32([kpImage3[m.trainIdx].pt for m in good3]).reshape(-1, 1, 2)
#    
#    M3, mask3 = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
#      
#    dst3 = cv2.perspectiveTransform(pts, M3)
    
else:
    print ("Not enough matches are found - %d/%d \n %d/%d \n %d/%d" % (len(good1),MIN_MATCHES,len(good2),MIN_MATCHES,len(good3),MIN_MATCHES))
    matchesMask = None
    
cv2.imshow('result',Image)

cv2.waitKey(0)
cv2.destroyAllWindows()

