import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

Img = cv2.imread('road 1.jpg') # trainImage 1
Img_orig = Img.copy()

Img = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(Img,(5,5),cv2.BORDER_DEFAULT)

edges = cv2.Canny(img,50,110,apertureSize = 3)

#cv2.imshow('GaussianBlur',edges)


#cv2.imshow('edges',edges)
#
#ret,thresh = cv2.threshold(Img,127,255,1)
# 
#contours,h = cv2.findContours(thresh,1,2)
# 

        


        
#    elif len(approx) == 9:
#        print ("half-circle")
#        cv2.drawContours(Img,[cnt],0,(255,255,0),-1)
#    elif len(approx) > 15:
#        print ("circle")
#        cv2.drawContours(Img,[cnt],0,(0,255,255),-1)

#cv2.imshow("Lines", Img_orig)

#sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
#abs_sobel64f = np.absolute(sobelx64f)
#sobel_8u = np.uint8(abs_sobel64f)

#median = cv2.medianBlur(sobel_8u,1)

#cv2.imshow('test',median)
#        
#cv2.drawContours(Img_orig, [approx], -1, (0, 255, 0), 3)
#
#Circle Detection

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.9,120)
circles = np.uint16(np.around(circles))

count = 0
[h, w, d] = Img_orig.shape
for i in circles[0,:]:
    
    # draw the outer circle
    cv2.circle(Img_orig,(i[0],i[1]),i[2],0)
    
    if not (i[0] > w or i[1] > h):

        x_circle = i[0] - i[2]
        y_circle = i[1] - i[2]
    
        rect_width = i[0] + i[2]
        rect_height = i[1] + i[2]
    
        #w_flag = False
        #h_flag = False
        if rect_width >= w:
            rect_width = w
            #w_flag = True
        if rect_height >= h:
            rect_height = h
            #h_flag = True
    
        cv2.rectangle(Img_orig, (x_circle-15, y_circle-15), (rect_width+15, rect_height+15), (0, 255, 0), 0)
        
        ROI_circle = Img_orig[y_circle-15:rect_height+15, x_circle-15:rect_width+15, :]

        ROI_circle = cv2.resize(ROI_circle, (2*i[2], 2*i[2]))
        
        
        ROI_circle = cv2.resize(ROI_circle,(32,32))
        cv2.imshow(f'ROI_circle{count}',ROI_circle)
        cv2.imwrite(f'ROI_circle{count}.png',ROI_circle)
    # draw the center of the circle
    cv2.circle(Img_orig,(i[0],i[1]),2,(0,255,0),5)
    count+=1
    cv2.putText(Img_orig,"Circle"+ str(count),(i[0]+80,i[1]+10), cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,0),2)

print(f'I have found {str(count)} circle(s) in the image')


contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

count_square = 0
count_triangle = 0
# loop over our contours
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
    
    
    
    print (len(approx))
#    if len(approx)==5:
#        print ("pentagon")
#        cv2.drawContours(Img,[cnt],0,255,-1)
    
    if len(approx)==3:
        
        count_triangle += 1
        
        cv2.drawContours(Img_orig,[cnt],-1,0)
        x, y, w_triangle, h_triangle = cv2.boundingRect(cnt)
        cv2.rectangle(Img_orig, (x-10, y-10), (x+w_triangle+10, y+h_triangle+10), (0, 255, 0), 0)        
        ROI_triangle = Img_orig[y-10:y+h_triangle+10,x-10:x+w_triangle+10]
        
        
        ROI_triangle = cv2.resize(ROI_triangle,(32,32))
        cv2.imshow(f'ROI_triangle{count_triangle}',ROI_triangle)
        cv2.imwrite(f'ROI_triangle{count_triangle}.png',ROI_triangle)
        
    if len(approx)==4:
        
        count_square += 1
       
        cv2.drawContours(Img_orig,[cnt],-1,0)        
        x, y, w_square, h_square = cv2.boundingRect(cnt)
        cv2.rectangle(Img_orig, (x-10, y-10), (x+w_square+10, y+h_square+10), (0, 255, 0), 0)        
        ROI_square = Img_orig[y-10:y+h_square+10,x-10:x+w_square+10]
        
        
        ROI_square = cv2.resize(ROI_square,(32,32))
        cv2.imshow(f'ROI_square{count_square}',ROI_square)
        cv2.imwrite(f'ROI_square{count_square}.png',ROI_square)
#        
#    if len(approx)==7:
#        print ("circled-square")
#        cv2.drawContours(Img_orig,[cnt],-1,(0,0,255),3)
        


cv2.imshow('detected circles',Img_orig)


cv2.waitKey(0)
cv2.destroyAllWindows()