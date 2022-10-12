#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask
import werkzeug
import time


# # python idle path :- C:\Users\Lenovo P500\AppData\Local\Programs\Python\Python35

# In[2]:


import numpy as np
import urllib
import urllib.request
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import scipy.ndimage


# In[3]:


from flask import Flask,redirect,url_for,render_template
from flask import send_file
import flask
from flask import request
import requests
import base64
import os
import webbrowser


# In[4]:


from pathlib import Path
from flask import send_from_directory
# from flask import send
import time


# In[5]:


app = flask.Flask(__name__)


# In[7]:


from scipy.interpolate import UnivariateSpline
def LookupTable(x, y):
    
    spline = UnivariateSpline(x, y)
    return spline(range(256))


# In[8]:


#to display the connection status
@app.route('/connect', methods=['GET', 'POST'])
def handle_call():
    return "Successfully Connected ghhfgcfg"


# In[ ]:


@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    
   
    imagefile = flask.request.files['image']#for getting image byte array
    filter_type = flask.request.values.get('imgtype') #for getting string value
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    imagefile.save(filename)
    imagefile = cv2.imread('androidFlask.jpg')
        
        
    if filter_type == str(1):
        k_size=201
        grey_img=cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY)

        # Invert Image
        invert_img=cv2.bitwise_not(grey_img)
        #invert_img=255-grey_img

        # Blur image
        blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

        # Invert Blurred Image
        invblur_img=cv2.bitwise_not(blur_img)
        #invblur_img=255-blur_img

        sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)
        timestr = time.strftime("%Y%m%d-%H%M%S")
    #     Segments = url_sketch.rpartition('/')
    #     Segments = Segments[2]
        filename1 = "static/"+'sketch_'+timestr+'.jpg'
        cv2.imwrite(filename1, sketch_img)
        url_path = url_for('static', filename='sketch_'+timestr+'.jpg')
#         full_path = "192.168.221.248:5000" + url_path
#         return full_path
        
    elif filter_type == str(2): 
        sk_gray, sketch_color_img = cv2.pencilSketch(imagefile, sigma_s=60, sigma_r=0.05, shade_factor=0.03)
         # Sketch Image
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename2 = "static/"+'sketchcolor_'+timestr+'.jpg'
        cv2.imwrite(filename2, sketch_color_img)
        url_path = url_for('static', filename='sketchcolor_'+timestr+'.jpg')
#         full_path = "192.168.221.248:5000" + url_path
#         return full_path
    
    elif filter_type == str(3): 
        oilpaint_img = cv2.xphoto.oilPainting(imagefile, 15, 1)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename3 = "static/"+'oil_'+timestr+'.jpg'
        cv2.imwrite(filename3, oilpaint_img)
        url_path = url_for('static', filename='oil_'+timestr+'.jpg')
     
    elif filter_type == str(4): 
        water_img = cv2.stylization(imagefile, sigma_s=500, sigma_r=0.3)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename4 = "static/"+'water_'+timestr+'.jpg'
        cv2.imwrite(filename4, water_img)
        url_path = url_for('static', filename='water_'+timestr+'.jpg')
        
        
    elif filter_type == str(5): 
        grayed = cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(grayed)
        blurred = cv2.GaussianBlur(inverted, (51, 51), sigmaX=0, sigmaY=0)# x and y are odd
        artistic_img = cv2.divide(grayed, 255 - blurred, scale=256)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename5 = "static/"+'artistic_'+timestr+'.jpg'
        cv2.imwrite(filename5, artistic_img)
        url_path = url_for('static', filename='artistic_'+timestr+'.jpg')
        
         
    elif filter_type == str(6): 
        hdr_img = cv2.detailEnhance(imagefile, sigma_s=12, sigma_r=0.15)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename6 = "static/"+'hdr_'+timestr+'.jpg'
        cv2.imwrite(filename6, hdr_img)
        url_path = url_for('static', filename='hdr_'+timestr+'.jpg')
        
         
    elif filter_type == str(7): 
        grayImage = cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 95, 255, cv2.THRESH_BINARY)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename7 = "static/"+'blacknwhite_'+timestr+'.jpg'
        cv2.imwrite(filename7, blackAndWhiteImage)
        url_path = url_for('static', filename='blacknwhite_'+timestr+'.jpg')
        
         
    elif filter_type == str(8): 
        gray = cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY)
        normalized_gray = np.array(gray, np.float32)/255
        #solid color
        sepia = np.ones(imagefile.shape)
        sepia[:,:,0] *= 153 #B
        sepia[:,:,1] *= 204 #G
        sepia[:,:,2] *= 255 #R
        #hadamard
        sepia[:,:,0] *= normalized_gray #B
        sepia[:,:,1] *= normalized_gray #G
        sepia[:,:,2] *= normalized_gray #R
        sepia = np.array(sepia, np.uint8)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename8 = "static/"+'sepia_'+timestr+'.jpg'
        cv2.imwrite(filename8, sepia)
        url_path = url_for('static', filename='sepia_'+timestr+'.jpg')
        
         
    elif filter_type == str(9): 
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel = cv2.split(imagefile)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        winter= cv2.merge((blue_channel, green_channel, red_channel))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename9 = "static/"+'winter_'+timestr+'.jpg'
        cv2.imwrite(filename9, winter)
        url_path = url_for('static', filename='winter_'+timestr+'.jpg')
        
         
    elif filter_type == str(10): 
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel,red_channel  = cv2.split(imagefile)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        summer= cv2.merge((blue_channel, green_channel, red_channel ))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename10 = "static/"+'summer_'+timestr+'.jpg'
        cv2.imwrite(filename10, summer)
        url_path = url_for('static', filename='summer_'+timestr+'.jpg')
        
         
    elif filter_type == str(11): 
        invert=cv2.bitwise_not(imagefile)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename11 = "static/"+'invert_'+timestr+'.jpg'
        cv2.imwrite(filename11, invert)
        url_path = url_for('static', filename='invert_'+timestr+'.jpg')
        
        
    elif filter_type == str(12): 
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(imagefile, -1, kernel)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename12 = "static/"+'sharpen_'+timestr+'.jpg'
        cv2.imwrite(filename12, img_sharpen)
        url_path = url_for('static', filename='sharpen_'+timestr+'.jpg')
        
    elif filter_type == str(13): 
        total_color = 8
        k=total_color
        # Transform the image
        data = np.float32(imagefile).reshape((-1, 3))
        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 2)
        # Implementing K-Means
        ret, label, center = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        cartoon = result.reshape(imagefile.shape)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename13 = "static/"+'cartoon_'+timestr+'.jpg'
        cv2.imwrite(filename13, cartoon)
        url_path = url_for('static', filename='cartoon_'+timestr+'.jpg')
        
    elif filter_type == str(14): 
        gray = cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY) 
        clahe = cv2.createCLAHE(clipLimit =0) 
        clahe_img = clahe.apply(gray)
        denoised =cv2.fastNlMeansDenoising(clahe_img,None,1,1,1) 
        _, oreo = cv2.threshold(denoised, 105, 255, cv2.THRESH_BINARY)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename14 = "static/"+'oreo_'+timestr+'.jpg'
        cv2.imwrite(filename14, oreo)
        url_path = url_for('static', filename='oreo_'+timestr+'.jpg')
        
    elif filter_type == str(15): 
        gray = cv2.cvtColor(imagefile, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE() 
        clahe_img = clahe.apply(gray)
        mercury =cv2.fastNlMeansDenoising(clahe_img,None,40,7,21)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename15 = "static/"+'mercury_'+timestr+'.jpg'
        cv2.imwrite(filename15, mercury)
        url_path = url_for('static', filename='mercury_'+timestr+'.jpg')    
         
    elif filter_type == str(16): 
        rgb=cv2.cvtColor(imagefile, cv2.COLOR_BGR2RGB)
        alchemy=cv2.fastNlMeansDenoisingColored(rgb,None,20,10,7,18)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename16 = "static/"+'alchemy_'+timestr+'.jpg'
        cv2.imwrite(filename16, alchemy)
        url_path = url_for('static', filename='alchemy_'+timestr+'.jpg')
        
    elif filter_type == str(17):
        hsv=cv2.cvtColor(imagefile, cv2.COLOR_BGR2HSV)
        _,s,v=cv2.split(hsv)
        wacko= cv2.merge([s,v,v])
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename17 = "static/"+'wacko_'+timestr+'.jpg'
        cv2.imwrite(filename17, wacko)
        url_path = url_for('static', filename='wacko_'+timestr+'.jpg')
        
    elif filter_type == str(18): 
        kernel=np.array([[0.272, 0.1, 0.131],[0.349, 0.1, 0.168],[0.393, 0.1, 0.189]])
        unstable=cv2.filter2D(imagefile, -1, kernel)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename18 = "static/"+'unstable_'+timestr+'.jpg'
        cv2.imwrite(filename18, unstable)
        url_path = url_for('static', filename='unstable_'+timestr+'.jpg') 
    
    elif filter_type == str(19): 
        kernel=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        ore=cv2.filter2D(imagefile, -1, kernel)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename19 = "static/"+'ore_'+timestr+'.jpg'
        cv2.imwrite(filename19, ore)
        url_path = url_for('static', filename='ore_'+timestr+'.jpg') 
        
    elif filter_type == str(20): 
        clone=imagefile.copy()
        denoised=cv2.fastNlMeansDenoisingColored(clone, None, 5, 5, 7, 15)
        snicko=cv2.Canny(denoised,50,150)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename20 = "static/"+'snicko_'+timestr+'.jpg'
        cv2.imwrite(filename20, snicko)
        url_path = url_for('static', filename='snicko_'+timestr+'.jpg')
        
    elif filter_type == str(21): 
        denoised_color=cv2.fastNlMeansDenoisingColored(imagefile, None, 10, 10, 7, 15)
        gray=cv2.cvtColor(denoised_color,cv2.COLOR_BGR2GRAY)
        adap=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,10)
        contours,hierarchy=cv2.findContours(adap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour=denoised_color.copy()
        color=(255,255,255)
        for c in contours:
            cv2.drawContours(contour,[c],-1,color,1)    
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename21 = "static/"+'contour_'+timestr+'.jpg'
        cv2.imwrite(filename21, contour)
        url_path = url_for('static', filename='contour_'+timestr+'.jpg')
        
        
#     elif filter_type == str(22): 
#         template = cv2.imread("C:/Users/Lenovo P500/AppData/Local/Programs/Python/Python35/Scripts/images/flag.jpg")
#         row1,cols1,_= imagefile.shape

#         row2,cols2,_ = template.shape
#         x=cols1/cols2
#         y=row1/row2
#         res = cv2.resize(template, (0, 0), fx = x, fy = y) 
#         indus = cv2.addWeighted(imagefile,0.5,res,0.75,0)
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         filename22 = "static/"+'indus_'+timestr+'.jpg'
#         cv2.imwrite(filename22, indus)
#         url_path = url_for('static', filename='indus_'+timestr+'.jpg')
        
#     elif filter_type == str(23): 
#         template = cv2.imread("C:/Users/Lenovo P500/AppData/Local/Programs/Python/Python35/Scripts/images/temp.png")
#         row1,cols1,_= imagefile.shape
#         row2,cols2,_ = template.shape
#         x=cols1/cols2
#         y=row1/row2
#         res = cv2.resize(template, (0, 0), fx = x, fy = y) 
#         spectra = cv2.addWeighted(imagefile,0.5,res,0.75,0)
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         filename23 = "static/"+'spectra_'+timestr+'.jpg'
#         cv2.imwrite(filename23, spectra)
#         url_path = url_for('static', filename='spectra_'+timestr+'.jpg')
        
#     elif filter_type == str(24): 
#         template = cv2.imread("C:/Users/Lenovo P500/AppData/Local/Programs/Python/Python35/Scripts/images/dots1.jpg")
#         row1,cols1,_= imagefile.shape
#         row2,cols2,_ = template.shape
#         x=cols1/cols2
#         y=row1/row2
#         res = cv2.resize(template, (0, 0), fx = x, fy = y) 
#         molecule = cv2.addWeighted(imagefile,1,res,0.5,0)
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         filename24 = "static/"+'molecule_'+timestr+'.jpg'
#         cv2.imwrite(filename24, molecule)
#         url_path = url_for('static', filename='molecule_'+timestr+'.jpg')
        
#     elif filter_type == str(25): 
#         template = cv2.imread("C:/Users/Lenovo P500/AppData/Local/Programs/Python/Python35/Scripts/images/water.jpeg")
#         row1,cols1,_= imagefile.shape
#         row2,cols2,_ = template.shape
#         x=cols1/cols2
#         y=row1/row2
#         res = cv2.resize(template, (0, 0), fx = x, fy = y) 
#         lynn = cv2.addWeighted(imagefile,1,res,0.5,0)
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         filename25 = "static/"+'lynn_'+timestr+'.jpg'
#         cv2.imwrite(filename25, lynn)
#         url_path = url_for('static', filename='lynn_'+timestr+'.jpg')
        
    elif filter_type == str(22): 
        sharp_img = cv2.detailEnhance(imagefile,sigma_s=150,sigma_r=0.8)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename22 = "static/"+'sharp_'+timestr+'.jpg'
        cv2.imwrite(filename22, sharp_img)
        url_path = url_for('static', filename='sharp_'+timestr+'.jpg')
        
    elif filter_type == str(23): 
        inverted=255-imagefile
        blered=cv2.GaussianBlur(inverted,(451,451),0)
        inverted_blured=255-blered
        sketch=imagefile/inverted_blured
        paint_sketch=sketch*255
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename23 = "static/"+'psketch_'+timestr+'.jpg'
        cv2.imwrite(filename23, paint_sketch)
        url_path = url_for('static', filename='psketch_'+timestr+'.jpg')
        
    elif filter_type == str(24): 
        img = cv2.cvtColor(imagefile,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename24 = "static/"+'gray_'+timestr+'.jpg'
        cv2.imwrite(filename24, gray)
        url_path = url_for('static', filename='gray_'+timestr+'.jpg')
        
    elif filter_type == str(25): 
        Y= 0.299*196 + 0.587*160 + 0.114*30
        def grayscale(rgb): 
        #     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        gray_img = grayscale(imagefile)
        inverted_img = 255-gray_img
        blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=501)
        result=blur_img*150/(255-gray_img)  
        result[result>255]=255 
        result[gray_img==255]=255 
        shadow = result.astype('uint8')
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename25 = "static/"+'shadow_'+timestr+'.jpg'
        cv2.imwrite(filename25, shadow)
        url_path = url_for('static', filename='shadow_'+timestr+'.jpg')
        
    try:
        full_path = "192.168.38.68:5000" + url_path
        return full_path
    except FileNotFoundError:
        abort(404)


# In[10]:


app.run(host='0.0.0.0', port=5000, debug=True)


# In[14]:


imagefile = cv2.imread("DSC_2561.jpg")
image_resized = cv2.resize(imagefile, None, fx=0.5, fy=0.5)
image_cleared = cv2.medianBlur(image_resized, 3)
image_cleared = cv2.medianBlur(image_cleared, 3)
image_cleared = cv2.medianBlur(image_cleared, 3)

image_cleared = cv2.edgePreservingFilter(image_cleared, sigma_s=25)
image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)

for i in range(2):
    image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)

for i in range(3):
    image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)
    
gaussian_mask= cv2.GaussianBlur(image_filtered, (7,7), 2)
image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)
timestr = time.strftime("%Y%m%d-%H%M%S")
filename25 = 'image_sharp_'+timestr+'.jpg'
cv2.imwrite(filename25, image_sharp)


# In[ ]:





# In[ ]:





# In[ ]:


# imagefile = cv2.imread("DSC_2561.jpg")
# inverted=255-imagefile
# blered=cv2.GaussianBlur(inverted,(451,451),0) 
# # 201
# inverted_blured=255-blered
# sketch=imagefile/inverted_blured
# paint_sketch=sketch*255
# timestr = time.strftime("%Y%m%d-%H%M%S")
# filename27 = 'psketch_'+timestr+'.jpg'
# cv2.imwrite(filename27, paint_sketch)


# In[ ]:


# template = cv2.imread("C:/Users/Lenovo P500/AppData/Local/Programs/Python/Python35/Scripts/images/flag.jpg")
# imagefile = cv2.imread("Originalmg.jpg")
# inverted=255-imagefile
# blered=cv2.GaussianBlur(inverted,(21,21),0)
# inverted_blured=255-blered
# sketch=imagefile/inverted_blured
# sketch=sketch*255
# cv2.imwrite("sketch.jpg", sketch)

