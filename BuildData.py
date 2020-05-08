import os.path
from os import path
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np 

#Add image to trainingset at data_path with given id value
def AddTrainingSetImage(input_img, img_id, data_path):

    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    
    df1 = pd.DataFrame()
    
    if path.exists(data_path):
        df1 = pd.read_csv(data_path)
        
    #HSV histograms
    hist_hue = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_hue = hist_hue/np.sum(hist_hue)
    hist_sat = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_sat = hist_sat/np.sum(hist_sat)
    hist_bright = cv2.calcHist([img],[2],None,[256],[0,256])
    hist_bright = hist_bright/np.sum(hist_bright)
    
    #Brightness channel is grayscale of image
    gray  = img[:,:,2]
    
    #blur
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    #apply x and y sobel
    sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5)
    sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    
    sobel_m = np.mean(sobel)/1000
    
    #Add values to data frame
    df2 = pd.DataFrame()
    
    hcol = 'h'
    for i in range(0, 256):
        col = hcol + str(i)
        df2.insert(0, col, hist_hue[i], True) 
        
    scol = 's'
    for i in range(0, 256):
        col = scol + str(i)
        df2.insert(0, col, hist_sat[i], True) 
        
    df2.insert(0, "sobel_m", sobel_m, True) 
        
    df2.insert(0, "id", img_id, True) 
    
    #Add datafram to training set
    df1 = df1.append(df2, ignore_index = True) 
    
    #Save new training set
    df1.to_csv(data_path, index=False)
    
def GetDataFrame(img_path):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #HSV histograms
    hist_hue = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_hue = hist_hue/np.sum(hist_hue)
    hist_sat = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_sat = hist_sat/np.sum(hist_sat)
    hist_bright = cv2.calcHist([img],[2],None,[256],[0,256])
    hist_bright = hist_bright/np.sum(hist_bright)
    
    #Brightness channel is grayscale of image
    gray  = img[:,:,2]
    
    #blur
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    #apply x and y sobel
    sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5)
    sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    
    sobel_m = np.mean(sobel)/1000
    
    print("Sobel m ", sobel_m)
    
    #Add values to dataframe
    df2 = pd.DataFrame()
    
    hcol = 'h'
    for i in range(0, 256):
        col = hcol + str(i)
        df2.insert(0, col, hist_hue[i], True) 
        
    scol = 's'
    for i in range(0, 256):
        col = scol + str(i)
        df2.insert(0, col, hist_sat[i], True) 
        
    df2.insert(0, "sobel_m", sobel_m, True) 
        
    df2.insert(0, "id", 0, True) 
        
    #return dataframe
    return df2

def GetDataFrameIMG(img):
    
    #.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #HSV histograms
    hist_hue = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_hue = hist_hue/np.sum(hist_hue)
    hist_sat = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_sat = hist_sat/np.sum(hist_sat)
    hist_bright = cv2.calcHist([img],[2],None,[256],[0,256])
    hist_bright = hist_bright/np.sum(hist_bright)
    
    #Brightness channel is grayscale of image
    gray  = img[:,:,2]
    
    #blur
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    #apply x and y sobel
    sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5)
    sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    
    sobel_m = np.mean(sobel)/1000
    
    print("Sobel m ", sobel_m)
    
    #Add values to dataframe
    df2 = pd.DataFrame()
    
    hcol = 'h'
    for i in range(0, 256):
        col = hcol + str(i)
        df2.insert(0, col, hist_hue[i], True) 
        
    scol = 's'
    for i in range(0, 256):
        col = scol + str(i)
        df2.insert(0, col, hist_sat[i], True) 
        
    df2.insert(0, "sobel_m", sobel_m, True) 
        
    df2.insert(0, "id", 0, True) 
    #Return dataframe
    return df2
    
def MakeTrainingSetFromFolders(directory_path, img_id, data_path):
    i = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            #print(filename,"\n")
            i = i+1
            if(np.mod(i,10)==0): print("...")
            AddTrainingSetImage(cv2.imread(directory_path + "/" + filename), img_id, data_path)
            continue
        else:
            continue
def JDMakeSet(data_path):
    MakeTrainingSetFromFolders("JD_YogaMat/",4, data_path)
    MakeTrainingSetFromFolders("JD_Table/",7, data_path)


