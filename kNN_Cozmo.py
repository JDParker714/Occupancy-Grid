import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
from BuildData import GetDataFrame
from BuildData import GetDataFrameIMG

def Train_KNN(data_path):
    global knn
    
    #Read datapath
    df = pd.read_csv(data_path)
    
    #Take of id column
    X = df.drop(columns=['id'])
    
    y = df['id'].values
    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10, stratify=y)
    
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 5)
    
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    
    #Print score
    print("Score: ", knn.score(X_test, y_test), "\n") 
    
    #Adaptive thresholding
    #Get mean and standard deviation values for each unique id
    adapt = {}
    adapt_amt = {}
    for i in range(X_test.shape[0]):
        #print(np.average(knn.kneighbors(X_test.iloc[[i]],3,True)[0]))
        if y_test[i] in adapt:
            adapt[y_test[i]] = adapt[y_test[i]]+np.average(knn.kneighbors(X_test.iloc[[i]],3,True)[0])
            adapt_amt[y_test[i]] = adapt_amt[y_test[i]]+1
            #print(y_test[i], " : " , adapt[y_test[i]])
        else:
            adapt[y_test[i]] = np.average(knn.kneighbors(X_test.iloc[[i]],3,True)[0])
            adapt_amt[y_test[i]] = 1
            #print(y_test[i])
        #print(np.average(knn.kneighbors(X_test.iloc[[i]],3,True)[0]))
    adapt_threshold = {}
    for key in adapt:
        adapt_threshold[key] = adapt[key]/adapt_amt[key]
        print(key, " Thresh: " , adapt_threshold[key])
        
    adapt_sd = {}
    for key in adapt:
        mean = adapt_threshold[key]
        n = adapt_amt[key]
        for i in range(X_test.shape[0]):
            if y_test[i] == key:
                mv = (np.average(knn.kneighbors(X_test.iloc[[i]],3,True)[0])-mean)**2
                if y_test[i] in adapt_sd:
                    adapt_sd[key] = adapt_sd[key] + mv
                else:
                    adapt_sd[key] = mv
        adapt_sd[key] = (adapt_sd[key]/(n-1))**0.5
        print(key, " SD: " , adapt_sd[key])
        
    return knn, adapt_threshold, adapt_sd

def Get_KNN_IsFloor(img, knn, adapt, adapt_sd):
    
    test = GetDataFrameIMG(img)
    test = test.drop(columns=['id'])
    
    #Get nearest neighbors to point
    #predict id value
    pred = knn.predict(test)[0]
    knn_test = knn.kneighbors(test,5,True)
    
    #Get average distance tho neighbors
    av_knn_dist = np.average(knn_test[0])
    print("Average Distance: ",av_knn_dist)
    
    #Use adaptive threshold
    if pred in adapt:
        threshold = adapt[pred]+(adapt_sd[pred]*3)
        print("threshold: ", threshold)
    else:
        threshold = 0.3
    
    #Get floor type from id value
    floor_type = "unknown"
    if pred==9:
        floor_type = "paper"
    if pred==7:
        floor_type = "wood table"
    if pred==4:
        floor_type = "yoga mat"
    
    #Return id value if withoin threnshold, otherwise return -1 for obstacle
    if av_knn_dist>threshold:
        print("NO - Not floor ","floor type: ", floor_type)
        return -1
    else:
        print("YES - Floor ","floor type: ", floor_type)
        return pred
    
