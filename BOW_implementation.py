# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:39:32 2019

@author: Jama Hussein Mohamud
"""
'''
data/
    train/
        neg/
            0579.PNG
            0581.PNG
            ...
        pos/
            0576.PNG
            0576.PNG
            ...
    Test/
        neg/
            0021.PNG
            0027.PNG
            ...
        pos/
            0001.PNG
            0002.PNG
'''

#importing 

import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.neighbors import KNeighborsClassifier

#%%

# Get the training classes names and store them in a list
train_path = 'data'
training_names = os.listdir(train_path)

def opening(path):
    """
    The function opening returns all the names of the files in 
    the directory path.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]
#%%
    
# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = opening(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

#%%
# plot hitogram od class distribution
labels = np.array(image_classes)
print(np.unique(labels, return_counts=True))

plt.hist(labels)
plt.xlabel('labels')
plt.ylabel('Frequency')
plt.title('Class distribution')

#%%
# Creating feature extraction and keypoint detector objects
surf = cv2.xfeatures2d.SURF_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = surf.detect(im, None)
    kpts, des = surf.compute(im, kpts)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

#%%

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
train_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        train_features[i][w] += 1

# Scaling the words
stdSlr = StandardScaler().fit(train_features)
train_features = stdSlr.transform(train_features)

#%%
# Training the classifier, you must choose either KNN or SVM by uncommenting. 

param_grid = {'C': [100, 1000, 10000, 100000.0, 10000000, 10600506], 'gamma' : [0.0001,0.001, 0.01, 0.1, 1,10, 100], 'kernel': ['linear','rbf']}
clf = GridSearchCV(svm.SVC(), param_grid, cv= 5)
clf.fit(train_features, np.array(image_classes))
print('The best parameters found by gridSearch: ', clf.best_params_)

#param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
#clf = grid_search.GridSearchCV(KNeighborsClassifier(), param_grid, cv= 10)
#clf.fit(train_features, np.array(image_classes))
#print('The best parameters found by gridSearch: ', clf.best_params_)

result = clf.score(train_features, np.array(image_classes))
print('Training Accuracy: ',result)

# Save the the classifier
# save classifier, class names, scaler, number of clusters and vocabulary
joblib.dump((clf, training_names, stdSlr, k, voc), open("trained_data_svm.sav",'wb'))   

#%%
# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("trained_data_svm.sav")

# Getting the path of the testing image(s) and storing them in a list
image_paths_test = []
image_classes_test = []
class_id = 0

test_path = 'Test'
testing_names = os.listdir(test_path)

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = opening(dir)
    image_classes_test+=[class_id]*len(class_path)
    image_paths_test+=class_path
    class_id+=1
    
# Create SURF feature extraction and keypoint detector objects
surf = cv2.xfeatures2d.SURF_create()

# all the descriptors will be stored in this list
des_list_test = []

for image_path in image_paths_test:
    im = cv2.imread(image_path)
    kpts = surf.detect(im, None)
    kpts, des = surf.compute(im, kpts)
    des_list_test.append((image_path, des))   
    
# Stack all the descriptors vertically
descriptors = des_list_test[0][1]
for image_path, descriptor in des_list_test[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

test_features = np.zeros((len(image_paths_test), k), "float32")
for i in range(len(image_paths_test)):
    words, distance = vq(des_list_test[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Scale the features
test_features = stdSlr.transform(test_features)

result_test = clf.score(test_features, np.array(image_classes_test))
print('Test Accuracy: ',result_test)

#%% Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#%%

# Compute confusion matrix
predictions = clf.predict(test_features)
cnf_matrix = confusion_matrix(image_classes_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=classes_names,
                      title='Confusion matrix')


#%%

# Performing predictions
predic =  [classes_names[i] for i in clf.predict(test_features)]

# Visualize the results
for image_path, prediction in zip(image_paths_test, predic):
    image = cv2.imread(image_path)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    pt = (0, 3 * image.shape[0] // 4)
    cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 3, cv2.LINE_AA)
    cv2.imshow("Image", image)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()










