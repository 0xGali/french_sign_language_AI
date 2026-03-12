#!/usr/bin/env python
# coding: utf-8

## Simulating an defective security cam effect on a recorded video ##
import cv2
import os
import numpy as np
import random


directory = os.getcwd() # getting the current directory
name_video = 'videos/coude.webm' # entering the name of the file to be read

## Step 1
# Creating a directory for the video frames named original_frames

try:
    if not os.path.exists('original_frames'):
        os.makedirs('original_frames')
except OSError:
    print("It was not possible to create the 'original_frames' directory")
    
#creating a video instance
cap = cv2.VideoCapture(directory + "\\" + str(name_video))

# checking some video properties
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height)
fps    = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# Initializing an auxiliary variable to extract each frame from the video
current_frame = 0

# Transforming each frame of the video into a .jpg image

while (True):
    ret, frame = cap.read()
    
    if ret == True:
        name = './original_frames/frame' + str(current_frame) + '.jpg'
        print("Creating... " + name)
        
        cv2.imwrite(name,frame) # saving the image at the frame "current_frame"
        
        current_frame += 1 # next frame
    else:
        break
        
cap.release() # releasing the video
cv2.destroyAllWindows() # destroying the windows
print("Done!")


## Step 2
# Applying a filter to each frame

# Creating a new directory for frames with filter
try:
    if not os.path.exists("directory_filter"):
        os.makedirs("directory_filter")
except OSError:
    print("It was not possible to create the 'directory_filter' directory")

# Make a FOR loop to apply the filter in each of the frames    
for count in range(len(os.listdir(directory  + "\\original_frames"))):
    filename = directory + "\\original_frames" + "\\frame" + str(count) + '.jpg'
    img = cv2.imread(filename) # opening each frame, one at a time
    
    # Applying the filter
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9,9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    carton = cv2.bitwise_and(color, color, mask = edges)
    
    # saving the frame with the filter
    cv2.imwrite(directory + "\\directory_filter" + '\\image' + str(count) + '.jpg', carton)
    print("Creating... image" + str(count) + ".jpg" )
    
cv2.destroyAllWindows()    
print("Done!")    

# Generating the video with the filter

img_array = []

for count in range(len(os.listdir(directory + "\\directory_filter"))):
    filename = directory + "\\directory_filter" + "\\image" + str(count) + ".jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter("filter.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)    

for i in range(len(img_array)):
    out.write(img_array[i])
out.release() 
print("done!")


## Step 3
# Creating security camera effect (Part one of two)
# Creating a new directory for frames with cam security like landscape (part1)
try:
    if not os.path.exists("security_cam1"):
        os.makedirs("security_cam1")
except OSError:
    print("It was not possible to create the 'security_cam' directory")
    
# Transforming the color frames into black and white frames
# Make a FOR loop to apply gray_scale in all frames    
for count in range(len(os.listdir(directory + "\\directory_filter"))):
   
    filename = directory + "\\directory_filter" + '\\image' + str(count) + '.jpg'
    img = cv2.imread(filename) 
    
    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # saving the frame with the gray scale
    cv2.imwrite(directory + "\\security_cam1" + '\\image' + str(count) + '.jpg', gray)
    print("Creating... image" + str(count) + ".jpg" )
    
cv2.destroyAllWindows() 
print("Done!")

# generating the video with the gray_scale effect

img_array = []

for count in range(len(os.listdir(directory + "\\security_cam1"))):
    filename = directory + "\\security_cam1" + "\\image" + str(count) + ".jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter("gray_scale.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)    

for i in range(len(img_array)):
    out.write(img_array[i])
out.release() 
print("done!")



## Creating security camera effect (Part two of two)
# adding security cam effect
try:
    if not os.path.exists("security_cam2"):
        os.makedirs("security_cam2")
except OSError:
    print("It was not possible to create the 'security_cam2' directory")
    
# Creating some constants
i = 0 # to use for each iteration
TIME = 21 # the firt second to be shown on the camera
COLOR_LINE = (0,0,255) # the color used to add the security cam effect
THICKNESS = 2 # the thickness of the lines

for count in range(len(os.listdir(directory + "\\security_cam1"))):

    filename = directory + "\\security_cam1" + '\\image' + str(count) + '.jpg'
    img = cv2.imread(filename) 
    
    # image properties
    height, width, camadas = img.shape
    size = (width,height)
        
    # adding the lines (the values for each position were obtained by trial and error)

    cv2.line(img, (15,20), (15,60), COLOR_LINE, THICKNESS)
    cv2.line(img, (15,20), (100,20), COLOR_LINE, THICKNESS)
    
    cv2.line(img, (width-15,20), (width-15,60), COLOR_LINE, THICKNESS)
    cv2.line(img, (width-15,20), (width-15-100,20), COLOR_LINE, THICKNESS)
    
    cv2.line(img, (width-15,height-20), (width-15,height-60), COLOR_LINE, THICKNESS)
    cv2.line(img, (width-15,height-20), (width-15-100,height-20), COLOR_LINE, THICKNESS)
    
    cv2.line(img, (15,height-20), (15,height-60), COLOR_LINE, THICKNESS)
    cv2.line(img, (15,height-20), (100,height-20), COLOR_LINE, THICKNESS)
    
    # Putting the day and time
    # the video has 30 fps, then we need to change the number of the seconds every 30 fps 
    # So, I'm gonna sum i until i reaches 30, then change the time.
    # after, add i = -1 to start counting again
    
    if i == 30:
        TIME = TIME + 1
        i = -1
    i = i + 1
    
    # adding the text (brazilian style). The date was chosen as any day
    label = "23/01/2020, QUI 20:03:" + str(TIME)
    cv2.putText(img, label, (18,40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    
    # putting the name of the camera (any name)
    label = "Camera 3"    
    cv2.putText(img, label, (width-110,height-30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # saving the frame with the security cam style
    cv2.imwrite(directory + "\\security_cam2" + '\\image' + str(count) + '.jpg', img)
    
    print("Creating... image" + str(count) + ".jpg" )
    
cv2.destroyAllWindows() 
print("Done!")


# Generating the video with the cam security effect

img_array = []

for count in range(len(os.listdir(directory + "\\security_cam2"))):
    filename = directory + "\\security_cam2" + "\\image" + str(count) + ".jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter("cam_security.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)    

for i in range(len(img_array)):
    out.write(img_array[i])
out.release() 
print("done!")


## Step 4
## Adding defective effect to the security cam
# To simulate the defective effect, I'm going to use the Salt-and-Pepper noise

try:
    if not os.path.exists("defective"):
        os.makedirs("defective")
except OSError:
    print("It was not possible to create the 'defective' directory")

# the function to add the Salt-and-Pepper noise (source: https://stackoverflow.com/a/27342545)    
# this function takes 2 args: image, which is the image instance, and prob, which is the probability. 
# to get more noisy, prob must be closer to 0.5
# to get jus a bit of noise, prb must be closer to 0.01
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# Here I'm going to add the effect in some range of frames. Just remember that every 30 frames is equal to 1 second

for count in range(len(os.listdir(directory + "\\security_cam2"))):
    filename = directory + "\\security_cam2" + '\\image' + str(count) + '.jpg'
    img = cv2.imread(filename) 
    
    if count < 200:
        noise_img = sp_noise(img,0.0505)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)

    elif count > 200 and count < 210:
        noise_img = sp_noise(img,0.01)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)
        
    elif count > 350 and count < 410:
        noise_img = sp_noise(img,0.04)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)

    elif count > 450 and count < 480:
        noise_img = sp_noise(img,0.35)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)

    elif count > 610 and count < 670:
        noise_img = sp_noise(img,0.012)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)
        
    elif count > 745:
        noise_img = sp_noise(img,0.5)
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', noise_img)

    else: # do not apply the Salt-and-Pepper effect
        cv2.imwrite(directory + "\\defective" + '\\image' + str(count) + '.jpg', img)
        
    print("Creating... image" + str(count) + ".jpg" )
    
cv2.destroyAllWindows() 
print("Done!")


# Generating the video with the Salt-and-Pepper effect

img_array = []

for count in range(len(os.listdir(directory + "\\defective"))):
    filename = directory + "\\defective" + "\\image" + str(count) + ".jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter("defective.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)    

for i in range(len(img_array)):
    out.write(img_array[i])
out.release() 
print("done!")
