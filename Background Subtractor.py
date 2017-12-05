# Imported PIL Library
from PIL import Image

# Open an Image
def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'jpg')

def blobbing_image(folder, src,threshold,iteration):
    from PIL import Image
    #im = Image.open('C:/Users/hitesh b/Desktop/Reviewfolder1/erode/frame15.jpg')
    #'C:/Users/hitesh b/Desktop/testVideo/ReviewfolderVid2/frame7+.jpg'
    im = Image.open(folder+'/'+src)
    dummy=im


    pixelMap = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if pixelMap[i,j]<128 :
                pixelMap[i,j] = 0
            else:
                pixelMap[i,j] = 255
    #im.show()
    pixelMap2=dummy.load()


    #Creating my own blob detector
    keypoint=[]
    newPixelMap=pixelMap2
    max_point=100
    #Edge making
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if i==0 or j==0:
                newPixelMap[i,j]=0

    for i in range(1,im.size[0]-1):
        for j in range(1,im.size[1]-1):
            newPixelMap[i,j]=int((pixelMap[i-1,j-1]+pixelMap[i-1,j]+pixelMap[i-1,j+1]+pixelMap[i,j-1]+pixelMap[i,j+1]+pixelMap[i+1,j-1]+pixelMap[i+1,j]+pixelMap[i+1,j+1])/8)
            if newPixelMap[i,j]>max_point:
                keypoint.append(newPixelMap[i,j])
                max_point=keypoint[-1]
            #print(newPixelMap[i,j])
    keypoint.sort()

    #Implementation Blobbing

    #Implementation Blobbing
    for k in range(0,iteration):
        for i in range(1,im.size[0]-1):
            for j in range(1,im.size[1]-1):
                if newPixelMap[i,j] in keypoint:
                    if(newPixelMap[i-1,j-1]<newPixelMap[i,j]):
                        newPixelMap[i-1,j-1]=newPixelMap[i,j]-1
                    if(newPixelMap[i-1,j]<newPixelMap[i,j]):
                        newPixelMap[i-1,j]=newPixelMap[i,j]-1
                    if(newPixelMap[i-1,j+1]<newPixelMap[i,j]):
                        newPixelMap[i-1,j+1]=newPixelMap[i,j]-1
                    if(newPixelMap[i,j-1]<newPixelMap[i,j]):
                        newPixelMap[i,j-1]=newPixelMap[i,j]-1
                    if(newPixelMap[i,j+1]<newPixelMap[i,j]):
                        newPixelMap[i,j+1]=newPixelMap[i,j]-1
                    if(newPixelMap[i+1,j-1]<newPixelMap[i,j]):
                        newPixelMap[i+1,j-1]=newPixelMap[i,j]-1
                    if(newPixelMap[i+1,j]<newPixelMap[i,j]):
                        newPixelMap[i+1,j]=newPixelMap[i,j]-1
                    if(newPixelMap[i+1,j+1]<newPixelMap[i,j]):
                        newPixelMap[i+1,j+1]=newPixelMap[i,j]-1
        for p in range(0,len(keypoint)):
            keypoint[p]=keypoint[p]-1
        #keypoint[idx]=newPixelMap[i,j]-1
        #print("idx.{}".format(idx))
        #print("keypoint.{}".format(keypoint[idx]))


    for i in range(1,im.size[0]-1):
        for j in range(1,im.size[1]-1):
            if(newPixelMap[i,j]<threshold):
                newPixelMap[i,j]=0
            else:
                newPixelMap[i,j]=255
    #'C:/Users/hitesh b/Desktop/golf7_10.jpg'
    #dummy.save(dest)
    dummy.save(os.path.join(folder+'/blob',"frame{:d}.jpg".format(count)))


def ero_dilution(folder,img):
    # Python program to demonstrate erosion and
    # dilation of images.
    import cv2
    import os
    import numpy as np
    # Reading the input image
    img = cv2.imread(folder+"/"+img, 0)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_dilation = cv2.dilate(img, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)

    cv2.imwrite(os.path.join(folder+'/erode',"frame{:d}.jpg".format(count)), img_dilation)     # save frame as JPEG file
    #cv2.imshow('Input', img)
    #cv2.imshow('Erosion', img_erosion)
    #cv2.imshow('Dilation', img_dilation)
    #cv2.waitKey(0)
    print("Completed working on frame{:d}.jpg".format(count))



import cv2
# create a folder to store extracted images
import os
folder = 'C:/Users/kisha/Desktop/testVideo/Diving'
os.mkdir(folder)
os.mkdir(folder+'/erode')
os.mkdir(folder+'/blob')
"""
-->BackgroundSubtractorMOG2(int history, float varThreshold, bool bShadowDetection=true )
Parameters:
history – Length of the history.
varThreshold – Threshold on the squared Mahalanobis distance to decide whether it is well described by the background model (see Cthr??). This parameter does not affect the background update. A typical value could be 4 sigma, that is, varThreshold=4*4=16; (see Tb??).
bShadowDetection – Parameter defining whether shadow detection should be enabled (true or false).
"""
fgbg = cv2.createBackgroundSubtractorMOG2(history=20,varThreshold=500,detectShadows=False)

'''
-->cv2.BackgroundSubtractorMOG([history, nmixtures, backgroundRatio[, noiseSigmla]]) → <BackgroundSubtractorMOG object>
Parameters:
history – Length of the history.
nmixtures – Number of Gaussian mixtures.
backgroundRatio – Background ratio.
noiseSigma – Noise strength.
Default constructor sets all parameters to default values.
'''
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 5,decisionThreshold = 0.5 )

"""
--->createBackgroundSubtractorGMG()
cv2.bgsegm.createBackgroundSubtractorGMG	(initializationFrames = 120,
decisionThreshold = 0.8 )
Creates a GMG Background Subtractor.
Parameters
initializationFrames:	number of frames used to initialize the background models.
decisionThreshold:	Threshold value, above which it is marked foreground, else background.
"""
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG (history=10,nmixtures=5, backgroundRatio=0.7,noiseSigma=0)

print(cv2.__version__)
vidcap = cv2.VideoCapture('C:/Users/kisha/Desktop/testVideo/2538-5_70133.avi')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    #Saving Noraml colored frame
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    #Saving MOG applied frame
    cv2.imwrite(os.path.join(folder,"frame{:d}+.jpg".format(count)), fgbg.apply(image))     # save frame as JPEG file
    img_str=folder + "/frame{:d}+.jpg".format(count)


    ero_dilution(folder,"frame{:d}+.jpg".format(count))

    blobbing_image(folder,"frame{:d}+.jpg".format(count),100,8)

    count += 1
print("{} images are extacted in {}.".format(count,folder))

vidcap.release()
cv2.destroyAllWindows()

img1=cv2.imread('C:/Users/kisha/Desktop/testVideo/Diving/blob/frame0.jpg')
height , width , layers =  img1.shape
video = cv2.VideoWriter('C:/Users/kisha/Desktop/testVideo/Diving/outputvideo.avi',-1,1,(720,404))
count=0
while True:
    video.write(cv2.imread('frame{:d}.jpg'.format(count)))
    count=count+1
cv2.destroyAllWindows()
video.release()