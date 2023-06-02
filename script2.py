import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import dlib
import cv2

import operator

#Model for face detect
predictor_path = 'predictors/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def analyseFace( img, quality=1, offset=(0,0)):
    dets = detector(np.array(img), quality)
    result=[]
    for k, d in enumerate(dets):
        instantFacePOI = np.zeros((7,2),dtype=np.float32)
        eyeCorners=np.zeros((2,2,2),dtype=np.float32)
        # Get the landmarks/parts for the face in box d.
        shape = predictor(np.array(img), d)
        #oreille droite
        instantFacePOI[0][0]=shape.part(0).x+offset[0];
        instantFacePOI[0][1]=shape.part(0).y+offset[1];
        #oreille gauche
        instantFacePOI[1][0]=shape.part(16).x+offset[0];
        instantFacePOI[1][1]=shape.part(16).y+offset[1];
        #nez
        instantFacePOI[2][0]=shape.part(30).x+offset[0];
        instantFacePOI[2][1]=shape.part(30).y+offset[1];
        #bouche gauche
        instantFacePOI[3][0]=shape.part(48).x+offset[0];
        instantFacePOI[3][1]=shape.part(48).y+offset[1];
        #bouche droite
        instantFacePOI[4][0]=shape.part(54).x+offset[0];
        instantFacePOI[4][1]=shape.part(54).y+offset[1];

        leftEyeX=0
        leftEyeY=0
        for i in range(36, 42):
            leftEyeX+=shape.part(i).x
            leftEyeY+=shape.part(i).y
        leftEyeX=int(leftEyeX/6.0)
        leftEyeY=int(leftEyeY/6.0)
        eyeCorners[0][0]=[shape.part(36).x+offset[0],shape.part(36).y+offset[1]]
        eyeCorners[0][1]=[shape.part(39).x+offset[0],shape.part(39).y+offset[1]]

        instantFacePOI[5][0]=leftEyeX+offset[0];
        instantFacePOI[5][1]=leftEyeY+offset[1];

        rightEyeX=0
        rightEyeY=0
        for i in range(42, 48):
            rightEyeX+=shape.part(i).x
            rightEyeY+=shape.part(i).y
        rightEyeX=int(rightEyeX/6.0)
        rightEyeY=int(rightEyeY/6.0)
        eyeCorners[1][0]=[shape.part(42).x+offset[0],shape.part(42).y+offset[1]]
        eyeCorners[1][1]=[shape.part(45).x+offset[0],shape.part(45).y+offset[1]]
        instantFacePOI[6][0]=rightEyeX+offset[0];
        instantFacePOI[6][1]=rightEyeY+offset[1];
        data=[instantFacePOI, (int(d.left()+offset[0]),int(d.top()+offset[1]),int(d.right()+offset[0]),int(d.bottom()+offset[1])),eyeCorners]
        result.append(data)
    return result



kGradientThreshold = 10.0
kWeightBlurSize = 5;
maxEyeSize=10;

def computeGradient(img):
    out = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32) #create a receiver array
    if img.shape[0] < 2 or img.shape[1] < 2: # TODO I'm not sure that secure out of range
        print("EYES too small")
        return out
    for y in range(0,out.shape[0]):
        out[y][0]=img[y][1]-img[y][0]
        for x in range(1,out.shape[1]-1):
            out[y][x]=(img[y][x+1]-img[y][x-1])/2.0
        out[y][out.shape[1]-1]=img[y][out.shape[1]-1]-img[y][out.shape[1]-2]
    return out

def testPossibleCentersFormula(x, y, weight, gx, gy, out):
    for cy in range(0,out.shape[0]):
        for cx in range(0,out.shape[1]):
            if x==cx and y==cy :
                continue
            dx= x-cx
            dy= y-cy
            magnitude= math.sqrt(dx*dx+dy*dy)
            dx=dx/magnitude
            dy=dy/magnitude
            dotProduct=dx*gx+dy*gy
            dotProduct=max(0.0, dotProduct)
            out[cy][cx]+=dotProduct*dotProduct*weight[cy][cx]

def matrixMagnitude(gradX,gradY):
    mags = np.zeros((gradX.shape[0],gradX.shape[1]),dtype=np.float32) #create a receiver array
    for y in range(0,mags.shape[0]):
        for x in range(0,mags.shape[1]):
            gx=gradX[y][x]
            gy=gradY[y][x]
            magnitude=math.sqrt(gx*gx+gy*gy)
            mags[y][x]=magnitude
    return mags


def computeDynamicThreshold(gradientMatrix,DevFactor ):
    (meanMagnGrad, meanMagnGrad) = cv2.meanStdDev(gradientMatrix)
    stdDev=meanMagnGrad[0]/math.sqrt(gradientMatrix.shape[0]*gradientMatrix.shape[1])
    return DevFactor*stdDev+meanMagnGrad[0]

def getEyePOI(eyes):
    result=[]
    for eye in eyes:
        left=eye[0][0]
        right=eye[1][0]
        middle=(eye[0][1]+eye[1][1])/2.0
        width=eye[1][0]-eye[0][0]
        height=width/4.0
        result.append((int(left),int(middle-height),int(right),int(middle+height)))
    return result

def scale(rectangle,scale):
    width=rectangle[2]-rectangle[0]
    height=rectangle[3]-rectangle[1]
    midddle=(width/2+rectangle[0], height/2+rectangle[1])
    left=midddle[0]-int(scale*width/2)
    top=midddle[1]-int(scale*height/2)
    right=midddle[0]+int(scale*width/2)
    bottom=midddle[1]+int(scale*height/2)
    return (left, top, right, bottom)

def getEyePos(corners,img):
    #here we don't need both but the biggest one
    eyes=getEyePOI(corners)
    choosen=0
    eyeToConsider=eyes[0]
    if((eyes[0][0]-eyes[0][2])>(eyes[1][0]-eyes[1][2])):
        eyeToConsider=eyes[1]
        choosen=1


    scalesrect=scale(eyeToConsider,1.2)
    croppedImage = img[
        int(max(scalesrect[1],0)):int(max(scalesrect[3],0)),
        int(max(scalesrect[0],0)):int(max(scalesrect[2],0))
    ]
    return [findEyeCenter(croppedImage, [scalesrect[0],scalesrect[1]]),corners[choosen]]


def findEyeCenter(eyeImage, offset):
    cv2.imshow("ciao", eyeImage)
    if (len(eyeImage.shape) <= 0 or eyeImage.shape[0] <= 0 or eyeImage.shape[1] <= 0):
        return tuple(map(operator.add, (0, 0), offset))
    eyeImg = np.asarray(cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY))
    eyeImg = eyeImg.astype(np.float32)
    scaleValue=1.0;
    if(eyeImg.shape[0] > maxEyeSize or eyeImg.shape[1] > maxEyeSize):
        scaleValue=max(maxEyeSize/float(eyeImg.shape[0]),maxEyeSize/float(eyeImg.shape[1]))
        eyeImg=cv2.resize(eyeImg,None, fx=scaleValue,fy= scaleValue, interpolation = cv2.INTER_AREA)



    gradientX= computeGradient(eyeImg)
    gradientY= np.transpose(computeGradient(np.transpose(eyeImg)))
    gradientMatrix=matrixMagnitude(gradientX, gradientY)

    gradientThreshold=computeDynamicThreshold(gradientMatrix,kGradientThreshold)
    #Normalisation
    for y in range(0,eyeImg.shape[0]):  #Iterate through rows
        for x in range(0,eyeImg.shape[1]):  #Iterate through columns
            if(gradientMatrix[y][x]>gradientThreshold):
                gradientX[y][x]=gradientX[y][x]/gradientMatrix[y][x]
                gradientY[y][x]=gradientY[y][x]/gradientMatrix[y][x]
            else:
                gradientX[y][x]=0.0
                gradientY[y][x]=0.0

    #Invert and blur befor algo
    weight = cv2.GaussianBlur(eyeImg,(kWeightBlurSize,kWeightBlurSize),0)
    for y in range(0,weight.shape[0]):  #Iterate through rows
        for x in range(0,weight.shape[1]):  #Iterate through columns
            weight[y][x]=255-weight[y][x]

    outSum = np.zeros((eyeImg.shape[0],eyeImg.shape[1]),dtype=np.float32) #create a receiver array
    for y in range(0,outSum.shape[0]):  #Iterate through rows
        for x in range(0,outSum.shape[1]):  #Iterate through columns
            if(gradientX[y][x]==0.0 and gradientY[y][x]==0.0):
                continue
            testPossibleCentersFormula(x, y, weight, gradientX[y][x], gradientY[y][x], outSum)

    #scale all the values down, basically averaging them
    numGradients = (weight.shape[0]*weight.shape[1]);
    out= np.divide(outSum, numGradients*10)
    #find maxPoint
    (minval, maxval,mincoord,maxcoord) = cv2.minMaxLoc(out)
    maxcoord=(int(maxcoord[0]/scaleValue),int(maxcoord[1]/scaleValue))
    return tuple(map(operator.add, maxcoord, offset))




vid = cv2.VideoCapture(0)

while 1:
    ret, test = vid.read()


    #TEST PICTURE
    
    #test = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_data= analyseFace(test)

    eye_centers=[]
    for index,POI in enumerate(faces_data):
        eye_corners=POI[2]
        eye_center=getEyePos(eye_corners,test)
        eye_centers.append(eye_center)
        cv2.circle(test,(int(eye_center[0][0]),int(eye_center[0][1])), 4, (255,0,0), -1)
        cv2.circle(test,(int(eye_center[1][0][0]),int(eye_center[1][0][1])), 4, (0,0,255), -1)
        cv2.circle(test,(int(eye_center[1][1][0]),int(eye_center[1][1][1])), 4, (0,0,255), -1)
    
    cv2.imshow('frame',  test)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break


vid.release()