import cv2
import numpy as np
from tensorflow.keras.models import load_model


#### READ THE MODEL WEIGHTS
def intializePredectionModel():
    model = load_model('myData1/resources/myModel.h5')
    return model

def preProcess(img):
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),1)
    thres = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return thres

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:  #to ensure that the biggest contour is a quadrilateral
                biggest = approx #approx is a list of corner points
                max_area = area
    return biggest,max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        #predictions = model.predict(img)

        predictions = model.predict(img)
        classes = np.argmax(predictions, axis=1)

        #classIndex = model.predict_classes(img)  #class index is the digit that it is predicting
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        #result.append(classes)
        if probabilityValue > 0.95:
            result.append(classes[0])
        else:                              #If prob is less than 80% it means that it is a blank space
            result.append(0)
    return result


#Display the detected numbers on the warped screen of ours that is 450 * 450 according to their index of detection
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, color,2, cv2.LINE_AA)
    return img

def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


#To stack all the images in one frame
def stackImages(imgarr,scale):
    rows = len(imgarr)
    cols = len(imgarr[0])
    rowsavail = isinstance(imgarr[0], list)
    width = imgarr[0][0].shape[1]
    height = imgarr[0][0].shape[0]
    if rowsavail:
        for x in range(0, rows):
            for y in range(0, cols):
                imgarr[x][y] = cv2.resize(imgarr[x][y], (0, 0), None, scale, scale)
                if len(imgarr[x][y].shape) == 2:
                    imgarr[x][y]= cv2.cvtColor(imgarr[x][y], cv2.COLOR_GRAY2BGR)
        blank = np.zeros((height, width, 3), np.uint8)
        hor = [blank]*rows
        hor_con = [blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgarr[x])
            hor_con[x] = np.concatenate(imgarr[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0,rows):
            imgarr[x] = cv2.resize(imgarr[x], (0, 0), None, scale, scale)
            if len(imgarr[x].shape) == 2:
                imgarr[x] = cv2.cvtColor(imgarr[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgarr)
        #hor_con = np.concatenate(imgarr)
        ver = hor
    return ver

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes
