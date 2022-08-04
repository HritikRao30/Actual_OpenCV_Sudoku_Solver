print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import backtrack
###########################################################
pathImg = "myData1/resources/sudoku.png";
height = 450
width = 450

model = intializePredectionModel()  # LOAD THE DIGIT RECOGNITION CNN MODEL
###########################################################

#1 prepare the threshold image
img = cv2.imread(pathImg)
img = cv2.resize(img, (width,height))
imgblank = np.zeros((height,width,3),np.uint8)
imgThres = preProcess(img)    #This is done to get an adaptive threshold image of the soduko where the boundary gets highlighted to a high extent


#2 get the contour image
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR drawing the biggest contour
contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS RETR external will only draw the external contours
cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 3) # DRAW ALL DETECTED CONTOURS color is blue

#3 get the biggest contour
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR

if biggest.size != 0:
    rearranged = reorder(biggest)   #has been rearranged to fit into the order of warp perspective algorithm
    print(rearranged)
    cv2.drawContours(imgBigContour,rearranged,-1,(255,0,0),10)
    pts1 = np.float32(rearranged) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # generate the transformation matrix
    imgWarped = cv2.warpPerspective(img, matrix, (width, height))
    imgDetectedDigits = imgblank.copy()   #blank image to show the detected digits on the blank screen
    imgWarpedGrey = cv2.cvtColor(imgWarped,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("stacked_images", imgWarped)

    #4 divide the warped image into smaller boxes of containing  single digits

    imgSolvedDigits = imgblank.copy()    #to display the solved digits that were added at the end
    boxes = splitBoxes(imgWarpedGrey)    #prediction works only on grey scale as we have trained it only on grey scale hence we will have to preprocess the images before testing them
    #print(len(boxes))
    #cv2.imshow("Sample",boxes[62])
    board = getPredection(boxes,model)
    #print(board)
    imgDetectedDigits = displayNumbers(imgDetectedDigits,board)
    
    board = np.asarray(board)
    #print(board)
    posArray = np.where(board > 0, 0, 1)    #this is going to give the empty array for us to backtrack

    #### 5. FIND SOLUTION OF THE BOARD
    brd = np.array_split(board,9)
    #print(brd)
    try:
        if backtrack.solve(brd) == False:
            print("No solution Possible")#this is the backtracking function
    except:
        pass
    print(brd)

    flatList = []
    for sublist in brd:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray    #it will containe the solved digits that were added by our bactracking code
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)
    #6. OVERLAY SOLUTION
    pts2 = np.float32(rearranged) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[width, 0], [0, height],[width, height]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER   Now wev are doing the reverse warp from warped to the original biggest contour polygon
    imgInvWarpColored = img.copy()  #inverse perspetive to overlap the solution back on the original partially solved sudoku
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width,height))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)
    imgarr = ([imgBigContour,inv_perspective])
    stackedImg = stackImages(imgarr, 1)
    cv2.imshow("stacked_images", stackedImg)
else:   #if no polygon or box detected
    print("No Sudoku Found")

cv2.waitKey(0)
