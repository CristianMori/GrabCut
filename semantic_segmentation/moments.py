import numpy as np
import cv2

classImgs = []

def getCentralMoments(image):
    assert len(image.shape) == 2
    x, y = np.mgrid[:image.shape[0],:image.shape[1]]

    moments = {}
    meanX = np.sum(x*image)/np.sum(image)
    meanY = np.sum(y*image)/np.sum(image)

    moments['11'] = np.sum((x - meanX) * (y-meanY) * image)
    moments['02'] = np.sum((y - meanY) ** 2 * image)
    moments['20'] = np.sum((x - meanX) ** 2 * image)
    moments['12'] = np.sum((x - meanX) * (y-meanY) ** 2 * image)
    moments['21'] = np.sum((x - meanX) ** 2 * (y-meanY) * image) 
    moments['03'] = np.sum((y - meanY) ** 3 * image) 
    moments['30'] = np.sum((x - meanX) ** 3 * image)

    return moments

def getHuMoments(n):
    hu = []

    hu.append( n['20'] + n['02'] )
    hu.append( (n['20'] - n['02'])**2 + 4*n['11']**2 )
    hu.append( (n['30'] - 3*n['12'])**2 + (3*n['21'] - n['03'])**2 )
    hu.append( (n['30'] + n['12'])**2 + (n['21'] + n['03'])**2 )
    hu.append( (n['30'] - 3*n['12'])*(n['30'] + n['12'])*( (n['30'] + n['12'])**2 - 3*(n['21'] - n['03'])**2 ) + (3*n['21'] - n['03'])*(n['21'] + n['03']) * ( 3*(n['30'] + n['12'])**2 - (n['21'] + n['03'])**2) )
    hu.append( (n['20'] - n['02'])*( (n['30'] + n['12'])**2 - (n['21'] + n['03'])**2 ) + 4*n['11']*(n['30'] + n['12'])*(n['21'] + n['03']) )
    hu.append( (3*n['21'] - n['03'])*(n['30'] + n['12'])*( 3*(n['30'] + n['12'])**2 - 3*(n['21'] + n['03'])**2 ) - (n['30'] - 3*n['12'])*(n['21'] + n['03'])*( 3*(n['30'] + n['12'])**2 - (n['21'] + n['03'])**2 ) )

    return hu

def getHuDistance(moments1, moments2):
    dist = 0

    for i in range(0, 7):
        left = np.sign(moments1[i])/np.log(np.abs(moments1[i]))
        right = np.sign(moments2[i])/np.log(np.abs(moments2[i]))
        dist = dist + np.abs(left - right)

    return dist

def getSegmentDistance(image, classNum):
    dist = float('inf')

    segmentMoments = getCentralMoments(image)
    segmentHuMoments = getHuMoments(segmentMoments)

    for i in range(0, 4):
        classMoments = getCentralMoments(classImgs[classNum][i])
        classHuMoments = getHuMoments(classMoments)
        curDist = getHuDistance(segmentHuMoments, classHuMoments)

        if curDist < dist:
            dist = curDist

    return dist    


def getSegmentClass(image):
    classNum = -1
    distances = []

    for i in range(0, len(classImgs)):
        curDist = getSegmentDistance(image, i)
        distances.append(curDist)

    print("Distance to Car class: ", distances[0])
    print("Distance to Flower class: ", distances[1])

    return distances.index(min(distances))

if __name__ == '__main__':

    # Build training sets
    cars = []
    cars.append(cv2.imread("Training Set/Foreground/Car/Car1.png",  flags=cv2.IMREAD_UNCHANGED))
    cars.append(cv2.imread("Training Set/Foreground/Car/Car2.png",  flags=cv2.IMREAD_UNCHANGED))
    cars.append(cv2.imread("Training Set/Foreground/Car/Car3.png",  flags=cv2.IMREAD_UNCHANGED))
    cars.append(cv2.imread("Training Set/Foreground/Car/Car4.png",  flags=cv2.IMREAD_UNCHANGED))

    flowers = []
    flowers.append(cv2.imread("Training Set/Foreground/Flower/Flower1.png",  flags=cv2.IMREAD_UNCHANGED))
    flowers.append(cv2.imread("Training Set/Foreground/Flower/Flower2.png",  flags=cv2.IMREAD_UNCHANGED))
    flowers.append(cv2.imread("Training Set/Foreground/Flower/Flower3.png",  flags=cv2.IMREAD_UNCHANGED))
    flowers.append(cv2.imread("Training Set/Foreground/Flower/Flower4.png",  flags=cv2.IMREAD_UNCHANGED))

    classImgs.append(cars)
    classImgs.append(flowers)

    testCar = cv2.imread("./Test Images/flower1.png", flags=cv2.IMREAD_UNCHANGED)

    for i in range(0, len(classImgs)):
        for j in range(0, len(classImgs[i])):
            for k in range(0, len(classImgs[i][j])):
                for l in range(0, len(classImgs[i][j][k])):                
                    if classImgs[i][j][k][l][3] <= 20:
                        classImgs[i][j][k][l][0] = 0
                        classImgs[i][j][k][l][1] = 0
                        classImgs[i][j][k][l][2] = 0

    for i in range(0, len(testCar)):
        for j in range(0, len(testCar[i])):
            if testCar[i][j][3] <= 20:
                testCar[i][j][0] = 0
                testCar[i][j][1] = 0
                testCar[i][j][2] = 0

    testCarGray = cv2.cvtColor(testCar, cv2.COLOR_BGR2GRAY)

    for i in range(0, len(classImgs)):
        for j in range(0, len(classImgs[i])):
            classImgs[i][j] = cv2.cvtColor(classImgs[i][j], cv2.COLOR_BGR2GRAY)


    cv2.imshow("test", testCarGray)
    cv2.waitKey(0)

    print(getSegmentClass(testCarGray))
