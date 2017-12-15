import numpy as np

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
        classMoments = getCentralMoments(classImg[classNum][i])
        classHuMoments = getHuMoments(classMoments)
        curDist = getHuDistance(segmentHuMoments, classHuMoments)

        if curDist < dist:
            dist = curDist

    return dist    


def getSegmentClass(image):
    dist = float('inf')
    classNum = -1

    for i in range(0, len(classImgs)):
        curDist = getSegmentDistance(image, i)

        if curDist < dist:
            dist = curDist
            classNum = i

    return classNum

if __name__ == '__main__':
    imgGray1 = cv2.cvtColor(classImgs[0][0], cv2.COLOR_BGR2GRAY)
    imgGray2 = cv2.cvtColor(classImgs[0][1], cv2.COLOR_BGR2GRAY)

    moment1 = getCentralMoments(imgGray1)
    moment2 = getCentralMoments(imgGray2)

    print(moment2)

    huMoment1 = getHuMoments(moment1)
    huMoment2 = getHuMoments(moment2)
 
    print(getHuDistance(huMoment1, huMoment2))

    print(getSegment(classImgs[0][0]))
