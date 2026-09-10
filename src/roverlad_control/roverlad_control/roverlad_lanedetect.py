import cv2
import numpy as np
import math


class LaneHandler:
    def __init__(self):
        self.pureWhiteRGB = (255, 255, 255) 
        self.lowerWhite = np.array([0, 0, 140])
        self.upperWhite = np.array([180, 40, 255])

        self.pureYellowRGB = (0, 255, 255) 
        self.lowerYellow = np.array([15, 20, 70]) 
        self.upperYellow = np.array([40, 255, 255])

    def run(self, img, debugCV=False):        
        img = cv2.resize(img, (960, 540))        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
        
        imgCopy = img.copy()

        h, w = imgCopy.shape[:2]
        # Get Masks

        hsvWhiteMask = cv2.inRange(hsv, self.lowerWhite, self.upperWhite)

        b, g, r = cv2.split(img)

        bgrWhiteMask = (
            (r > 170) & (g > 170) & (b > 170) &
            (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 30) &
            (np.abs(r.astype(np.int16) - b.astype(np.int16)) < 30) &
            (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 30)
        ).astype(np.uint8) * 255

        whiteMask = cv2.bitwise_and(hsvWhiteMask, bgrWhiteMask)

        # whiteMask = cv2.inRange(hsv, self.lowerWhite, self.upperWhite) 
        yellowMask = cv2.inRange(hsv, self.lowerYellow, self.upperYellow) 

        roiMask = np.zeros_like(whiteMask)
        roiMask[int(h * 0.40):, :] = 255

        whiteMask = cv2.bitwise_and(whiteMask, roiMask)
        yellowMask = cv2.bitwise_and(yellowMask, roiMask)

        whiteMask = self.processLaneMask(whiteMask, closeKernal=(5, 5))   
        yellowMask = self.processLaneMask(yellowMask)  

        if self.detectIntersection(whiteMask):
            print("Maybe at an intersection?")
            return imgCopy, 0

        laneMask = cv2.bitwise_or(whiteMask, yellowMask)
        
        if (debugCV):
            cv2.imshow("whiteMask", whiteMask)
            cv2.imshow("yellowMask", yellowMask)
            cv2.imshow("laneMask", laneMask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()        

        # Left and Right point clusters
        leftCluster, rightCluster, vis = self.getLineClusters(laneMask, yellowMask, whiteMask)

        if (debugCV):
            cv2.imshow("Vis", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()     

        # Left and Right lines
        ptl = self.getLineSegmentFromCluster(leftCluster) if leftCluster else None
        ptr = self.getLineSegmentFromCluster(rightCluster) if rightCluster else None        

        # ptl1 and ptr2 are upper points
        if ptl:
            ptl1, ptl2 = ptl
            cv2.line(imgCopy, ptl1, ptl2, (255, 0, 0), 3)        
        if ptr:
            ptr1, ptr2 = ptr
            cv2.line(imgCopy, ptr1, ptr2, (0, 0, 255), 3)


        # Check if points are valid
        if ptl and ptr:
            midPt = ((ptl1[0] + ptr2[0]) // 2, (ptl1[1] + ptr2[1]) // 2 )        
        elif ptl and not ptr:
            midPt = ((ptl1[0] + w) // 2,  (ptl1[1]+h)//3)
        elif ptr and not ptl:
            midPt = ((0 + ptr2[0]) // 2, (ptr1[1]+h)//3)
        else:
            midPt = (w // 2, h // 2)

        # Center/Current Line/Vehicle heading
        self.drawCentralLine(imgCopy)
                
        # Target Line/Vehicle heading
        self.drawCentralLineTo(imgCopy, midPt)
        cv2.circle(imgCopy, midPt, 6, (100, 255, 100), -1)  # light green dot
        
        angleDeg = self.getSteeringAngle(imgCopy, midPt)

        return imgCopy, angleDeg
    
    def getSteeringAngle(self, img, pt):
        # Compute Target Angle of Heading
        h, w = img.shape[:2]

        xBot = w // 2
        yBot = h

        v2 = (pt[0] - xBot, pt[1] - yBot)

        angle = math.atan2(v2[0], -v2[1]) 
        angleDeg = math.degrees(angle)

        return angleDeg
        
    # Draw line from bottom of image to center of image
    def drawCentralLine(self, img):
        h, w = img.shape[:2]
        xBot = w // 2
        yBot = h
        yTop = h // 2
        
        p1 = (xBot, yBot)
        p2 = (xBot, yTop)
        
        cv2.line(img, p1, p2, (0, 255, 0), 5)

    # Draw line from bottom of image to pt
    def drawCentralLineTo(self, img, pt):
        h, w = img.shape[:2]
        xBot = w // 2
        yBot = h
        
        p1 = (xBot, yBot)
        
        cv2.line(img, p1, pt, (150, 255, 150), 5)


    # Get white and yellow lines
    def getLineClusters(self, laneMask, leftMask, rightMask):
        leftPts = self.getLineCluster(leftMask, "left")
        rightPts = self.getLineCluster(rightMask, "right")
        
        vis = np.zeros((laneMask.shape[0], laneMask.shape[1], 3), dtype=np.uint8)

        if leftPts is not None:
            pts = np.array(leftPts)
            vis[pts[:, 1], pts[:, 0]] = self.pureWhiteRGB
        
        if rightPts is not None:
            pts = np.array(rightPts)
            vis[pts[:, 1], pts[:, 0]] = self.pureYellowRGB

        return leftPts, rightPts, vis

    # Get line from side of image
    def getLineCluster(self, laneMask, lane="left"):
        pt = self.findFirstPoint(laneMask, lane)
        # print(lane, pt)
        cluster = self.getConnectedComponent(laneMask, pt) if pt else []

        if cluster:
            return cluster
            
        return None
    
    def processLaneMask(self, laneMask, openKernal=(3,3), closeKernal=(8,8)):
        # Filter out shadows and buildings and other large objects
        badMask = self.contourFilter(laneMask)
        laneMask = cv2.bitwise_and(laneMask, cv2.bitwise_not(badMask)) 
    
        # Removes Small noise and specks and Fills gaps
        kernel = np.ones(openKernal, np.uint8)
        laneMask = cv2.morphologyEx(laneMask, cv2.MORPH_OPEN, kernel)    
        kernel = np.ones(closeKernal, np.uint8)
        laneMask = cv2.morphologyEx(laneMask, cv2.MORPH_CLOSE, kernel)    
    
        # Remove thin vertical lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # horizontal bias
        laneMask = cv2.erode(laneMask, kernel)
        laneMask = cv2.dilate(laneMask, kernel)
        return laneMask
            
    # Detects and returns large areas of noise (like shadows, buildings)
    def contourFilter(self, inMask):
        contours, _ = cv2.findContours(inMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        badMask = np.zeros_like(inMask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
        
            x, y, w, h = cv2.boundingRect(cnt)
            elongation = max(w, h) / (min(w, h) + 1e-6)
        
            vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = abs(vy / (vx + 1e-6))
        
            if (elongation > 3 and slope > 0.3) or slope > 1.0:
                cv2.drawContours(badMask, [cnt], -1, 255, -1)
    
        return badMask

    # Scan a side of an image from bottom-to-top for a point.
    def findFirstPoint(self, edges, side="left"):
        h, w = edges.shape
    
        if side == "left":
            xs = np.arange(w)
        else:
            xs = np.arange(w - 1, -1, -1)
    
        for x in xs:
            ys = np.where(edges[:, x] > 0)[0]
            if ys.size > 0:
                return (x, ys[-1])  # bottom-most
    
        return None

    # From a point, get the edges/blobs that it's connected to.
    def getConnectedComponent(self, edges, startPt):
        mask = (edges > 0).astype(np.uint8)
        numLabels, labels = cv2.connectedComponents(mask, connectivity=8)
    
        x, y = startPt
        label = labels[y, x]
    
        if label == 0:
            return []
    
        ys, xs = np.where(labels == label)
        return list(zip(xs, ys))

    # Draw a line that fits a point/pixel cluster
    def drawLineFromCluster(self, vis, cluster, color):
        if len(cluster) < 2:
            return
    
        pts = np.array(cluster, dtype=np.int32)    
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()    
        h, w = vis.shape[:2]
    
        if abs(vy) < 1e-6:
            return
    
        y1 = 0
        y2 = h - 1
    
        x1 = int(x0 + (y1 - y0) * (vx / vy))
        x2 = int(x0 + (y2 - y0) * (vx / vy))
    
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)

    # From a blob, draw a line to fit it.
    def getLineSegmentFromCluster(self, cluster, maxLineLength=300):
        if len(cluster) < 2:
            return None
    
        pts = np.array(cluster, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    
        v = np.array([vx, vy])
    
        projections = [
            np.dot(np.array([x - x0, y - y0]), v)
            for x, y in pts
        ]
    
        projections = np.array(projections)
    
        tMin = projections.min()
        tMax = projections.max()

        # Original endpoints
        pt1 = np.array([x0 + tMax * vx, y0 + tMax * vy])
        pt2 = np.array([x0 + tMin * vx, y0 + tMin * vy])

        # Limit line length
        lineVector = pt1 - pt2
        lineLength = np.linalg.norm(lineVector)

        if lineLength > maxLineLength:
            direction = lineVector / lineLength
            pt1 = pt2 + direction * maxLineLength

        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))

        return pt1, pt2

    
    def detectIntersection(self, whiteMask):
        h, w = whiteMask.shape

        # bottomMask = whiteMask[h // 2:, :]
        bottomMask = whiteMask[int(h * 0.3):, :]

        contours, _ = cv2.findContours(bottomMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largeBlobs = []
        totalWhiteArea = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 100:
                continue

            x, y, blobW, blobH = cv2.boundingRect(cnt)

            # Require the blob to have some physical size
            if blobW < 10 or blobH < 5:
                continue

            largeBlobs.append(cnt)
            totalWhiteArea += area

        blobCount = len(largeBlobs)

        print("blobs:", blobCount, " white area ", totalWhiteArea)

        return (blobCount >= 3 and totalWhiteArea > 4000)