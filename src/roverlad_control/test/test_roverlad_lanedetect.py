import pytest
import cv2
import numpy as np
import math
from roverlad_control.roverlad_lanedetect import LaneHandler


laneHandler = LaneHandler()


def getSteeringAngle(pt) -> None:
    img = cv2.imread("src/roverlad_control/test/resources/TestImageStraight.png")    
    angle = laneHandler.getSteeringAngle(img, pt)
    return angle    

# Get steering angle at center-point
def test_getSteeringAngle_0deg() -> None:
    angle = getSteeringAngle((640, 360))
    assert angle > -2 and angle < 2
    
# Get steering angle at 45 degrees
def test_getSteeringAngle_45deg() -> None:
    angle = getSteeringAngle((960, 360))
    assert angle > 40 and angle < 50




def findFirstPoint(edges, side="left"):
    point = laneHandler.findFirstPoint(edges, side)
    return point

# Put a point on the left and find it.
def test_findFirstPoint_left() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    # Put a lane pixel at x=20
    edges[80, 20] = 255

    point = findFirstPoint(edges, "left")

    assert point == (20, 80)

# Put a point on the right and find it.
def test_findFirstPoint_right() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    # Put pixels at x=20 and x=80
    edges[80, 20] = 255
    edges[90, 80] = 255

    point = findFirstPoint(edges, "right")

    # Right scan should find x=80 first
    assert point == (80, 90)

# Try and find no points.
def test_findFirstPoint_noPoint() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    point = findFirstPoint(edges, "left")

    assert point is None




def getConnectedComponent(edges, startPt):
    cluster = laneHandler.getConnectedComponent(edges, startPt)
    return cluster

def test_getConnectedComponent() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    # Create a 10x10 connected component
    edges[20:30, 20:30] = 255

    cluster = getConnectedComponent(edges, (25, 25))

    assert len(cluster) == 100
    assert (25, 25) in cluster

def test_getConnectedComponent_separateComponents() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    # Component 1
    edges[20:30, 20:30] = 255

    # Component 2
    edges[60:70, 60:70] = 255

    cluster = getConnectedComponent(edges, (25, 25))

    assert (25, 25) in cluster
    assert (65, 65) not in cluster

def test_getConnectedComponent_background() -> None:
    edges = np.zeros((100, 100), dtype=np.uint8)

    cluster = getConnectedComponent(edges, (50, 50))

    assert cluster == []



def getLineCluster(mask, side="left"):
    cluster = laneHandler.getLineCluster(mask, side)
    return cluster

def test_getLineCluster() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Vertical lane
    mask[20:90, 20] = 255

    cluster = getLineCluster(mask, "left")

    assert cluster is not None
    assert len(cluster) > 1

def test_getLineCluster_noLane() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)

    cluster = getLineCluster(mask, "left")

    assert cluster is None


def test_detectIntersection_threeLargeBlobs() -> None:
    whiteMask = np.zeros((200, 200), dtype=np.uint8)

    # Three separate, sufficiently large blobs in the lower half represent
    # the broad lane markings expected at an intersection.
    cv2.rectangle(whiteMask, (10, 120), (30, 140), 255, -1)
    cv2.rectangle(whiteMask, (80, 130), (100, 150), 255, -1)
    cv2.rectangle(whiteMask, (150, 140), (170, 160), 255, -1)

    assert laneHandler.detectIntersection(whiteMask) is True




    
