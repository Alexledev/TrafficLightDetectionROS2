import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int16, Bool
from .TFL_handler import TFLightInferenceRunner
# from .streetSignModelHandler import StreetSignInferenceRunner
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import time
import numpy as np
import cv2

class RoverladCV(Node):
    def __init__(self):
        super().__init__('RoverLadCV')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.cameraSubscription = self.create_subscription(Image, '/roverlad_camera_rgb', self.callback, qos)
        self.depthSubscription = self.create_subscription(Image, '/roverlad_camera_depth', self.depthCallback, qos)
        
        self.cvPublisher = self.create_publisher(Image, '/roverlad_camera_cv', qos)
        self.tlPublisher = self.create_publisher(String, '/roverlad_tflight', 10)
        self.inRangePub  = self.create_publisher(Bool, '/roverlad_tflight_inrange', 10)

        self.tflModel  = TFLightInferenceRunner("/home/lehongnhatminh/TFL_ws/models/model.onnx")

        self.TLlabels = {
            0: ("Red",    (0, 0,   105)),
            1: ("Yellow", (0, 105, 105)),
            2: ("Green",  (0, 105, 0))
        }

        self.priorityMap = {
            0: 1,  # Green
            1: 2,  # Yellow
            2: 3,  # Red
        }

        # self.signMap = {0: "S15", 1: "S20", 2: "S30", 3: "S40"}

        self.TFLightAreaThresh = 200
        self.TFLightDepthThresh = 7.5

        # self.StreetSignAreaThresh = 1600

        self.bestBox = (0, 0, 0, 0)
        self.poi = (0, 0)
        self.depth = 0

        self.latestFrame = None

        self.create_timer(0.01, self.processLatestFrame)
        self.startTime = time.time()

        self.get_logger().info("-- RoverDudeCV node started. --")

    def depthCallback(self, msg):
        depthImg = self.imgMsgToCV(msg)
        u, v = self.poi
        if u >= depthImg.shape[1] or v >= depthImg.shape[0]:
            return
        depthVal = depthImg[v, u]

        if self.bestBox is None:
            return
                        
        if np.isnan(depthVal) or np.isinf(depthVal):
            return

        self.depth = depthVal
        # print(f"Depth at {self.poi}: {self.depth:.3f} m")

    def getMedianDepth(self, depthImg, box):
        x1, y1, x2, y2 = box

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        w = (x2 - x1) // 4
        h = (y2 - y1) // 4

        x1c = max(0, cx - w)
        x2c = min(depthImg.shape[1], cx + w)
        y1c = max(0, cy - h)
        y2c = min(depthImg.shape[0], cy + h)

        roi = depthImg[y1c:y2c, x1c:x2c]

        valid = roi[np.isfinite(roi)]
        if valid.size == 0:
            return None

        return np.median(valid)

    def callback(self, msg):
        # print(f"Received a new image at {time.time():.2f}")
        self.latestFrame = msg
        receiveTime = time.time()

        imageTime = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        print(
            f"Image ROS time: {imageTime:.6f} | "
            f"Received wall time: {receiveTime:.6f}"
        )

    def processLatestFrame(self):

        if self.latestFrame == None:
            return

        cvImg = self.imgMsgToCV(self.latestFrame)
        if cvImg is None or cvImg.size == 0:
            print("Invalid Img received")
            return
        
        cvImg = self.runTFLight(cvImg)

        endTime = time.time()

        imgOut = self.imgCVToMsg(cvImg)
        self.cvPublisher.publish(imgOut)

        executionTime = endTime - self.startTime
        self.startTime = endTime
        # print(f"Execution time between frames: {executionTime:.6f} seconds")


    def runTFLight(self, cvImg):
        boxesNP, boxesPX, scoresNP, labelsNP = self.tflModel.run(cvImg, conf=0.95, debug=False)

        bestIdx = max(range(len(labelsNP)), key=lambda i: self.priorityMap[labelsNP[i]], default=None)

        bestLabel = labelsNP[bestIdx] if bestIdx is not None else None
        self.bestBox = boxesPX[bestIdx] if bestIdx is not None else None

        tlMsg = String()
        tlMsg.data = self.TLlabels[bestLabel][0] if bestLabel is not None else "None"
        self.tlPublisher.publish(tlMsg)

        if len(boxesPX) > 0:
            x1, y1, x2, y2 = boxesPX[0]
            self.poi = ((x1 + x2) // 2, (y1 + y2) // 2)

        msg = Bool()
        msg.data = bool((self.computeArea(self.bestBox) >= self.TFLightAreaThresh) and (self.depth <= self.TFLightDepthThresh))
        self.inRangePub.publish(msg)

        # print("Box size:", self.computeArea(self.bestBox))

        imgCopy = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
        self.drawLabels(imgCopy, boxesPX, scoresNP, labelsNP, self.bestBox)
        # cvImg = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)

        return cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)


    def drawLabels(self, imgCopy, boxesPX, scoresNP, labelsNP, bestBox):
        # BGR Color         

        for i in range(len(boxesPX)):
            x1, y1, x2, y2 = boxesPX[i]
            score = scoresNP[i]
            label = labelsNP[i]

            area = self.computeArea(boxesPX[i])

            txt, color = self.TLlabels[label]

            centerPoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            cv2.rectangle(imgCopy, (x1, y1), (x2, y2), color, 2)
            cv2.circle(imgCopy, centerPoint, radius=2, color=(0, 0, 0), thickness=-1)
                
            cv2.putText(imgCopy, f"{txt}: {area}px", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(c * 2 for c in color), 1)
            cv2.putText(imgCopy, f"{score:.3f}", (x1 - 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(c * 2 for c in color), 1)

        if (bestBox is not None) and self.computeArea(bestBox) >= self.TFLightAreaThresh:
            x1, y1, x2, y2 = bestBox
            cv2.rectangle(imgCopy, (x1, y1), (x2, y2), (255, 255, 255), 3)

        text = (f"Traffic Light distance: {self.depth:.2f}m" if len(boxesPX) > 0 else "No Traffic Lights found yet")
        color = (0, 255, 0) if len(boxesPX) > 0 else (0, 0, 255)
        textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        textX = imgCopy.shape[1] - textSize[0] - 8
        cv2.putText(imgCopy, text, (textX, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

    def computeArea(self, boxPX):
        if boxPX is None:
            return 0

        x1, y1, x2, y2 = boxPX
        return abs(x2 - x1) * abs(y2 - y1)


    def imgMsgToCV(self, imageMsg):
        if imageMsg.encoding == '32FC1':
            img = np.frombuffer(imageMsg.data, dtype=np.float32)
            img = img.reshape((imageMsg.height, imageMsg.width))
            return img

        if imageMsg.encoding == '16UC1':
            img = np.frombuffer(imageMsg.data, dtype=np.uint16)
            img = img.reshape((imageMsg.height, imageMsg.width))
            return img

        if imageMsg.encoding == 'rgb8':
            img = np.frombuffer(imageMsg.data, dtype=np.uint8)
            img = img.reshape((imageMsg.height, imageMsg.width, 3))
            return img

        if imageMsg.encoding == 'bgr8':
            img = np.frombuffer(imageMsg.data, dtype=np.uint8)
            img = img.reshape((imageMsg.height, imageMsg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        if imageMsg.encoding == 'mono8':
            img = np.frombuffer(imageMsg.data, dtype=np.uint8)
            img = img.reshape((imageMsg.height, imageMsg.width))
            return img

        self.get_logger().warn(f"Unsupported encoding: {imageMsg.encoding}")
        return None
    
    def imgCVToMsg(self, img):
        msg = Image()

        msg.height = img.shape[0]
        msg.width = img.shape[1]

        if len(img.shape) == 3:
            msg.encoding = 'rgb8'   # OpenCV default
            msg.step = img.shape[1] * 3
        else:
            msg.encoding = 'mono8'
            msg.step = img.shape[1]

        msg.data = img.tobytes()
        msg.is_bigendian = False

        return msg

def main(args=None):
    rclpy.init(args=args)

    node = RoverladCV()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()