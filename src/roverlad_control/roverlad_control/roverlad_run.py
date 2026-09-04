import cv2
import math
import numpy as np
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from .roverlad_lanedetect import LaneHandler
from .roverlad_helper import RoverladPIDDrive
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RoverladRun(Node):
    def __init__(self):
        super().__init__('RoverladRun')
 
        self.subscription = self.create_subscription(Float32, '/roverlad_angular', self.imuCallback, 10)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        # self.roverladcvSub = self.create_subscription(Image, '/roverlad_camera_rgb', self.cameraCallback, qos)
        self.roverladcvSub = self.create_subscription(Image, '/roverlad_camera_cv', self.cameraCallback, qos)
        
        self.tflightSubscription = self.create_subscription(String, '/roverlad_tflight', self.tflightCallback, 10)
        self.tflightInRangeSubscription = self.create_subscription(Bool, '/roverlad_tflight_inrange', self.tflightInRangeCallback, 10)
        
        self.cvPublisher = self.create_publisher(Image, '/roverlad_lane_cv', qos)        
        self.cmdVel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.fullLinearSpeed = 0.15
        self.slowLinearSpeed = 0.04
        self.linearVel = self.fullLinearSpeed #self.fullLinearSpeed
        self.angularVel = 0.5

        self.laneHandler = LaneHandler()

        self.currAngle = 0.0
        self.imuAngle = 0.0
        self.tflightState = 'None'
        self.tflightInRange = False

        self.maxAngular = 1.5

        self.pidDriver = RoverladPIDDrive(self.linearVel, self.angularVel)
        
        self.get_logger().info("-- RoverladRun node started. --")


    def imuCallback(self, msg):
        self.imuAngle = -1 * msg.data

    def tflightCallback(self, msg):
        self.tflightState = msg.data.strip()
        # print("Current Traffic Light state:", self.tflightState)

    def tflightInRangeCallback(self, msg):
        self.tflightInRange = msg.data
        # print("Traffic Light in range?:", self.tflightState)
        

    def cameraCallback(self, msg):
        # self.get_logger().warn("> Got a new Image")

        startTime = time.time()

        cvImg = self.imgMsgToCV(msg)

        if cvImg is None:
            return
        
        imgCopy = cv2.cvtColor(cvImg.copy(), cv2.COLOR_RGB2BGR)

        outImg, angleDeg = self.processFrame(imgCopy)

        
        # self.get_logger().info(">> Processing...")

        img = self.imgCVToMsg(outImg)
        self.cvPublisher.publish(img)

        self.updateCurrentAngle(angleDeg)

        self.handleTFLight()

        self.controlLoop()

        self.pidDriver.setLinearAngularVel(self.linearVel, self.angularVel)

        endTime = time.time()
        
        executionTime = endTime - startTime
        print(f"Execution time: {executionTime:.6f} seconds")

        # self.get_logger().info(">>> Processed Image")
        
    def handleTFLight(self):
        
        if not self.tflightInRange:
            self.get_logger().info("No traffic light found yet...")
            self.linearVel = self.fullLinearSpeed
            return

        if self.tflightState == "Red":
            self.get_logger().info("Red light, stopping")
            self.linearVel = 0
        elif self.tflightState == "Yellow":
            self.get_logger().info("Yellow light, slowing down")
            self.linearVel = self.slowLinearSpeed
        else:
            self.get_logger().info("Green light, moving on")
            self.linearVel = self.fullLinearSpeed
            

    def updateCurrentAngle(self, angleDeg):
        angleDiff = angleDeg - self.imuAngle
        angleDiff = (angleDiff + 180) % 360 - 180
        self.currAngle = angleDiff

    def processFrame(self, imgCopy):
        outImg, angleDeg = self.laneHandler.run(imgCopy)
        self.drawIMULine(outImg, self.imuAngle)
        cv2.putText(outImg, f"Current Speed: {self.linearVel:.2f}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)
        return outImg,angleDeg

    def drawIMULine(self, img, angleDeg):
        h, w, _ = img.shape

        x0 = w // 2
        y0 = h - 1

        lineLen = 100
        x1 = int(x0 + lineLen * math.sin(angleDeg))
        y1 = int(y0 - lineLen * math.cos(angleDeg))

        # Draw line
        cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    def clamp(self, value, minVal, maxVal):
        return max(min(value, maxVal), minVal)

    def controlLoop(self):
        twist = Twist()

        linear, angular = self.pidDriver.PIDCalc(self.currAngle)

        twist.linear.x = float(linear)
        twist.angular.z = -float(angular)

        self.cmdVel.publish(twist)
    
    def getLinearSpeed(self, error):
        if abs(error) > 10:
            return self.linearVel * 0.3

        if abs(error) < 5:
            return self.linearVel * 1.5

        return self.linearVel

    def imgMsgToCV(self, imageMsg):
        img = np.frombuffer(imageMsg.data, dtype=np.uint8)

        if imageMsg.encoding == 'rgb8':
            img = img.reshape((imageMsg.height, imageMsg.width, 3))
        elif imageMsg.encoding == 'bgr8':
            img = img.reshape((imageMsg.height, imageMsg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif imageMsg.encoding == 'mono8':
            img = img.reshape((imageMsg.height, imageMsg.width))
        else:
            self.get_logger().warn(f"Unsupported encoding: {imageMsg.encoding}")
            return None
        
        return img
    
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

    node = RoverladRun()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()