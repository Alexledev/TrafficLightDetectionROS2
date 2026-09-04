import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from sensor_msgs.msg import Imu

class RoverladIMU(Node):
    def __init__(self):
        super().__init__('RoverladIMU')
        self.subscription = self.create_subscription(Imu, '/roverlad_imu', self.imuCallback, 10)
        self.velPublisher = self.create_publisher(Float32, '/roverlad_angular', 10)
        
        self.get_logger().info("-- RoverladIMU node started. --")

    def imuCallback(self, msg: Imu):
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        self.get_logger().info(f"AngVel: [{wx:.2f}, {wy:.2f}, {wz:.2f}] | LinAcc: [{ax:.2f}, {ay:.2f}, {az:.2f}]")

        # degPerSec = msg.angular_velocity.y * 180.0 / math.pi

        self.velPublisher.publish(Float32(data=wy))

def main(args=None):
    rclpy.init(args=args)

    node = RoverladIMU()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
