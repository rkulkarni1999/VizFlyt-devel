import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubscriber(Node):
    """
    Subscribes to RGB and Depth images and visualizes them using OpenCV.
    """
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()
        
        # Create subscribers
        self.rgb_subscriber = self.create_subscription(Image, "/vizflyt/rgb_image", self.rgb_callback, 10)
        self.depth_subscriber = self.create_subscription(Image, "/vizflyt/depth_image", self.depth_callback, 10)

        self.get_logger().info("Image Subscriber Node Initialized")

    def rgb_callback(self, msg):
        """Receives and displays RGB images."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            cv2.imshow("RGB Image", rgb_image)  # Convert to BGR for OpenCV display
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in RGB Callback: {e}")

    def depth_callback(self, msg):
        """Receives and displays Depth images."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

            # Normalize depth image for better visualization
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)

            cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in Depth Callback: {e}")

def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
