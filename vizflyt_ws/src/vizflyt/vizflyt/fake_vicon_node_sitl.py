import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from vicon_receiver.msg import Position
from vizflyt.utils import quaternion_multiply
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('fake_vicon_node_sitl')

        # Publisher to the render node
        self.publisher_ = self.create_publisher(Position, '/vicon/VizFlyt/VizFlyt', 10)

        # Reliable QoS profile for MAVROS
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # Subscriber to MAVROS local position
        self.subscription = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos_profile
        )

        # Timer to publish at a higher rate (e.g., 30 Hz)
        self.publish_rate = 30.0  # Hz
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_position)

        # Store the latest received pose
        self.latest_pose = None  

        self.get_logger().info("Fake Vicon Node from SITL Initialized")

    def pose_callback(self, msg):
        """Receives MAVROS pose message and stores it for publishing."""
        self.latest_pose = msg  # Store latest pose message

    def publish_position(self):
        """Publishes the latest received position at a fixed rate."""
        if self.latest_pose is None:
            self.get_logger().warn("No pose received yet.")
            return
        
        msg = self.latest_pose  # Use stored pose message
        vicon_msg = Position()

        # Convert NED to NWU
        vicon_msg.x_trans = msg.pose.position.x
        vicon_msg.y_trans = -msg.pose.position.y
        vicon_msg.z_trans = -msg.pose.position.z

        # Quaternion transformation
        x_rot_ned = msg.pose.orientation.x
        y_rot_ned = msg.pose.orientation.y
        z_rot_ned = msg.pose.orientation.z
        w_rot_ned = msg.pose.orientation.w 
        
        q_ned = np.array([w_rot_ned, x_rot_ned, y_rot_ned, z_rot_ned])
        q_ned2nwu = np.array([0., 1., 0., 0.])
        
        q_nwu = quaternion_multiply(q_ned2nwu, q_ned)

        vicon_msg.x_rot = q_nwu[1]
        vicon_msg.y_rot = q_nwu[2]
        vicon_msg.z_rot = q_nwu[3]
        vicon_msg.w     = q_nwu[0]

        # Publish at the higher rate
        self.publisher_.publish(vicon_msg)
    

def main():
    rclpy.init()
    node = PositionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
