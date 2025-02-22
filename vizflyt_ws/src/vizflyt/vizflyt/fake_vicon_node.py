import rclpy
import math
from rclpy.node import Node
from vicon_receiver.msg import Position

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('fake_vicon_node')

        self.publisher_ = self.create_publisher(Position, '/vicon/VizFlyt/VizFlyt', 10)

        self.radius = 200.0 
        self.height = 100.0 
        self.angular_speed = 0.5 
        self.timer_period = 0.005 
        self.time_elapsed = 0.0  

        self.qx, self.qy, self.qz, self.qw = 0.0, 0.0, 0.0, 1.0  

        self.timer = self.create_timer(self.timer_period, self.publish_trajectory)
        
        self.get_logger().info("Fake Vicon Node Initialized")

    def publish_trajectory(self):
        """Generate a circular trajectory and publish position messages in NWU frame."""
        msg = Position()

        msg.x_trans = self.radius * math.cos(self.angular_speed * self.time_elapsed)  
        msg.y_trans = self.radius * math.sin(self.angular_speed * self.time_elapsed)  
        msg.z_trans = self.height  

        msg.x_rot = self.qx
        msg.y_rot = self.qy
        msg.z_rot = self.qz
        msg.w = self.qw

        # msg.x_trans = 0.0  
        # msg.y_trans = 0.0  
        # msg.z_trans = 0.0

        # msg.x_rot = 0.0
        # msg.y_rot = 0.0
        # msg.z_rot = 0.0
        # msg.w = 1.0

        self.publisher_.publish(msg)

        self.time_elapsed += self.timer_period

def main():

    rclpy.init()
    node = PositionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
