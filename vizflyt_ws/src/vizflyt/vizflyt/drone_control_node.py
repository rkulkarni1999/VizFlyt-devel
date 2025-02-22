import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time
from pymavlink import mavutil
from dronekit import connect, VehicleMode
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy

class DroneController(Node):
    """
    ROS2-based Autonomous Drone Controller.
    """

    def __init__(self):
        super().__init__('drone_control_node')
        
        # Connect to the drone via MAVLink
        self.vehicle = self.init_vehicle()
        
        # QoS Profile to ensure MAVROS compatibility
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Avoids reliability mismatch
            depth=10
        )

        # Subscribers
        self.pose_subscription = self.create_subscription(
            PoseStamped, '/vizflyt/drone_pose_NED', self.pose_callback, qos_profile
        )
        self.rgb_subscription = self.create_subscription(
            Image, '/vizflyt/rgb_image', self.rgb_callback, qos_profile
        )
        self.depth_subscription = self.create_subscription(
            Image, '/vizflyt/depth_image', self.depth_callback, qos_profile
        )

        # Image processor
        self.bridge = CvBridge()
        
        # State variables
        self.current_rgb = None
        self.current_depth = None
        self.current_pose = None
        self.latest_timestamp = None
        
        # Control loop timer (10 Hz)
        self.control_timer = self.create_timer(20.0, self.control_loop)

        self.get_logger().info("Drone Control Node Initialized")

    ######################
    # Dronekit Functions
    ######################
    def init_vehicle(self):
        """
        Connects to the vehicle using UDP connection.
        """
        connection_string = 'udp:127.0.0.1:14550'  # Connection to ArduPilot SITL
        
        self.get_logger().info(f'Connecting to vehicle on: {connection_string}')
        vehicle = connect(connection_string, wait_ready=True, rate=90)        
        print("Connected. Initial mode: %s" % vehicle.mode.name)
        
        return vehicle
    
    def takeoff(self, altitude=10):
        """
        Arms the drone and takes off to a specified altitude.
        """
        self.get_logger().info("Arming motors...")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            self.get_logger().info("Waiting for drone to arm...")
            time.sleep(1)

        self.get_logger().info(f"Taking off to {altitude} meters...")
        self.vehicle.simple_takeoff(altitude)

        # Wait until the drone reaches the target altitude
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            self.get_logger().info(f"Altitude: {current_alt:.2f} meters")
            if current_alt >= altitude * 0.95:
                self.get_logger().info("Target altitude reached!")
                break
            time.sleep(1)


    def send_ned_velocity(self, velocity_x, velocity_y, velocity_z):
        """
        Sends velocity commands to the drone.
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms (not used)
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
            0b0000111111000111,  # type_mask (only speeds enabled)
            0, 0, 0,  # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
            0, 0, 0,  # x, y, z acceleration (ignored)
            0, 0  # yaw, yaw_rate (ignored)
        )

        self.vehicle.send_mavlink(msg)
        time.sleep(0.01)  # Control loop rate

    def goto_position_target_local_ned(self, north, east, down):
        """ Moves the drone to a specified NED position. """

        if self.vehicle.mode.name != "GUIDED":
            self.vehicle.mode = VehicleMode("GUIDED")
            while self.vehicle.mode.name != "GUIDED":
                self.get_logger().info("Waiting for GUIDED mode...")
                time.sleep(1)

        self.get_logger().info(f"Moving to position: North={north}, East={east}, Down={down}")

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,  # Position enabled
            north, east, down, 0, 0, 0, 0, 0, 0, 0, 0
        )

        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()  # **Ensures command execution**



    def condition_yaw(self, heading, relative=False):
        """
        Commands the drone to change its yaw.
        """
        is_relative = 1 if relative else 0

        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
            0,  # confirmation
            heading,  # param 1: yaw in degrees
            0,  # param 2: yaw speed (deg/s)
            0,  # param 3: direction (-1 ccw, 1 cw)
            is_relative,  # param 4: relative offset (1), absolute angle (0)
            0, 0, 0  # params 5-7 not used
        )

        self.vehicle.send_mavlink(msg)
        time.sleep(0.1)

    def return_to_land(self):
        """
        Commands the drone to land.
        """
        if self.vehicle.mode.name != "LAND":
            print("Changing mode to LAND")
            self.vehicle.mode = VehicleMode("LAND")
            print("Mode changed to LAND")

    def get_current_ned_vel(self):
        """
        Gets velocity feedback from the flight controller.
        """
        return self.vehicle.velocity

    #################
    # ROS2 Callbacks
    #################
    def rgb_callback(self, msg):
        """Stores the latest RGB image."""
        self.current_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_timestamp = msg.header.stamp

    def depth_callback(self, msg):
        """Stores the latest Depth image."""
        self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def pose_callback(self, msg):
        """Stores the latest pose."""
        self.current_pose = msg.pose

    ############################
    # Drone Command Computation
    ############################
    def get_velocities(self, rgb_image, depth_image):
        """
        Computes velocities (vx, vy, vz, yaw) from RGB and Depth images.
        """
        vx, vy, vz = 0.5, 0.0, 0.0  # Placeholder values
        yaw = 0.0
        return vx, vy, vz, yaw

    def get_waypoints(self, rgb_image, depth_image):
        """
        Computes waypoints (px, py, pz, yaw) from RGB and Depth images.
        """
        px, py, pz = 2.0, 1.0, -1.5  # Placeholder values
        yaw = 0.0
        return px, py, pz, yaw

    def control_loop(self):
        """Control loop to move the drone and verify target position reached."""

        self.takeoff()
        
        if self.current_rgb is None or self.current_depth is None or self.current_pose is None:
            self.get_logger().info("Waiting for synchronized data...")
            return

        # Get new waypoint
        px, py, pz, yaw = self.get_waypoints(self.current_rgb, self.current_depth)

        # Ensure the drone takes off before moving
        if self.vehicle.location.global_relative_frame.alt < 9.5:
            self.get_logger().info("Waiting for takeoff to complete...")
            return

        # Send movement command
        self.get_logger().info(f"Sending movement command to: px={px}, py={py}, pz={pz}, yaw={yaw}")
        self.goto_position_target_local_ned(px, py, pz)
        self.condition_yaw(yaw, relative=False)

        # Wait for drone to reach target
        while not self.has_reached_target(px, py, pz):
            self.get_logger().info("Moving to target...")
            time.sleep(1)

        self.get_logger().info("Target position reached!")

        # Stop the control loop **only after reaching target**
        self.control_timer.cancel()


        
    def has_reached_target(self, target_x, target_y, target_z, tolerance=0.2):
        """Checks if the drone has reached the target position."""
        if self.current_pose is None:
            return False

        # Convert NWU to NED
        drone_x = self.current_pose.position.x
        drone_y = -self.current_pose.position.y
        drone_z = -self.current_pose.position.z  # Convert altitude sign

        dx = abs(drone_x - target_x)
        dy = abs(drone_y - target_y)
        dz = abs(drone_z - target_z)

        return dx < tolerance and dy < tolerance and dz < tolerance



def main():
    """ Main function """
    rclpy.init()
    node = DroneController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
