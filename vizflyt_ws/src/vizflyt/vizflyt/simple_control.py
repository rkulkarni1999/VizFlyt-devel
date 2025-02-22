import rclpy
from rclpy.node import Node
import time
from pymavlink import mavutil
from dronekit import connect, VehicleMode

class DroneControlNode(Node):
    """
    ROS2 Node for controlling a drone using position commands via MAVLink.
    """

    def __init__(self):
        super().__init__('drone_control_node')
        
        # Connect to the drone
        self.vehicle = self.init_vehicle()

        # Take off to target altitude
        self.takeoff(10)

        # Move to a specific position (5m North, 0m East, -10m Down)
        self.goto_position_target_local_ned(10, 0, -8)
        time.sleep(5)  # Wait for some time before landing

        # Land the drone
        self.land()

    def init_vehicle(self):
        """
        Connects to the vehicle using UDP and ensures it's ready.
        """
        connection_string = 'udp:127.0.0.1:14550'  # ArduPilot SITL

        self.get_logger().info(f'Connecting to vehicle on: {connection_string}')
        vehicle = connect(connection_string, wait_ready=True)

        # Ensure drone is in GUIDED mode
        vehicle.mode = VehicleMode("GUIDED")
        while vehicle.mode.name != "GUIDED":
            self.get_logger().info("Waiting for GUIDED mode...")
            time.sleep(1)

        # Arm the drone
        vehicle.armed = True
        while not vehicle.armed:
            self.get_logger().info("Waiting for drone to arm...")
            time.sleep(1)

        self.get_logger().info("Drone armed and ready.")
        return vehicle

    def takeoff(self, target_altitude):
        """
        Arms the drone and makes it take off to the specified altitude.
        """
        self.get_logger().info(f"Taking off to {target_altitude} meters...")
        self.vehicle.simple_takeoff(target_altitude)

        # Wait until target altitude is reached
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            self.get_logger().info(f"Altitude: {current_alt:.2f}m")

            if current_alt >= target_altitude * 0.95:
                self.get_logger().info("Target altitude reached!")
                break
            time.sleep(1)

    def goto_position_target_local_ned(self, north, east, down):
        """
        Moves the drone to a specified **NED** position.
        - **North (+) / South (-)**
        - **East (+) / West (-)**
        - **Down (-) / Up (+)**
        """
        self.get_logger().info(f"Moving to position: North={north}m, East={east}m, Down={down}m")
        
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # Frame
            0b0000111111111000,  # Type mask (only position enabled)
            north, east, down,  # x, y, z positions
            0, 0, 0,  # x, y, z velocity (not used)
            0, 0, 0,  # x, y, z acceleration (not used)
            0, 0  # yaw, yaw rate (not used)
        )

        self.vehicle.send_mavlink(msg)
        time.sleep(0.001)
        # self.vehicle.flush()

    def land(self):
        """
        Commands the drone to land.
        """
        self.get_logger().info("Landing the drone...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND":
            self.get_logger().info("Waiting for landing mode...")
            time.sleep(1)

        self.get_logger().info("Drone landed successfully!")
        self.vehicle.close()

def main():
    """
    Initializes the ROS2 node.
    """
    rclpy.init()
    node = DroneControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
