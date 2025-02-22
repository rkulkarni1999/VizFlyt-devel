from __future__ import annotations
import os
import time

from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, get_args)

import cv2
import numpy as np
import torch

# Nerfstudio imports
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.viewer.utils import CameraState, get_camera

import time

######################
# CUSTOM VIEWER CLASS
######################
@decorate_all([check_main_thread])
class Custom_Viewer(object):
    """
    Custom Viewer class for rendering images and depth maps using NeRFStudio.
    """
    config: TrainerConfig
    pipeline: Pipeline
    
    def __init__(self, config: TrainerConfig, pipeline: Pipeline):
        super().__init__()

        self.config = config
        self.pipeline = pipeline
        self.model = self.pipeline.model
        self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=self.model.device)
        
        # image settings
        self.scale_ratio = 1
        self.max_res = 640
        self.image_height = 480
        self.image_width = 640
        self.depth_res = 640
        self.fov = 1.3089969389957472
        self.aspect_ratio = 1.774976538533895
        
        # colormap options
        self.colormap_options_rgb = ColormapOptions(colormap='default', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        self.colormap_options_depth = ColormapOptions(colormap='gray', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        
        # calculate focal length based on image height and field of view   
        pp_h = self.image_height / 2.0
        self.focal_length = pp_h / np.tan(self.fov / 2.0)
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
                
        # for washburn obstacle course {Forward Facing Camera}
        self.init_position    = np.array([1.0528e+00, -3.0682e+00, -1.1188e+00])
        self.init_orientation = np.array([[ 9.4053e-01, -3.6536e-02,  3.3775e-01],
                                          [ 3.3972e-01,  1.0115e-01, -9.3507e-01],
                                          [ -8.3267e-17,  9.9420e-01,  1.0755e-01]])
        
        # creating the directories and sub-directories outputs
        self.experiment_directory = "./ExpTest"
        os.makedirs(self.experiment_directory, exist_ok=True)
        self.create_output_directory()
        
        self.rgb_directory = os.path.join(self.output_directory, "rgb")
        self.depth_directory = os.path.join(self.output_directory, "depth")
        
        os.makedirs(self.rgb_directory, exist_ok=True)
        os.makedirs(self.depth_directory, exist_ok=True)
        
        self.image_id = 0
        self.running = False
        
        # set origin as where you start
        self.origin_pose = np.array([None,None,None,None,None,None])
        if np.all(self.origin_pose == None):
            self.origin_pose = self.get_latest_pose()
    
    #########################
    # GENERAL UTILS METHODS
    #########################       
    def create_output_directory(self):
        """
        Creating output directory for saving images
        """
        subdirs = [d for d in os.listdir(self.experiment_directory) if os.path.isdir(os.path.join(self.experiment_directory, d))]
        output_dirs = [d for d in subdirs if d.startswith('output') and d[6:].isdigit()]
        if output_dirs:
            highest_index = max([int(d[6:]) for d in output_dirs])
        else:
            highest_index = 0
        output_dir_name = f'output{highest_index + 1}'
        self.output_directory = os.path.join(self.experiment_directory, output_dir_name)
        os.makedirs(self.output_directory)
        print(f'Created new output directory: {self.output_directory}')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Converts Euler Angles to Quaternion
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qw, qx, qy, qz]
    
    def quaternion_multiply(self, quaternion1, quaternion0):
        """
        Multiplies 2 quaternions
        """
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    

    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        """
        make rotation matrix from rol, pitch, yaw
        """
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return R
    
    def get_latest_pose(self):
        try:
            response = requests.get("http://localhost:8000/get_pose")
            data = response.json()
            x = data["x"]
            y = data["y"]
            z = data["z"]
            roll = data["roll"]
            pitch = data["pitch"]
            yaw = data["yaw"]
            return np.array([x, y, z, roll, pitch, yaw])
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
    
    def convert_to_ned(self, current_pose, origin_pose):
        """
        convert from NWU to NED frame 
        """    
        x =  (current_pose[0] - origin_pose[0]) * 0.001 
        y = -(current_pose[1] - origin_pose[1]) * 0.001 
        z = -(current_pose[2] - origin_pose[2]) * 0.001
        roll = current_pose[3] - origin_pose[3]  
        pitch = -(current_pose[4] - origin_pose[4])  
        yaw = -(current_pose[5] - origin_pose[5])
        
        return np.array([x,y,z]), np.array([roll, pitch, yaw])


    #########################
    # Nerfstudio Functions
    #########################
    def get_camera_state(self, rot_mat, pos):
        """
        Gets the state of the camera
        """
        rot_adjustment = np.eye(3)
        r = self.rotation_matrix_from_euler(np.radians(0), 0, 0)
        R = rot_adjustment @ rot_mat
        R = torch.tensor(r @ R)   
        pos = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float64) / self.scale_ratio
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        camera_state = CameraState(fov=self.fov, aspect=self.aspect_ratio, c2w=c2w, camera_type=CameraType.PERSPECTIVE)
        return camera_state
    
    def _render_img(self, camera_state: CameraState):
        """
        Renders the RGB and depth images at the current camera frame
        """
        obb = None
        camera = get_camera(camera_state, self.image_height, self.image_width)
        camera = camera.to(self.model.device)
        self.model.set_background(self.background_color)
        self.model.eval()
        outputs = self.model.get_outputs_for_camera(camera, obb_box=obb)
        desired_depth_pixels = self.depth_res
        current_depth_pixels = outputs["depth"].shape[0] * outputs["depth"].shape[1]
        scale = min(desired_depth_pixels / current_depth_pixels, 1.0)
        
        return outputs
           
    def current_images(self, outputs):
        """
        Returns RGB and Depth Image Outputs
        """
        rgb = outputs['rgb']
        depth = outputs["depth"]
        rgb_image = colormaps.apply_colormap(image=rgb, colormap_options=self.colormap_options_rgb)
        rgb_image = (rgb_image * 255).type(torch.uint8)
        rgb_image = rgb_image.cpu().numpy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = colormaps.apply_colormap(image=depth, colormap_options=self.colormap_options_depth)
        depth_image = (depth_image * 255).type(torch.uint8)
        depth_image = depth_image.cpu().numpy()
        
        return rgb_image, depth_image

    #######################
    # EXECUTION FUNCTIONS
    #######################        
    def save_images(self):
        
        while self.running:
            
            current_pose = self.get_latest_pose()
            current_local_position_ned, current_local_orientation_ned = self.convert_to_ned(current_pose, self.origin_pose) 
            
            # Computing position update within splat for drone control {NOTE: This is the current position the drone. Called update because we'll add it to the Origin Position}
            position_update_cam = self.init_orientation @ np.array([[current_local_position_ned[1]],
                                                                    [-current_local_position_ned[2]],
                                                                    [-current_local_position_ned[0]]])
            
            # Updating the position in the splat for drone control loop
            current_position_cam = [self.init_position[0] + position_update_cam[0, 0],
                                    self.init_position[1] + position_update_cam[1, 0],
                                    self.init_position[2] + position_update_cam[2, 0]]
            
            # Updated Orientation in the nerfstudio cam frame
            r_cam = self.rotation_matrix_from_euler(current_local_orientation_ned[1],
                                                    -current_local_orientation_ned[2],
                                                    -current_local_orientation_ned[0])
            
            # Current Orientation with respect to the origin frame in nerfstudio
            current_orientation_cam = self.init_orientation @ r_cam
            
            # Getting camera state, rendering RGB and depth images at that camera state 
            camera_state = self.get_camera_state(current_orientation_cam, current_position_cam)
            outputs = self._render_img(camera_state)
            current_rgb, current_depth = self.current_images(outputs)
            
            # Saving images locally
            cv2.imwrite(os.path.join(self.rgb_directory, f"{self.image_id:04d}.png"), current_rgb)
            cv2.imwrite(os.path.join(self.depth_directory, f"{self.image_id:04d}.png"), current_depth)
            
            # Updating index for next iteration of the loop
            self.image_id += 1
            
            # Loop rate of saving images
            time.sleep(0.01)

    

    def execute(self):
        """ Starts the image capturing process"""
        
        self.running = True
        
        try:
            self.save_images()
        
        except KeyboardInterrupt:
            print("Interrupted! Stopping image capture...")
            self.running = False