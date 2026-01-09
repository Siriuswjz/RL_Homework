from . import airsim
import gymnasium as gym
import numpy as np


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.setup_flight()
        obs, info = self.get_obs()
        return obs, info

    def render(self):
        return self.get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(-1, 1)
        
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        if self.random_start:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = section["offset"][0]
        self.target_pos = section["target"]

        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        self.target_dist_prev = np.linalg.norm(np.array([y_pos, z_pos]) - self.target_pos)

    def do_action(self, select_action):
        speed = 0.4
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0

        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([y,-z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr)*20

        self.target_dist_prev = target_dist_curr
        agent_traveled_x = np.abs(self.agent_start_pos - x)

        if target_dist_curr < 0.30:
            reward += 12
            if agent_traveled_x > 2.9:
                reward += 7
        elif target_dist_curr < 0.45:
            reward += 7

        if self.is_collision():
            reward = -100
            done = 1
        elif agent_traveled_x > 3.7:
            reward += 10
            done = 1
        elif (target_dist_curr-0.3) > (3.7-agent_traveled_x)*1.732:
            reward = -100
            done = 1

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class ContinuousTestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        super().__init__(ip_address, image_shape, env_config)
        self.random_start = False
        self.start_x = None
        self.eps_n = 0
        self.agent_traveled = []
        
    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(-1, 1)
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
        section = self.sections[0]
        self.agent_start_pos = section["offset"][0]
        self.target_pos = section["target"]
        
        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        self.target_dist_prev = np.linalg.norm(np.array([y_pos, z_pos]) - self.target_pos)
        self.start_x = self.agent_start_pos
        
    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        
        collision = self.is_collision()
        if collision:
            x = self.drone.simGetVehiclePose().position.x_val
            self.agent_traveled.append(abs(x))
            self.eps_n += 1
            
            if self.eps_n % 5 == 0:
                print("-" * 40)
                print(f"> 总episodes: {self.eps_n}")
                print(f"> 平均飞行距离: {np.mean(self.agent_traveled):.2f} m")
                print(f"> 最大穿洞数: {int(np.max(self.agent_traveled)//4)}")
                print(f"> 平均穿洞数: {int(np.mean(self.agent_traveled)//4)}")
                print("-" * 40)
            
            self._respawn()
        
        return obs, 0, False, False, info
    
    def _respawn(self):
        idx = np.random.randint(len(self.sections))
        section = self.sections[idx]
        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(section["offset"][0], y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
    def compute_reward(self):
        return 0, False


class MultiHoleEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        super().__init__(ip_address, image_shape, env_config)
        self.random_start = False
        self.holes_passed = 0
        self.start_x = None
        self.last_hole_x = None
        
    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(-1, 1)
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
        section = self.sections[0]
        self.agent_start_pos = section["offset"][0]
        self.target_pos = section["target"]
        
        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        self.target_dist_prev = np.linalg.norm(np.array([y_pos, z_pos]) - self.target_pos)
        self.holes_passed = 0
        self.start_x = self.agent_start_pos
        self.last_hole_x = self.agent_start_pos
        
    def compute_reward(self):
        reward = 0
        done = 0
        
        x, y, z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([y, -z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr) * 20
        self.target_dist_prev = target_dist_curr
        
        agent_traveled_x = np.abs(self.start_x - x)
        hole_traveled_x = np.abs(self.last_hole_x - x)
        
        if target_dist_curr < 0.30:
            reward += 12
            if hole_traveled_x > 2.9:
                reward += 7
        elif target_dist_curr < 0.45:
            reward += 7
        
        if self.is_collision():
            reward = -100
            done = 1
        elif hole_traveled_x > 3.7:
            self.holes_passed += 1
            reward += 50 + self.holes_passed * 10
            self.last_hole_x = x
            
        return reward, done
