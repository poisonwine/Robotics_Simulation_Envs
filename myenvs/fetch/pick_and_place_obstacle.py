import os
from gym import utils
from gym.envs.robotics import fetch_env, rotations
import gym.envs.robotics.utils as robot_utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'pick_and_place_obstacle.xml')


class FetchPnPObstacle_v1(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', penaltize_height=False, random_box=True,
                 random_ratio=1.0, random_gripper=False):
        XML_PATH = MODEL_XML_PATH
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.2, 0.53, 0.4, 1., 0., 0., 0.],

            #'object1:joint': [1.35, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = 1
        self.penaltize_height = penaltize_height
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.random_gripper = random_gripper
        fetch_env.FetchEnv.__init__(
            self, XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        #self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('object1')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp = object_velp - grip_velp

        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # stick_pos = stick_rot = stick_velp = stick_velr = stick_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
            # achieved_goal = self.sim.data.get_site_xpos('object0').copy()
            # achieved_goal = self.sim.data.get_site_xpos('object1').copy()
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])  # dim 40
        # print('grip_pos', grip_pos.shape) # (3,)
        # print('object_pos', object_pos.shape) # (6,)
        # print('object_rel_pos', object_rel_pos.shape) # (6,)
        # print('object_rot', object_rot.shape) # (6,)
        # print('gripper_state', gripper_state.shape) # (2,)
        # print('object_velp', object_velp.shape) # (6,)
        # print('object_velr', object_velr.shape) # (6,)
        # print('grip_velp', grip_velp.shape) # (3,)
        # print('gripper_vel', gripper_vel.shape) # (2,)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # TODO: randomize mocap_pos
        if self.random_gripper:
            mocap_pos = np.concatenate([self.np_random.uniform([1.19, 0.6], [1.49, 0.9]), [0.355]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        # Randomize start position of object.
        if self.has_object:
            if self.random_box and self.np_random.uniform() < self.random_ratio:
                object_xpos = self.initial_gripper_xpos[:2]
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1
                       or (object_xpos[1] - self.pos_wall[1]) > self.size_wall[1] + self.size_object[1])\
                       or (self.pos_wall[1]-0.06 < object_xpos[1]):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)

            else:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])

            # Set the position of box. (two slide joints)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_wall'):
            self.size_wall = self.sim.model.geom_size[self.sim.model.geom_name2id('wall0')]
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'pos_wall'):
            self.pos_wall = self.sim.model.geom_pos[self.sim.model.geom_name2id('wall0')]
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[1] = self.np_random.uniform(low=0.85+self.size_object[0], high=1.1-self.size_object[0])
            goal[0] = self.np_random.uniform(low=1.075, high=1.45)
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.35)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()