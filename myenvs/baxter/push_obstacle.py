import os
from gym import utils
from myenvs.baxter import baxter_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('baxter', 'push_obstacle.xml')


class BaxterPushObstacleEnv(baxter_env.BaxterEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            # 'robot0:slide0': 0.0, #-----baxter
            # 'robot0:slide1': 0.0, #-----baxter
            # 'robot0:slide2': 0.0, #-----baxter
            'robot0:right_s0': 0.07669903930664063, #-----baxter
            'robot0:right_s1': -0.9660244000671387, #-----baxter
            'robot0:right_w0': -0.6438884349792481, #-----baxter
            'robot0:right_w1':  1.0074418812927246, #-----baxter
            'robot0:right_w2':  0.483203947631836, #-----baxter
            'robot0:right_e0':  1.1481846184204103, #-----baxter
            'robot0:right_e1':  1.9232284106140138, #-----baxter
            'robot0:left_w0': 0.6477233869445801, #-----baxter
            'robot0:left_w1': 1.007825376489258, #-----baxter
            'robot0:left_w2': -0.48282045243530275, #-----baxter
            'robot0:left_e0': -1.1504855895996096, #-----baxter
            'robot0:left_e1': 1.9232284106140138, #-----baxter
            'robot0:left_s0': -0.07823302009277344, #-----baxter
            'robot0:left_s1': -0.9675583808532715, #-----baxter
            'object0:joint': [1.65, 0.259027, -0.15, 1., 0., 0., 0.], #[1.65, 0.53, -0.15, 1., 0., 0., 0.]
        }
        baxter_env.BaxterEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, single_goal=False)
        utils.EzPickle.__init__(self)


    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.target_offset + self.np_random.uniform(-self.target_range,
                                                                                               self.target_range,
                                                                                               size=3)
            goal[1] = self.np_random.uniform(low=-0.1, high=0.2)
            goal[0] = self.np_random.uniform(low=0.45, high=0.53)
            goal[2] = -0.04
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]

            object_xpos[1] = self.np_random.uniform(low=0.05, high=0.3)
            object_xpos[0] =self.np_random.uniform(low=0.7, high=0.8)

        else:
            object_xpos = self.initial_gripper_xpos[:2] + np.asarray([self.obj_range * 0.9, self.obj_range / 2])

            # Set the position of box. (two slide joints)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        return True