"""
Setup:
pip install mujoco numpy
"""
from typing import Union, Tuple
import numpy as np
import rewards
import mujoco
import mujoco.viewer

class InvertedPendulumEnv ():
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """
    def __init__(
        self, reward_mode = 4, T_max = 1000
    ):
        
        self.reward_mode = reward_mode
        print(reward_mode)

        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)

        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)

        # for modifying the mass
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cpole")
        self.body_id = self.model.geom_bodyid[geom_id]
        self.mass = self.model.body_mass[self.body_id]
        
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.ball_pos = np.zeros(2)

        self.max_torque = 3.0  # maximal torque for control
        self.timestep = 0.0 
        self.T_max = T_max   # length of 1 episode

        self.set_dt(0.01)
        self.reset_model()

    
    def step(self, a):
        
        ob = self.obs()
    
        control_action = np.clip(a , -1.0, 1.0) * self.max_torque

        self.data.ctrl = control_action 

        mujoco.mj_step(self.model, self.data)
        
        self.viewer.sync()

        self.step_time +=1

        if self.reward_mode == 0 :
            r = rewards.compute_simplest_reward(ob)
        elif self.reward_mode == 1 :
            r = rewards.compute_exponential_reward(ob)
        elif self.reward_mode == 2 :
            r = rewards.compute_simplest_tolerance_reward(ob)
        elif self.reward_mode == 3 :
            r = rewards.compute_tolerance_reward(ob, control_action)
        elif self.reward_mode == 4 :
            r = rewards.compute_tolerance_reward_xy(ob, control_action)
        else: 
            raise NotImplementedError("Reward mode is between 0 and 4 only")
        
        terminated = bool(not np.isfinite(ob).all())
    
        if self.step_time == self.T_max - 1:
            terminated = True

        return ob, r, terminated

    def obs(self):
        
        cos_theta = np.cos(self.data.qpos[1])
        sin_theta = np.sin(self.data.qpos[1])

        ball_pos = self.ball_pos

        # observation = { x, cos_theta, sin_theta, x_vel, theta_vel, ball_x, ball_y, mass}

        return np.concatenate([[self.data.qpos[0]], [cos_theta], [sin_theta], self.data.qvel, ball_pos, [self.mass]]).ravel()

    def reset_model(self):
        self.data.qpos = self.init_qpos 
        self.data.qvel = self.init_qvel 
        self.data.qpos[1] = 3.14  # Set the pole to be facing down
        
        #set ball position
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        self.ball_pos = np.array([target_pos[0], target_pos[2]])
        self.draw_ball(target_pos, radius=0.05)

        #set mass from 3.0 to 7.0  (reference is ~5)
        self.model.body_mass[self.body_id] = 5.0+(np.random.rand()-0.5)*4.0
        self.mass  = self.model.body_mass[self.body_id]

        self.step_time = 0

        #print("mass ", self.model.body_mass[self.body_id])

        return self.obs()


    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position, color=[1, 0, 0, 1], radius=0.01):
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.array(position),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self):
        return self.data.time


