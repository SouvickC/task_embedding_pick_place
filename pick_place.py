from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


class PickPlaceEnv(gym.Env):
    """Two ordered placements controlled by Panda joint-position targets."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, required_stages=2):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        # Each action advances 10 * 0.002 = 0.02 seconds of simulation.
        self.frame_skip = 10
        self.step_count = 0
        self.stage = 0
        self.order = np.array([0, 1])
        self.required_stages = required_stages

        # Seven arm target changes and one gripper command.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32)

        self.hand_id = self.model.body("hand").id
        self.cube_ids = [self.model.body("red_cube").id, self.model.body("blue_cube").id]
        self.goal_ids = [self.model.site("red_goal").id, self.model.site("blue_goal").id]
        self.cube_qpos = [self.model.jnt_qposadr[self.model.body_jntadr[i]] for i in self.cube_ids]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.order = self.np_random.permutation(2)
        self.stage = 0
        self.step_count = 0

        # Positions are meters; randomize each cube by at most 5 mm in x and y.
        starts = ((0.42, -0.13), (0.42, 0.13))
        for address, (x, y) in zip(self.cube_qpos, starts):
            self.data.qpos[address : address + 3] = (x, y, 0.385)
            self.data.qpos[address + 3 : address + 7] = (1, 0, 0, 0)
            self.data.qpos[address : address + 2] += self.np_random.uniform(-0.005, 0.005, 2)
        mujoco.mj_forward(self.model, self.data)
        self.hand_rotation = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        return self.observation(), {"order": self.order.copy()}

    def observation(self):
        cubes = np.concatenate([self.data.xpos[i].copy() for i in self.cube_ids])
        goals = np.concatenate([self.data.site_xpos[i].copy() for i in self.goal_ids])

        # Task ID says which cube goes first: [red-first, blue-first, red-second, blue-second].
        task = np.zeros(4)
        task[self.order[0]] = 1.0
        task[2 + self.order[1]] = 1.0
        stage = np.eye(2)[min(self.stage, 1)]
        return np.concatenate((self.data.qpos[:7], self.data.qvel[:7], cubes, goals, task, stage)).astype(np.float32)

    def stage_complete(self, active, distance):
        cube_height = self.data.xpos[self.cube_ids[active], 2]
        return distance < 0.08 and cube_height < 0.43

    def step(self, action):
        action = np.asarray(action, dtype=float)
        # A full action changes each joint target by at most 0.04 radians.
        self.data.ctrl[:7] = np.clip(
            self.data.ctrl[:7] + 0.04 * action[:7],
            self.model.actuator_ctrlrange[:7, 0],
            self.model.actuator_ctrlrange[:7, 1],
        )
        self.data.ctrl[7] = 255 if action[7] > 0 else 0
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.step_count += 1

        active = self.order[min(self.stage, 1)]
        distance = np.linalg.norm(self.data.xpos[self.cube_ids[active]] - self.data.site_xpos[self.goal_ids[active]])
        reward = -distance
        # Success means within 8 cm of the goal and below 43 cm, near the table.
        if self.stage < 2 and self.stage_complete(active, distance):
            self.stage += 1
            reward += 5.0

        terminated = self.stage >= self.required_stages
        truncated = self.step_count >= 1000
        if self.viewer is not None:
            self.viewer.sync()
        return self.observation(), reward, terminated, truncated, {"stage": self.stage}

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.lookat[:] = (0.50, 0.0, 0.35)
            self.viewer.cam.distance = 1.45  # Camera distance in meters.
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
