import gymnasium as gym
import mujoco
import numpy as np

from pick_place import PickPlaceEnv


class RLPickPlaceEnv(PickPlaceEnv):
    """PPO chooses hand movement; IK remains the low-level robot controller."""

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        # Actions are x/y/z hand changes plus gripper. Position changes are scaled to 2 cm.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (35,), dtype=np.float32)

    def observation(self):
        # Hand XYZ saves PPO from first having to learn forward kinematics from joint angles.
        return np.concatenate((self.data.xpos[self.hand_id].copy(), super().observation())).astype(np.float32)

    def _joint_action(self, hand_target, gripper):
        position_error = hand_target - self.data.xpos[self.hand_id]
        current = self.data.xmat[self.hand_id].reshape(3, 3)
        rotation_error = 0.5 * sum(np.cross(current[:, i], self.hand_rotation[:, i]) for i in range(3))

        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacobian_pos, jacobian_rot, self.hand_id)
        jacobian = np.vstack((jacobian_pos[:, :7], jacobian_rot[:, :7]))
        error = np.concatenate((position_error, rotation_error))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03 * np.eye(6), error)

        joint_action = np.zeros(8)
        joint_action[:7] = np.clip(delta / 0.04, -1, 1)
        joint_action[7] = gripper
        return joint_action

    def step(self, action):
        action = np.asarray(action, dtype=float)
        target = self.data.xpos[self.hand_id] + 0.02 * action[:3]
        previous_stage = self.stage
        observation, _, terminated, truncated, info = super().step(
            self._joint_action(target, action[3])
        )

        active = self.order[min(previous_stage, 1)]
        cube = self.data.xpos[self.cube_ids[active]]
        goal = self.data.site_xpos[self.goal_ids[active]]
        hand_to_cube = np.linalg.norm(self.data.xpos[self.hand_id] - cube)
        cube_to_goal = np.linalg.norm(cube - goal)

        # First reward reaching; after lifting above 43 cm, reward transport to the goal.
        reward = -hand_to_cube if cube[2] < 0.43 else 1.0 - 2.0 * cube_to_goal
        if self.stage > previous_stage:
            reward += 10.0

        info.update(
            hand_to_cube=float(hand_to_cube),
            cube_to_goal=float(cube_to_goal),
            is_success=terminated,
        )
        return observation, reward, terminated, truncated, info
