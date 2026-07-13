import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from rl_env import RLPickPlaceEnv


class RLEnvironmentTest(unittest.TestCase):
    @staticmethod
    def _move_hand(env, target, gripper_open, seconds=1.2):
        completion_info = None
        for _ in range(int(seconds / (env.model.opt.timestep * env.frame_skip))):
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(
                (target - env.data.xpos[env.hand_id]) / 0.02,
                -1.0,
                1.0,
            )
            action[3] = 1.0 if gripper_open else -1.0
            _, _, terminated, truncated, info = env.step(action)
            if info["stage_completed"]:
                completion_info = info
            if terminated or truncated:
                break
        return completion_info

    def test_rl_action_and_one_hot_task(self):
        env = RLPickPlaceEnv()
        observation, _ = env.reset(seed=3)
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(observation[-6:].tolist(), [0, 1, 1, 0, 1, 0])

        next_observation, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertEqual(next_observation.shape, (39,))
        self.assertEqual(next_observation[4:7].tolist(), [0, 0, 0])
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("grasp_to_cube", info)
        self.assertIn("grasped", info)
        self.assertIn("first_box_success", info)
        self.assertIn("full_task_success", info)
        self.assertIn("first_box_grasped", info)
        self.assertIn("first_box_lifted", info)
        self.assertIn("first_box_released", info)
        self.assertFalse(info["first_box_success"])
        self.assertFalse(info["full_task_success"])
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(next_observation.shape, (39,))
        env.close()

    def test_milestone_flags_are_observable(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        active = env.order[0]
        env.grasp_held[active] = True
        env.has_lifted[active] = True
        env.has_released[active] = False

        observation = env.observation()

        self.assertEqual(observation.shape, (39,))
        self.assertEqual(observation[4:7].tolist(), [1, 1, 0])
        env.close()

    def test_pre_lift_action_reward_prefers_closed_and_upward(self):
        upward_closed = np.array([0.0, 0.0, 1.0, -1.0])
        downward_open = np.array([0.0, 0.0, -1.0, 1.0])

        self.assertGreater(
            RLPickPlaceEnv._pre_lift_action_reward(upward_closed),
            RLPickPlaceEnv._pre_lift_action_reward(downward_open),
        )

    def test_cube_cannot_be_lifted_without_a_grasp(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        active = env.order[0]
        address = env.cube_qpos[active]

        self.assertFalse(env.stage_complete(active, distance=0.0))
        env.data.qpos[address + 2] = 0.45
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env.stage_complete(active, distance=0.0))
        self.assertFalse(env.has_lifted[active])
        env.close()

    def test_training_stage_count_validation(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        env.set_required_stages(1)
        self.assertEqual(env.required_stages, 1)
        env.set_required_stages(2)
        self.assertEqual(env.required_stages, 2)
        with self.assertRaises(ValueError):
            env.set_required_stages(3)
        env.set_training_phase("lift")
        self.assertEqual(env.training_phase, "lift")
        self.assertEqual(env.required_stages, 1)
        env.set_training_phase("place_two")
        self.assertEqual(env.required_stages, 2)
        with self.assertRaises(ValueError):
            env.set_training_phase("grasp")
        env.close()

    def test_closed_fingers_pushing_cube_do_not_count_as_grasp(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        active = env.order[0]

        with (
            patch.object(env, "_gripper_opening", return_value=0.005),
            patch.object(
                env,
                "_grasp_point",
                return_value=env.data.xpos[env.cube_ids[active]].copy(),
            ),
        ):
            self.assertFalse(env._cube_grasped(active))
        env.close()

    def test_grasp_requires_centered_fingertip_contact(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        active = env.order[0]
        off_center = (
            env.data.xpos[env.cube_ids[active]].copy()
            + np.array([0.04, 0.0, 0.0])
        )

        with (
            patch.object(env, "_gripper_opening", return_value=0.05),
            patch.object(env, "_grasp_point", return_value=off_center),
        ):
            self.assertFalse(env._cube_grasped(active))
        env.close()

    def test_grasp_requires_stable_contact_and_tolerates_brief_flicker(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        active = env.order[0]
        cube_position = env.data.xpos[env.cube_ids[active]].copy()

        with (
            patch.object(env, "_cube_grasped", return_value=True),
            patch.object(env, "_grasp_point", return_value=cube_position),
            patch.object(env, "_gripper_opening", return_value=0.04),
        ):
            for _ in range(4):
                _, held = env._update_grasp_state(active)
                self.assertFalse(held)
            _, held = env._update_grasp_state(active)
            self.assertTrue(held)

        with (
            patch.object(env, "_cube_grasped", return_value=False),
            patch.object(env, "_grasp_point", return_value=cube_position),
            patch.object(env, "_gripper_opening", return_value=0.04),
        ):
            for _ in range(3):
                _, held = env._update_grasp_state(active)
                self.assertTrue(held)
            _, held = env._update_grasp_state(active)
            self.assertFalse(held)
        env.close()

    def test_rl_script_requires_release_and_settling(self):
        env = RLPickPlaceEnv(required_stages=1)
        env.reset(seed=3)
        active = env.order[0]
        cube = env.data.xpos[env.cube_ids[active]].copy()
        goal = env.data.site_xpos[env.goal_ids[active]].copy()

        self._move_hand(env, cube + [0, 0, 0.18], True)
        self._move_hand(env, cube + [0, 0, 0.105], True)
        self._move_hand(env, cube + [0, 0, 0.105], False, 0.8)
        self._move_hand(env, cube + [0, 0, 0.25], False)
        grasp_offset = env.data.xpos[env.hand_id] - env.data.xpos[env.cube_ids[active]]
        place = goal + grasp_offset + [0, 0, 0.025]
        self._move_hand(env, place + [0, 0, 0.12], False)
        self._move_hand(env, place, False)

        self.assertEqual(env.stage, 0)
        completion = self._move_hand(env, place + [0, 0, 0.08], True, 0.8)

        self.assertEqual(env.stage, 1)
        self.assertIsNotNone(completion)
        self.assertTrue(completion["released"])
        self.assertTrue(completion["settled"])
        self.assertGreaterEqual(completion["settle_steps"], 10)
        env.close()

    def test_lift_curriculum_terminates_on_secure_lift(self):
        env = RLPickPlaceEnv()
        env.reset(seed=3)
        env.set_training_phase("lift")
        active = env.order[0]
        cube = env.data.xpos[env.cube_ids[active]].copy()

        self._move_hand(env, cube + [0, 0, 0.18], True)
        self._move_hand(env, cube + [0, 0, 0.105], True)
        self._move_hand(env, cube + [0, 0, 0.105], False, 0.8)

        terminated = False
        info = None
        target = cube + [0, 0, 0.25]
        for _ in range(100):
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(
                (target - env.data.xpos[env.hand_id]) / 0.02,
                -1.0,
                1.0,
            )
            action[3] = -1.0
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertTrue(info["first_box_lifted"])
        self.assertTrue(info["training_success"])
        self.assertFalse(info["full_task_success"])
        env.close()


if __name__ == "__main__":
    unittest.main()
