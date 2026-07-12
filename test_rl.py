import unittest

import numpy as np

from rl_env import RLPickPlaceEnv


class RLEnvironmentTest(unittest.TestCase):
    def test_rl_action_and_one_hot_task(self):
        env = RLPickPlaceEnv()
        observation, _ = env.reset(seed=3)
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(observation[-6:].tolist(), [0, 1, 1, 0, 1, 0])

        next_observation, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertEqual(next_observation.shape, (35,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("hand_to_cube", info)
        env.close()


if __name__ == "__main__":
    unittest.main()
