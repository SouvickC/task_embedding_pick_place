import unittest

import numpy as np

from demo import run_scripted_episode
from pick_place import PickPlaceEnv


class PickPlaceTest(unittest.TestCase):
    def test_environment_and_scripted_expert(self):
        env = PickPlaceEnv()
        observation, info = env.reset(seed=3)
        self.assertEqual(observation.shape, (32,))
        self.assertEqual(sorted(info["order"]), [0, 1])

        run_scripted_episode(env)
        distances = [
            np.linalg.norm(env.data.xpos[env.cube_ids[i]] - env.data.site_xpos[env.goal_ids[i]])
            for i in range(2)
        ]
        self.assertEqual(env.stage, 2)
        self.assertTrue(all(distance < 0.08 for distance in distances))
        env.close()


if __name__ == "__main__":
    unittest.main()
