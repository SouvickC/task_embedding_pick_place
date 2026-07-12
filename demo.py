import time

import mujoco
import numpy as np

from pick_place import PickPlaceEnv


def joint_action_toward(env, target, gripper_open):
    """One damped-least-squares IK update toward a Cartesian hand target."""
    position_error = target - env.data.xpos[env.hand_id]
    current = env.data.xmat[env.hand_id].reshape(3, 3)
    rotation_error = 0.5 * sum(np.cross(current[:, i], env.hand_rotation[:, i]) for i in range(3))
    jacobian_pos = np.zeros((3, env.model.nv))
    jacobian_rot = np.zeros((3, env.model.nv))
    mujoco.mj_jacBody(env.model, env.data, jacobian_pos, jacobian_rot, env.hand_id)
    jacobian = np.vstack((jacobian_pos[:, :7], jacobian_rot[:, :7]))
    error = np.concatenate((position_error, rotation_error))
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03 * np.eye(6), error)
    action = np.zeros(8)
    action[:7] = np.clip(delta / 0.04, -1, 1)
    action[7] = 1 if gripper_open else -1
    return action


def move(env, target, gripper_open, seconds=1.2):
    for _ in range(int(seconds / (env.model.opt.timestep * env.frame_skip))):
        env.step(joint_action_toward(env, target, gripper_open))
        if env.viewer is not None:
            time.sleep(env.model.opt.timestep * env.frame_skip)


def run_scripted_episode(env):
    for cube_index in env.order:
        cube = env.data.xpos[env.cube_ids[cube_index]].copy()
        goal = env.data.site_xpos[env.goal_ids[cube_index]].copy()
        # All waypoint offsets are meters above the cube or goal.
        move(env, cube + [0, 0, 0.18], True)
        move(env, cube + [0, 0, 0.105], True)
        move(env, cube + [0, 0, 0.105], False, 0.8)
        move(env, cube + [0, 0, 0.25], False)
        grasp_offset = env.data.xpos[env.hand_id] - env.data.xpos[env.cube_ids[cube_index]]
        place = goal + grasp_offset + [0, 0, 0.025]
        move(env, place + [0, 0, 0.12], False)
        move(env, place, False)
        move(env, place + [0, 0, 0.08], True, 0.8)


def main():
    env = PickPlaceEnv(render_mode="human")
    _, info = env.reset(seed=3)
    env.render()
    print("Task order:", ["red", "blue"] if info["order"][0] == 0 else ["blue", "red"])
    run_scripted_episode(env)
    print("Finished. Close the viewer window to exit.")
    while env.viewer.is_running():
        env.render()
        time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
