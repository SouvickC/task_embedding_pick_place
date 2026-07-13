import gymnasium as gym
import mujoco
import numpy as np

from pick_place import PickPlaceEnv


class RLPickPlaceEnv(PickPlaceEnv):
    """PPO controls Cartesian hand motion while IK controls robot joints."""

    def __init__(self, render_mode=None, required_stages=2):
        super().__init__(render_mode, required_stages=required_stages)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32,
        )

        self.previous_grasp_to_cube = None
        self.previous_cube_to_goal = None
        self.previous_cube_height = None
        self.previous_gripper_opening = None
        self.finger_ids = {
            self.model.body("left_finger").id,
            self.model.body("right_finger").id,
        }
        self.table_geom_id = self.model.geom("table").id
        self.cube_qvel = [
            self.model.jnt_dofadr[self.model.body_jntadr[i]]
            for i in self.cube_ids
        ]

    def _gripper_opening(self):
        # Two 4 cm finger joints give an 8 cm maximum opening.
        return float(self.data.qpos[7] + self.data.qpos[8])

    def _grasp_point(self):
        # The fingertip center is 10.5 cm along the hand body's local z-axis.
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + 0.105 * rotation[:, 2]

    def observation(self):
        return np.concatenate(
            (
                self._grasp_point(),
                [self._gripper_opening()],
                super().observation(),
            )
        ).astype(np.float32)

    def _get_task_state(self, stage):
        active = self.order[min(stage, len(self.order) - 1)]

        grasp = self._grasp_point()
        cube = self.data.xpos[self.cube_ids[active]].copy()
        goal = self.data.site_xpos[self.goal_ids[active]].copy()

        grasp_to_cube = float(np.linalg.norm(grasp - cube))
        cube_to_goal = float(np.linalg.norm(cube - goal))

        return active, grasp, cube, goal, grasp_to_cube, cube_to_goal

    def reset(self, *, seed=None, options=None):
        self.has_lifted = np.zeros(2, dtype=bool)
        self.has_grasped = np.zeros(2, dtype=bool)
        self.has_released = np.zeros(2, dtype=bool)
        self.settle_steps = np.zeros(2, dtype=np.int32)
        self.grasp_contact_steps = np.zeros(2, dtype=np.int32)
        self.grasp_miss_steps = np.zeros(2, dtype=np.int32)
        self.grasp_held = np.zeros(2, dtype=bool)
        observation, info = super().reset(seed=seed, options=options)

        _, _, cube, _, grasp_to_cube, cube_to_goal = self._get_task_state(self.stage)

        self.previous_grasp_to_cube = grasp_to_cube
        self.previous_cube_to_goal = cube_to_goal
        self.previous_cube_height = float(cube[2])
        self.previous_gripper_opening = self._gripper_opening()

        return observation, info

    def _cube_grasped(self, active):
        cube_id = self.cube_ids[active]
        touching_fingers = set()
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == cube_id and body2 in self.finger_ids:
                touching_fingers.add(body2)
            if body2 == cube_id and body1 in self.finger_ids:
                touching_fingers.add(body1)
        return touching_fingers == self.finger_ids

    def _cube_touches_table(self, active):
        cube_id = self.cube_ids[active]
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if contact.geom1 == self.table_geom_id and body2 == cube_id:
                return True
            if contact.geom2 == self.table_geom_id and body1 == cube_id:
                return True
        return False

    def _update_grasp_state(self, active):
        raw_contact = self._cube_grasped(active)
        if raw_contact:
            self.grasp_contact_steps[active] += 1
            self.grasp_miss_steps[active] = 0
        else:
            self.grasp_contact_steps[active] = 0
            self.grasp_miss_steps[active] += 1

        if self.grasp_contact_steps[active] >= 5:
            self.has_grasped[active] = True

        grasp_distance = float(
            np.linalg.norm(
                self._grasp_point() - self.data.xpos[self.cube_ids[active]]
            )
        )
        briefly_missing_contact = (
            self.grasp_miss_steps[active] <= 3
            and self._gripper_opening() < 0.05
            and grasp_distance < 0.06
        )
        self.grasp_held[active] = bool(
            self.has_grasped[active]
            and (raw_contact or briefly_missing_contact)
        )
        return raw_contact, bool(self.grasp_held[active])

    def _placement_state(self, active):
        cube = self.data.xpos[self.cube_ids[active]]
        goal = self.data.site_xpos[self.goal_ids[active]]
        velocity_address = self.cube_qvel[active]
        linear_speed = float(np.linalg.norm(self.data.qvel[velocity_address : velocity_address + 3]))
        angular_speed = float(np.linalg.norm(self.data.qvel[velocity_address + 3 : velocity_address + 6]))
        horizontal_distance = float(np.linalg.norm(cube[:2] - goal[:2]))
        in_goal = horizontal_distance < 0.08 and 0.375 < cube[2] < 0.405
        released = self._gripper_opening() > 0.055 and not self._cube_grasped(active)
        settled = (
            in_goal
            and released
            and self._cube_touches_table(active)
            and linear_speed < 0.05
            and angular_speed < 0.5
        )
        return in_goal, released, settled, horizontal_distance, linear_speed, angular_speed

    def stage_complete(self, active, distance):
        _, grasp_held = self._update_grasp_state(active)
        if grasp_held and self.data.xpos[self.cube_ids[active], 2] > 0.43:
            self.has_lifted[active] = True

        in_goal, released, settled, _, _, _ = self._placement_state(active)
        if self.has_lifted[active] and in_goal and released:
            self.has_released[active] = True
        if self.has_lifted[active] and self.has_released[active] and settled:
            self.settle_steps[active] += 1
        else:
            self.settle_steps[active] = 0
        return self.settle_steps[active] >= 10

    def set_required_stages(self, required_stages):
        if required_stages not in (1, 2):
            raise ValueError("required_stages must be 1 or 2")
        self.required_stages = required_stages

    def _joint_action(self, hand_target, gripper):
        position_error = hand_target - self.data.xpos[self.hand_id]

        current_rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(current_rotation[:, i], self.hand_rotation[:, i])
            for i in range(3)
        )

        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacobian_pos,
            jacobian_rot,
            self.hand_id,
        )

        jacobian = np.vstack(
            (
                jacobian_pos[:, :7],
                jacobian_rot[:, :7],
            )
        )

        error = np.concatenate((position_error, rotation_error))

        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.03 * np.eye(6),
            error,
        )

        joint_action = np.zeros(8)
        joint_action[:7] = np.clip(delta / 0.04, -1.0, 1.0)
        joint_action[7] = gripper

        return joint_action

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        previous_stage = self.stage
        previous_active = self.order[min(previous_stage, len(self.order) - 1)]
        was_lifted = bool(self.has_lifted[previous_active])
        was_grasped = bool(self.has_grasped[previous_active])

        hand_target = (
            self.data.xpos[self.hand_id].copy()
            + 0.02 * action[:3]
        )

        observation, _, terminated, truncated, info = super().step(
            self._joint_action(hand_target, action[3])
        )

        (
            active,
            grasp,
            cube,
            goal,
            grasp_to_cube,
            cube_to_goal,
        ) = self._get_task_state(previous_stage)

        reach_progress = self.previous_grasp_to_cube - grasp_to_cube
        goal_progress = self.previous_cube_to_goal - cube_to_goal
        lift_progress = float(cube[2]) - self.previous_cube_height
        gripper_opening = self._gripper_opening()
        closing_progress = self.previous_gripper_opening - gripper_opening
        opening_progress = -closing_progress

        near_cube = grasp_to_cube < 0.04
        lifted = self.has_lifted[active]
        newly_lifted = bool(lifted and not was_lifted)
        grasped = self._cube_grasped(active)
        grasp_held = bool(self.grasp_held[active])
        newly_grasped = bool(self.has_grasped[active] and not was_grasped)
        in_goal, released, settled, horizontal_goal_distance, linear_speed, angular_speed = self._placement_state(active)

        reward = -0.005
        if not grasp_held:
            reward += 10.0 * reach_progress

        # Closing is useful only when the fingertips are aligned with the cube.
        if near_cube and not grasp_held:
            reward += 5.0 * closing_progress
        elif action[3] < 0.0 and not grasp_held:
            reward -= 0.01

        if newly_grasped:
            reward += 10.0

        # Teach raising only until the lift milestone; lowering is needed later.
        if grasp_held and not lifted:
            reward += 30.0 * lift_progress

        # Award this milestone only on the false -> true state transition.
        if newly_lifted:
            reward += 10.0

        # Transport only after the cube has been lifted.
        if lifted and grasp_held:
            reward += 20.0 * goal_progress
        elif lifted and not in_goal:
            reward -= 0.02

        # Opening is useful only after lowering the carried cube into the goal.
        if lifted and in_goal:
            reward += 5.0 * opening_progress
        elif action[3] > 0.0 and self.has_grasped[active] and not self.has_released[active]:
            reward -= 0.01

        stage_completed = self.stage > previous_stage
        if stage_completed:
            reward += 50.0

        task_success = bool(
            terminated and self.stage >= len(self.order)
        )
        if task_success:
            reward += 100.0

        if stage_completed and not terminated:
            _, _, next_cube, _, next_grasp_distance, next_goal_distance = self._get_task_state(self.stage)
            self.previous_grasp_to_cube = next_grasp_distance
            self.previous_cube_to_goal = next_goal_distance
            self.previous_cube_height = float(next_cube[2])
        else:
            self.previous_grasp_to_cube = grasp_to_cube
            self.previous_cube_to_goal = cube_to_goal
            self.previous_cube_height = float(cube[2])

        info.update(
            grasp_to_cube=grasp_to_cube,
            cube_to_goal=cube_to_goal,
            cube_height=float(cube[2]),
            gripper_opening=gripper_opening,
            grasped=grasped,
            grasp_held=grasp_held,
            grasp_contact_steps=int(self.grasp_contact_steps[active]),
            newly_grasped=newly_grasped,
            newly_lifted=newly_lifted,
            in_goal=in_goal,
            released=released,
            settled=settled,
            settle_steps=int(self.settle_steps[active]),
            horizontal_goal_distance=horizontal_goal_distance,
            cube_linear_speed=linear_speed,
            cube_angular_speed=angular_speed,
            reach_progress=float(reach_progress),
            lift_progress=float(lift_progress),
            goal_progress=float(goal_progress),
            active_cube=active,
            stage_completed=stage_completed,
            first_box_grasped=bool(self.has_grasped[self.order[0]]),
            first_box_lifted=bool(self.has_lifted[self.order[0]]),
            first_box_released=bool(self.has_released[self.order[0]]),
            first_box_success=self.stage >= 1,
            full_task_success=task_success,
            is_success=task_success,
        )

        self.previous_gripper_opening = gripper_opening

        return observation, float(reward), terminated, truncated, info
