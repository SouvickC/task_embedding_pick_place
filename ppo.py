import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_env import RLPickPlaceEnv


def make_env():
    return RLPickPlaceEnv()


class SuccessRateCallback(BaseCallback):
    """Log rolling partial and full task success without changing policy spaces."""

    def __init__(self, window_size=100):
        super().__init__()
        self.first_box_results = deque(maxlen=window_size)
        self.full_task_results = deque(maxlen=window_size)
        self.grasp_results = deque(maxlen=window_size)
        self.lift_results = deque(maxlen=window_size)
        self.release_results = deque(maxlen=window_size)
        self.training_results = deque(maxlen=window_size)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.grasp_results.append(float(info["first_box_grasped"]))
                self.lift_results.append(float(info["first_box_lifted"]))
                self.release_results.append(float(info["first_box_released"]))
                self.first_box_results.append(float(info["first_box_success"]))
                self.full_task_results.append(float(info["full_task_success"]))
                self.training_results.append(float(info["training_success"]))

        if self.first_box_results:
            self.logger.record("rollout/first_box_grasp_rate", np.mean(self.grasp_results))
            self.logger.record("rollout/first_box_lift_rate", np.mean(self.lift_results))
            self.logger.record("rollout/first_box_release_rate", np.mean(self.release_results))
            self.logger.record(
                "rollout/first_box_success_rate",
                np.mean(self.first_box_results),
            )
            self.logger.record(
                "rollout/full_task_success_rate",
                np.mean(self.full_task_results),
            )
            self.logger.record(
                "rollout/training_success_rate",
                np.mean(self.training_results),
            )
        return True


class CurriculumCallback(BaseCallback):
    """Progress from lifting to one placement and then the two-box task."""

    def __init__(self, threshold=0.7, window_size=100, min_timesteps=100_000):
        super().__init__()
        self.threshold = threshold
        self.window_size = window_size
        self.min_timesteps = min_timesteps
        self.successes = deque(maxlen=window_size)
        self.phases = ("lift", "place_one", "place_two")
        self.current_phase_index = 0
        self.phase_start_step = 0

    @property
    def current_phase(self):
        return self.phases[self.current_phase_index]

    def _set_phase(self, phase_index):
        if phase_index != self.current_phase_index or self.n_calls == 0:
            self.current_phase_index = phase_index
            self.training_env.env_method(
                "set_training_phase",
                self.current_phase,
            )
            self.successes.clear()
            self.phase_start_step = self.num_timesteps

    def _on_training_start(self):
        self._set_phase(0)

    def _on_step(self):
        if self.current_phase_index < len(self.phases) - 1:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.successes.append(float(info["training_success"]))

            ready = (
                self.num_timesteps - self.phase_start_step >= self.min_timesteps
                and len(self.successes) == self.window_size
                and np.mean(self.successes) >= self.threshold
            )
            if ready:
                self._set_phase(self.current_phase_index + 1)

        success_rate = np.mean(self.successes) if self.successes else 0.0
        self.logger.record("curriculum/phase", self.current_phase_index)
        self.logger.record(
            "curriculum/required_stages",
            2 if self.current_phase == "place_two" else 1,
        )
        self.logger.record("curriculum/phase_success_rate", success_rate)
        self.logger.record("curriculum/target_success_rate", self.threshold)
        return True


def train(
    steps,
    resume=None,
    device="auto",
    env_count=4,
    curriculum_threshold=0.7,
    curriculum_window=100,
    curriculum_min_steps=100_000,
):
    # Subprocesses let cluster CPU cores simulate environments in parallel.
    vec_type = DummyVecEnv if env_count == 1 else SubprocVecEnv
    env = VecMonitor(vec_type([make_env for _ in range(env_count)]))
    if resume:
        model = PPO.load(resume, env=env, tensorboard_log="runs/", device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.98,
            ent_coef=0.005,
            n_epochs=10,
            policy_kwargs={"net_arch": [128, 128]},
            tensorboard_log="runs/",
            verbose=1,
            device=device,
        )

    # save_freq counts vector steps, so divide by the number of environments.
    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // env_count, 1),
        save_path="checkpoints/",
        name_prefix="ppo_pick_place_v7",
    )
    callbacks = CallbackList([
        checkpoint,
        SuccessRateCallback(),
        CurriculumCallback(
            threshold=curriculum_threshold,
            window_size=curriculum_window,
            min_timesteps=curriculum_min_steps,
        ),
    ])
    model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=not resume)
    model.save("checkpoints/ppo_pick_place_v7")
    env.close()


def configure_camera(camera):
    camera.lookat[:] = (0.50, 0.0, 0.35)
    camera.distance = 1.45
    camera.azimuth = 135
    camera.elevation = -25


def record_playback(checkpoint, output_path):
    import imageio.v2 as imageio
    import mujoco

    env = RLPickPlaceEnv()
    model = PPO.load(checkpoint, device="cpu")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.model, height=1080, width=1920)
    camera = mujoco.MjvCamera()
    configure_camera(camera)
    writer = imageio.get_writer(
        output,
        fps=50,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
    )

    try:
        observation, info = env.reset()
        print("checkpoint:", checkpoint)
        print("Task order:", info["order"])

        while True:
            renderer.update_scene(env.data, camera=camera)
            writer.append_data(renderer.render())
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                renderer.update_scene(env.data, camera=camera)
                writer.append_data(renderer.render())
                print("Success:", info["is_success"])
                break
    finally:
        writer.close()
        renderer.close()
        env.close()

    print("Recording saved to:", output.resolve())


def play(checkpoint, record_path=None):
    if record_path:
        record_playback(checkpoint, record_path)
        return

    env = RLPickPlaceEnv(render_mode="human")
    model = PPO.load(checkpoint, device="cpu")
    print("checkpoint:", checkpoint)
    observation, info = env.reset()
    env.render()
    print("Task order:", info["order"])

    while env.viewer.is_running():
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        time.sleep(env.model.opt.timestep * env.frame_skip)
        if terminated or truncated:
            print("Success:", info["is_success"])
            observation, info = env.reset()
    env.close()


def evaluate(checkpoint, episodes):
    env = RLPickPlaceEnv()
    model = PPO.load(checkpoint, device="cpu")
    first_box_successes = 0
    full_task_successes = 0

    for seed in range(episodes):
        observation, _ = env.reset(seed=seed)
        min_grasp = np.inf
        max_height = 0.0

        for step in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            min_grasp = min(min_grasp, info["grasp_to_cube"])
            max_height = max(max_height, info["cube_height"])
            if terminated or truncated:
                break

        first_box_successes += int(info["first_box_success"])
        full_task_successes += int(info["full_task_success"])
        print(
            f"episode={seed} stage={env.stage} steps={step + 1} "
            f"min_grasp={min_grasp:.3f}m max_height={max_height:.3f}m"
        )

    print(f"first_box_success_rate={first_box_successes / episodes:.1%}")
    print(f"full_task_success_rate={full_task_successes / episodes:.1%}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument(
        "--record",
        nargs="?",
        const="recordings/ppo_pick_place.mp4",
        metavar="PATH",
        help="With --play, record one 1080p 50 FPS episode to MP4",
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--resume", help="Checkpoint path to continue training from")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--envs", type=int, default=4, help="Parallel simulation environments")
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.7,
        help="One-box success rate required before enabling the two-box task",
    )
    parser.add_argument("--curriculum-window", type=int, default=100)
    parser.add_argument("--curriculum-min-steps", type=int, default=100_000)
    parser.add_argument("--evaluate", type=int, metavar="EPISODES")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_pick_place_v7.zip")
    args = parser.parse_args()
    if args.play:
        play(args.checkpoint, args.record)
    elif args.evaluate:
        evaluate(args.checkpoint, args.evaluate)
    else:
        train(
            args.steps,
            args.resume,
            args.device,
            args.envs,
            args.curriculum_threshold,
            args.curriculum_window,
            args.curriculum_min_steps,
        )
