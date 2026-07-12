import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_env import RLPickPlaceEnv


def make_env():
    return RLPickPlaceEnv()


def train(steps, resume=None, device="auto", env_count=4):
    # Subprocesses let cluster CPU cores simulate environments in parallel.
    vec_type = DummyVecEnv if env_count == 1 else SubprocVecEnv
    env = VecMonitor(vec_type([make_env for _ in range(env_count)]))
    if resume:
        model = PPO.load(resume, env=env, tensorboard_log="runs/", device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.995,
            policy_kwargs={"net_arch": [128, 128]},
            tensorboard_log="runs/",
            verbose=1,
            device=device,
        )

    # save_freq counts vector steps, so divide by the number of environments.
    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // env_count, 1),
        save_path="checkpoints/",
        name_prefix="ppo_pick_place",
    )
    model.learn(total_timesteps=steps, callback=checkpoint, reset_num_timesteps=not resume)
    model.save("checkpoints/ppo_pick_place")
    env.close()


def play():
    env = RLPickPlaceEnv(render_mode="human")
    model = PPO.load("checkpoints/ppo_pick_place", device="cpu")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--resume", help="Checkpoint path to continue training from")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--envs", type=int, default=4, help="Parallel simulation environments")
    args = parser.parse_args()
    play() if args.play else train(args.steps, args.resume, args.device, args.envs)
