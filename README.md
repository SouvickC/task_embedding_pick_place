# MuJoCo Task-Embedding Pick and Place

A small learning project: a Franka Panda must place two cubes in an instructed
order. It runs on CPU with native MuJoCo; MJLab is not required.

## Run

Create the isolated Python 3.11 environment once:

```powershell
python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the IK demonstration with `.\.venv\Scripts\python.exe demo.py`.

The demo uses damped-least-squares inverse kinematics. Watch the arm complete
the first placement before starting the second.

## PPO

Train the CPU PPO policy and then visualize it:

```powershell
.\.venv\Scripts\python.exe ppo.py --steps 3000000
.\.venv\Scripts\python.exe ppo.py --play
```

Training saves every 25,000 steps. Continue from the final checkpoint with:

```powershell
.\.venv\Scripts\python.exe ppo.py --steps 200000 --resume checkpoints/ppo_pick_place_v5.zip
```

`--steps` means additional steps when resuming. A periodic checkpoint such as
`checkpoints/ppo_pick_place_v5_100000_steps.zip` can be passed the same way.

Fresh training uses a curriculum without changing the observation or action
shape. It stays on one-box episodes until the rolling one-box placement rate is
at least 70% across 100 episodes, after at least 100,000 timesteps. It then
switches permanently to the complete two-box task. Configure these gates with
`--curriculum-threshold`, `--curriculum-window`, and
`--curriculum-min-steps`. The stable-grasp task uses `v5` checkpoints; do not
resume older checkpoints trained with the previous grasp transition.

On a Linux GPU cluster, use several CPU simulation workers and CUDA for PPO:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ppo.py --steps 2000000 --envs 16 --device cuda
```

Download `checkpoints/ppo_pick_place_v5.zip` and place it in the laptop's
`checkpoints` directory. `ppo.py --play` always loads it on CPU.

View reward and optimization curves during training:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs
```

TensorBoard reports first-box grasp, lift, release, and placement rates, plus
`rollout/full_task_success_rate` and `curriculum/required_stages`. Placement
requires a grasped lift, release in the goal, table contact, and ten stable
simulation steps.

`rl_env.py` gives PPO four actions: hand movement in x/y/z and the gripper.
Inverse kinematics converts hand movement into Panda joint targets. The policy
therefore learns task behavior without also having to discover robot kinematics.
Its observation adds the fingertip XYZ position and gripper opening to the
original 32 environment values.

## Task embedding

`PickPlaceEnv.observation()` contains a four-value task vector:

```text
[red first, blue first, red second, blue second]
```

It also contains a two-value stage vector. PPO receives these values together
with robot, cube, and goal state. It uses the one-hot vectors directly; there
is no learned embedding layer.

Stage switching lives in `PickPlaceEnv.step()`, not PPO. A successful first
placement increments `stage`, selects the second cube for reward calculation,
and changes the stage one-hot values. The same PPO policy then responds to the
new observation automatically.

The project is intentionally compact: `scene.xml` defines the world,
`pick_place.py` defines the Gymnasium environment, `demo.py` is the visible
scripted expert, `rl_env.py` adds the RL action and reward, and `ppo.py` trains
or displays PPO.
