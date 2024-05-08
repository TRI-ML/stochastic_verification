import os
import time
import torch
import dill
import yaml
import hydra
import random
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner


# choose from "can", "lift", "square", "tool_hang", "transport"
task = "can"
modified = True # whether to apply domain modifications or not
total_rollouts = 1000
rollouts_per_sim = 10 # how many rollouts are run in parallel


### Make Results Folder ###
current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
run_folder = "results/" + task + "/" + str(current_datetime) + "/"
os.mkdir(run_folder)
outfile = run_folder + "reward_array.npy"
outfile_video_paths = run_folder + "video_paths.txt"


### Load Model ###
# pretrained policies
model_filename = "data/experiments/image/" + task + "_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
payload = torch.load(open(model_filename, 'rb'), pickle_module=dill)
print("--Loaded payload--")


### Make Workspace ###
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
print("--Loaded workspace--")


### Make Policy ###
policy = workspace.model
device = torch.device('cuda')
policy.eval().to(device)
print("--Loaded policy--")


### Modify Environment ###
lift_geom_names = ['cube_g0_vis', 'table_visual']

can_geom_names = ['floor', 'Can_g0_visual']

square_geom_names = ['table_visual', 'SquareNut_g0_visual', 'SquareNut_g1_visual',
                     'SquareNut_g2_visual', 'SquareNut_g3_visual', 'SquareNut_g4_visual']

tool_hang_geom_names = ['table_visual', 'wall_front_visual', 'tool_handle_g0_vis']

transport_geom_names = ['floor', 'wall_leftcorner_visual', 'wall_rightcorner_visual',
                   'wall_left_visual', 'wall_right_visual', 'wall_rear_visual',
                   'wall_front_visual', 'payload_handle_vis', 'payload_head_vis',
                   'trash_g0_vis', 'transport_start_bin_lid_handle_vis']

# colors, see www.rapidtables.com/web/color/RGB_Color.html
burly_wood = (222,184,135)
sienna = (160,82,45)
dodger_blue = (30,144,255)
dim_gray = (105,105,105)
beige = (245,245,220)
tan = (210,180,140)
wheat = (245,222,179)
dark_slate_gray = (47,79,79)
lime_green = (50,205,50)
silver = (192,192,192)
maroon = (128,0,0)
floral_white = (255,250,240)
azure = (240,255,255)
black = (0,0,0)

# define environment modifications
if modified:
    if task == "can":
        modded_geom_names = ['floor', 'Can_g0_visual']
        modded_geom_rgbs = [beige, lime_green]
    elif task == "lift":
        modded_geom_names = ['cube_g0_vis']
        modded_geom_rgbs = [dodger_blue]
    elif task == "square":
        modded_geom_names = ['SquareNut_g0_visual', 'SquareNut_g1_visual',
                        'SquareNut_g2_visual', 'SquareNut_g3_visual', 'SquareNut_g4_visual']
        modded_geom_rgbs = [wheat, wheat, wheat, wheat, wheat]
    elif task == "tool_hang":
        modded_geom_names = ['wall_front_visual']
        modded_geom_rgbs = [floral_white]
    elif task == "transport":
        modded_geom_names = ['trash_g0_vis', 'transport_start_bin_lid_handle_vis']
        modded_geom_rgbs = [lime_green, silver]
else:
    modded_geom_names = []
    modded_geom_rgbs = []


### Make Environment Runner ###
random.seed(a=None)
task_cfg_path = 'diffusion_policy/config/task/' + task + '_image_abs.yaml'
task_cfg = OmegaConf.load(task_cfg_path)
seeds = random.sample(range(1_000, 10_000_000), total_rollouts) # trained on seed=42, default test_start_seed is 10000

# for our videos we always use agentview
if task == "can" or task == "lift" or task == "square":
    render_obs_key = 'agentview_image'
    max_steps = 500
elif task == "transport":
    render_obs_key = 'shouldercamera0_image'
    max_steps = 750
elif task == "tool_hang":
    render_obs_key = 'sideview_image'
    max_steps = 750
# render_obs_key = 'agentview_image' # 'robot0_eye_in_hand_image', 'sideview_image', task_cfg["env_runner"]["render_obs_key"]

dataset_path = "data/robomimic/datasets/" + task + "/ph/image_abs.hdf5"


### Save Metadata ###
metadata = {"task": task,
          "max_steps": max_steps,
          "model_filename": model_filename,
          "modded_geom_names": modded_geom_names,
          "modded_geom_rgbs": modded_geom_rgbs,
          "seeds": seeds,
          "total_rollouts": total_rollouts,
          "rollouts_per_sim": rollouts_per_sim,
          "modified": modified
          }
run_yaml = run_folder + "metadata.yaml"
with open(run_yaml, 'w') as yamlfile:
    yaml.dump(metadata, yamlfile)
    print("\nsaved metadata")


### Make Runner ###
start_time = time.time()
for i in range(0, total_rollouts, rollouts_per_sim):
    print("\n\n\ni: ", i, "/", total_rollouts)
    elapsed = round(time.time() - start_time)
    print(elapsed, " seconds elapsed since start") 
    iter_seeds = seeds[i:i+rollouts_per_sim]
    runner = RobomimicImageRunner(output_dir='', dataset_path=dataset_path, shape_meta=task_cfg["shape_meta"], 
                                n_train=0, max_steps=max_steps, n_test=rollouts_per_sim, n_test_vis=rollouts_per_sim, 
                                abs_action=True, test_seeds=iter_seeds, render_obs_key=render_obs_key, 
                                modded_geom_names=modded_geom_names, modded_geom_rgbs=modded_geom_rgbs)
    print("--Made runner--")
    
    ### Execute Policy ###
    rewards, video_paths = runner.run(policy)
    print("--Finished policy run--")

    ### Save Results ###
    if i == 0:
        np.save(outfile, np.array(rewards))
    else:
        existing_array = np.load(outfile)
        new_array = np.vstack((existing_array, np.array(rewards)))
        np.save(outfile, new_array)
    
    with open(outfile_video_paths, "a") as file:
            for video_path in video_paths:
                file.write(video_path + "\n")


print("\n\nFinished!")