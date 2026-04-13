# imitate_episodes.py  -- Main Training Script
# This is the script you run to train the model. It loads the surgical
# demonstration data (camera images + robot kinematics), feeds them through
# the ACT model, computes how far off the predictions are from the real
# actions, and updates the model weights to reduce that error. Runs for
# thousands of "epochs" (passes through the data) and saves the best
# checkpoint. Used for all model versions: v1, v2, and GC-ACT.

import sys
sys.path.append("$PATH_TO_SUTUREBOT/src")  # to import aloha
import torch
import numpy as np
import os
import pickle
import argparse
import wandb
import cv2
import math
import threading
import time
import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from collections import deque
from queue import Queue


from copy import deepcopy

## TODO: merge load_merged_data and load_data_dvrk
from utils import load_data_dvrk  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy
from aloha_pro.aloha_scripts.utils import (
    initialize_model_and_tokenizer,
    encode_text,
    crop_resize,
    is_multi_gpu_checkpoint,
    modify_real_time,
    visualize_language_correction,
    create_dataset_path,
    memory_monitor,
    save_trajectory,
)
# from instructor.train import build_instructor

CROP_TOP = True  # for aloha pro, whose top camera is high
CKPT = 0  # 0 for policy_last, otherwise put the ckpt number here
AUDIO = False
option = 0
intervention_needed = threading.Event()  # flag to signal an intervention
recorded_commands = Queue()


def signal_handler(sig, frame):
    exit()


def on_press(key):
    global option
    if hasattr(key, "char") and key.char in ["1", "2", "3", "4", "5"]:
        option = int(key.char)
    else:
        option = 0


def on_release(key):
    global option
    if hasattr(key, "char") and key.char in ["1", "2", "3", "4", "5"]:
        option = 0


def predict_instruction(instructor, history_obs, history_step_size, query_frequency):
    # Ensuring that instructor_input has the last few observations with length history_len + 1
    # and that the last observation in history_obs is the last one in instructor_input.
    selected_indices = [
        -1 - i * max((history_step_size // query_frequency), 1)
        for i in range(instructor.history_len + 1)
    ]
    selected_obs = [
        history_obs[idx] for idx in selected_indices if idx >= -len(history_obs)
    ]
    selected_obs.reverse()
    instructor_input = torch.stack(selected_obs, dim=1)
    assert instructor_input.shape[1] == min(
        instructor.history_len + 1, len(history_obs)
    )

    logits, temperature = instructor(instructor_input)
    decoded_texts = instructor.decode_logits(logits, temperature)[0]
    return decoded_texts


def transcribe_from_ros(msg):
    """Listen for commands in the background."""
    global recorded_commands
    if msg.data:
        command = msg.data
        print(f"Transcribed raw command: {command}")
        if command in ["stop", "pardon", "wait"]:
            print("Stop command detected.")
            intervention_needed.set()
        else:
            if intervention_needed.is_set():
                command = modify_real_time(command)
                # Check if the command is valid after modifications
                if command and len(command.split()) > 1:
                    print(f"put into the queue: {command}")
                    recorded_commands.put(command)
            else:
                while not recorded_commands.empty():
                    command = recorded_commands.get(block=False)
                    print(f"Intervention not needed, ignoring command: {command}.")


def get_user_command():
    global recorded_commands
    if AUDIO:
        print("Listening for command...")
        command = recorded_commands.get()

        # If a valid command is detected
        if command:
            print(f"Transcribed user command: {command}")

    else:
        command = input("Please provide a command: ")
    # Removing leading numbers from the string
    command = "".join(filter(lambda x: not x.isdigit(), command))
    command = modify_real_time(command)
    return command


def generate_command_embedding(
    command, t, language_encoder, tokenizer, model, instructor=None
):
    print(f"Command at {t=}: {command}")

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding


def main(args):
    set_seed(1)

    signal.signal(signal.SIGINT, signal_handler)
    threading.Thread(
        target=memory_monitor, daemon=True
    ).start()  # Start the memory monitor thread

    # Command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args['batch_size']
    num_epochs = args["num_epochs"]
    log_wandb = args["log_wandb"]
    # Split the command by commas to get a list of commands
    commands = args["command"].split(",") if args["command"] else []
    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    multi_gpu = args["multi_gpu"]
    instructor_path = args["instructor_path"]
    history_len = args["history_len"]
    history_step_size = args["history_step_size"]
    hl_margin = args["hl_margin"]
    policy_level = args["policy_level"]
    use_gesture = args.get("use_gesture", False)
    gesture_dim = args.get("gesture_dim", 10)
    labels_dir = args.get("labels_dir", os.path.expanduser("~/data/labels"))
    resume_ckpt = args.get("resume_ckpt", None)

    # Set up wandb
    if log_wandb:
        if is_eval:
            # run_name += ".eval"
            log_wandb = False
        else:
            run_name = ckpt_dir.split("/")[-1] + f".{args['seed']}"
            wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
            # check if wandb run exists
            if os.path.exists(wandb_run_id_path):
                with open(wandb_run_id_path, "r") as f:
                    saved_run_id = f.read().strip()
                wandb.init(
                    project="suturebot",
                    entity=os.getenv("WANDB_ENTITY"),
                    name=run_name,
                    id=saved_run_id,
                    resume="allow",
                )
            else:
                wandb.init(
                    project="suturebot",
                    entity=os.getenv("WANDB_ENTITY"),
                    name=run_name,
                    config=args,
                    resume="allow",
                )
                # Ensure the directory exists before trying to open the file
                os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
                with open(wandb_run_id_path, "w") as f:
                    f.write(wandb.run.id)

    if args["gpu"] is not None and not multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['gpu']}"
        assert torch.cuda.is_available()

    # get task parameters
    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in task_name:
        is_sim = task[:4] == "sim_"
        from dvrk_scripts.constants_dvrk import TASK_CONFIGS
        task_config = TASK_CONFIGS[task]

        dataset_dirs.append(task_config["dataset_dir"])
        num_episodes_list.append(task_config["num_episodes"])
        max_episode_len = max(max_episode_len, task_config["episode_len"])
        camera_names = task_config["camera_names"]
        save_frequnecy = task_config['save_frequency']
        if task_config.get('no_qpos'):
            no_qpos = task_config['no_qpos']
        else:
            no_qpos = False
    
    max_skill_len = (
        args["max_skill_len"] if args["max_skill_len"] is not None else max_episode_len
    )

    # fixed parameters
    state_dim = 20 # changed from 14 to 20 for dvrk  
    lr_backbone = 1e-5
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "action_dim": 20 if policy_level == "low" else 10,
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": args["image_encoder"],
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "multi_gpu": multi_gpu,
            "use_gesture": use_gesture,
            "gesture_dim": gesture_dim,
        }
    
    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": max_episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "log_wandb": log_wandb,
        "use_language": use_language,
        "language_encoder": language_encoder,
        "max_skill_len": max_skill_len,
        "instructor_path": instructor_path,
        "history_len": history_len,
        "history_step_size": history_step_size,
        "hl_margin": hl_margin,
        "no_qpos": no_qpos,
        "use_amp": args.get("use_amp", False),
        "grad_accum_steps": args.get("grad_accum_steps", 1),
        "use_gesture": use_gesture,
        "gesture_dim": gesture_dim,
        "labels_dir": labels_dir,
        "resume_ckpt": resume_ckpt,
    }

    ### load dvrk data to train bc

    print("\n-----------Training low-level policy-----------\n")
    train_dataloader, val_dataloader, stats, _ = load_data_dvrk(
        dataset_dirs[0],
        num_episodes_list[0],
        camera_names,
        batch_size_train,
        batch_size_val,
        task_config,
        chunk_size=args["chunk_size"],
        use_language=use_language,
        use_gesture=use_gesture,
        labels_dir=labels_dir)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # train_bc(train_dataloader, config)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, save_frequnecy, config)

    if best_ckpt_info is not None:
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        # save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
    else:
        print('Warning: no best checkpoint info returned from training')


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "SRT":
        policy = SRTPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicyNoSpatialSoftmax(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def make_fixed_lr_scheduler(optimizer):
    return LambdaLR(optimizer, lambda epoch: 1.0)


def make_scheduler(optimizer, num_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 100, num_training_steps=num_steps
    )
    # scheduler = make_fixed_lr_scheduler(optimizer)
    return scheduler


def get_image(ts, camera_names, crop_top=True, save_dir=None, t=None):
    curr_images = []
    for cam_name in camera_names:
        curr_image = ts.observation["images"][cam_name]

        # Check for 'cam_high' and apply transformation
        if crop_top and cam_name == "cam_high":
            curr_image = crop_resize(curr_image)

        # Swap BGR to RGB
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    # Center crop and resize
    original_size = curr_image.shape[-2:]
    ratio = 0.95
    curr_image = curr_image[
        ...,
        int(original_size[0] * (1 - ratio) / 2) : int(
            original_size[0] * (1 + ratio) / 2
        ),
        int(original_size[1] * (1 - ratio) / 2) : int(
            original_size[1] * (1 + ratio) / 2
        ),
    ]
    curr_image = curr_image.squeeze(0)
    resize_transform = transforms.Resize(original_size, antialias=True)
    curr_image = resize_transform(curr_image)
    curr_image = curr_image.unsqueeze(0)

    if save_dir is not None:
        # Convert torch tensors back to numpy and concatenate for visualization
        concat_images = [
            rearrange(img.cpu().numpy(), "c h w -> h w c")
            for img in curr_image.squeeze(0)
        ]
        concat_image = np.concatenate(concat_images, axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        img_name = (
            "init_visualize.png" if t is None else f"query_frames/{t=}.png"
        )  # save image every query_frequency
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, (concat_image * 255).astype(np.uint8))

    return curr_image



def forward_pass(data, policy, no_qpos=False, use_gesture=False):
    command_embedding = None
    gesture_embedding = None
    if len(data) == 5:
        if use_gesture:
            image_data, qpos_data, action_data, is_pad, gesture_embedding = data
            gesture_embedding = gesture_embedding.cuda()
        else:  # use_language
            image_data, qpos_data, action_data, is_pad, command_embedding = data
            command_embedding = command_embedding.cuda()
    elif no_qpos:
        image_data, action_data, is_pad, command_embedding = data
        qpos_data = None
        command_embedding = command_embedding.cuda()
    else:
        image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda() if qpos_data is not None else None,
        action_data.cuda(),
        is_pad.cuda(),
    )
    if qpos_data is not None:
        return policy(qpos_data, image_data, action_data, is_pad,
                       command_embedding=command_embedding,
                       gesture_embedding=gesture_embedding)
    else:
        return policy(image_data, action_data, is_pad, command_embedding)


def create_ema(nets):
    ema = EMAModel(
    # parameters=nets.parameters(),
    update_after_step =  0,
    inv_gamma = 1.0,
    power = 0.75,
    min_value = 0.0,
    max_value = 0.9999,
    model=nets)
    return ema


def train_bc(train_dataloader, val_dataloader, save_frequnecy, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    log_wandb = config["log_wandb"]
    multi_gpu = config["policy_config"]["multi_gpu"]
    no_qpos = config["no_qpos"]
    use_amp = config.get("use_amp", False)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    use_gesture = config.get("use_gesture", False)
    resume_ckpt = config.get("resume_ckpt", None)

    if use_gesture:
        print("GC-ACT: Gesture conditioning ENABLED")

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    # Load checkpoint for fine-tuning (handles new/resized layers for GC-ACT)
    if resume_ckpt is not None:
        print(f"Loading checkpoint for fine-tuning from {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Filter out keys with shape mismatches (e.g., additional_pos_embed [2,512] -> [3,512])
        model_state = policy.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped_keys.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
            else:
                filtered_state[k] = v
        if skipped_keys:
            print(f"Skipped keys with shape mismatch (will init randomly):")
            for sk in skipped_keys:
                print(f"  {sk}")

        loading_status = policy.load_state_dict(filtered_state, strict=False)
        print(f"Checkpoint loaded. Missing keys: {loading_status.missing_keys}")
        print(f"Unexpected keys: {loading_status.unexpected_keys}")

        # For additional_pos_embed, copy the first 2 rows from checkpoint
        if use_gesture and 'model.additional_pos_embed.weight' in state_dict:
            old_pos_embed = state_dict['model.additional_pos_embed.weight']
            if old_pos_embed.shape[0] == 2:
                with torch.no_grad():
                    policy.model.additional_pos_embed.weight[:2] = old_pos_embed
                print("Copied 2/3 rows of additional_pos_embed from checkpoint (3rd row random)")

    ema = None

    # AMP: GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP (Automatic Mixed Precision) training ENABLED")
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps (effective batch size = batch_size * {grad_accum_steps})")

    optimizer = make_optimizer(policy_class, policy)
    scheduler = make_scheduler(optimizer, num_epochs)

    print(f"save_frequnecy: {save_frequnecy}")
    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 3:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = "y"
        # load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model_state_dict = checkpoint["model_state_dict"]
            # The model was trained on a single gpu, now load onto multiple gpus
            if multi_gpu and not is_multi_gpu_checkpoint(model_state_dict):
                # Add "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k if "model" not in k else f"model.module.{k.split('.', 1)[1]}": v
                    for k, v in model_state_dict.items()
                }
            # The model was trained on multiple gpus, now load onto a single gpu
            elif not multi_gpu and is_multi_gpu_checkpoint(model_state_dict):
                # Remove "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k.replace("module.", "", 1): v for k, v in model_state_dict.items()
                }
            loading_status = policy.deserialize(model_state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(loading_status)
        else:
            print("Not loading checkpoint")
            start_epoch = 0
    else:
        start_epoch = 0

    policy.cuda()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []

            for batch_idx, data in enumerate(val_dataloader):
                with torch.amp.autocast('cuda', enabled=use_amp):
                    forward_dict = forward_pass(data, policy, use_gesture=use_gesture)
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]

            if epoch_val_loss < min_val_loss:
                # Remove previous best checkpoint
                if best_ckpt_info is not None:
                    prev_ckpt_path = os.path.join(ckpt_dir, f"policy_best_epoch_{best_ckpt_info[0]}.ckpt")
                    if os.path.exists(prev_ckpt_path):
                        os.remove(prev_ckpt_path)
                        print(f"Removed previous best checkpoint at epoch {best_ckpt_info[0]}")

                # Update best checkpoint info
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch {epoch}")

                # Save new best checkpoint
                ckpt_path = os.path.join(ckpt_dir, f"policy_best_epoch_{epoch}.ckpt")
                torch.save(
                    {
                        "model_state_dict": policy.serialize(),  # or policy.state_dict() if serialize() is undefined
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

                
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            with torch.amp.autocast('cuda', enabled=use_amp):
                forward_dict = forward_pass(data, policy, no_qpos=no_qpos, use_gesture=use_gesture)
            # backward with gradient accumulation
            loss = forward_dict["loss"] / grad_accum_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_history.append(detach_dict(forward_dict))
        scheduler.step()



        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        epoch_summary["lr"] = np.array(scheduler.get_last_lr()[0])
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.5f} "
        print(summary_string)
        if log_wandb:
            epoch_summary_train = {f"train/{k}": v for k, v in epoch_summary.items()}
            wandb.log(epoch_summary_train, step=epoch)

        if epoch % save_frequnecy == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(
                {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of 1000 epochs
            prune_epoch = epoch - save_frequnecy
            if prune_epoch % 1000 != 0:
                prune_path = os.path.join(
                    ckpt_dir, f"policy_epoch_{prune_epoch}_seed_{seed}.ckpt"
                )
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(
        {
            "model_state_dict": policy.serialize(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )

    return best_ckpt_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', nargs='+', type=str, help='List of task names', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # language correction
    parser.add_argument('--model_type', type=str, default="ACT")
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=1, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=1, required=False)

    parser.add_argument('--policy_level', action='store', type=str, choices=['low'], default='low', help='Which level of policy to train: low', required=False)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--command', action='store', type=str, help='comma-separated list of commands', default='', required=False)
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0, required=False)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--language_encoder', action='store', type=str, choices=['distilbert', 'clip'], default='distilbert', help='Type of language encoder to use: distilbert or clip', required=False)
    parser.add_argument('--max_skill_len', action='store', type=int, help='max_skill_len', required=False)
    parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'resnet18film', 'resnet34film', 'resnet50film','efficientnet_b0film', 'efficientnet_b3film', 'efficientnet_b5film'], help="Which image encoder to use for the BC policy.")
    parser.add_argument('--low_res', action='store', type=int, help='lower resolution by a factor', required=False, default=1)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (FP16) training')
    parser.add_argument('--grad_accum_steps', action='store', type=int, default=1, help='Gradient accumulation steps (effective_bs = batch_size * grad_accum_steps)')
    parser.add_argument('--instructor_path', action='store', type=str, help='instructor_path', required=False)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=2)
    parser.add_argument('--history_step_size', action='store', type=int, help='history_step_size', default=50)
    parser.add_argument('--hl_margin', action='store', type=int, help='the number of timesteps to record before and after language correction', default=100)

    # GC-ACT gesture conditioning
    parser.add_argument('--use_gesture', action='store_true', help='Enable gesture conditioning (GC-ACT)')
    parser.add_argument('--gesture_dim', action='store', type=int, default=10, help='Dimension of gesture one-hot vector')
    parser.add_argument('--labels_dir', action='store', type=str, default=os.path.expanduser('~/data/labels'), help='Path to gesture labels directory')
    parser.add_argument('--resume_ckpt', action='store', type=str, default=None, help='Path to checkpoint to resume/fine-tune from (loaded with strict=False for new layers)')

    main(vars(parser.parse_args()))
