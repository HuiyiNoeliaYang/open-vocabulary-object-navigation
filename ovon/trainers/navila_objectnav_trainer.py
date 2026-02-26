"""
NaVILA ObjectNav Trainer

This trainer integrates NaVILA's VLM-based navigation with OVON's ObjectNav benchmark.
It inherits from VERPIRLNavTrainer to reuse environment infrastructure, but replaces
the RL policy with NaVILA's action selection.
"""

import copy
import http.server
import os
import re
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    inference_mode,
)
from omegaconf import OmegaConf
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from ovon.measurements.nav import FailureModeMeasure, OVONObjectGoalID
from ovon.trainers.ver_trainer import VERPIRLNavTrainer
from ovon.utils.utils import load_pickle
from ovon.utils.visualize.viz import overlay_frame


def _start_live_preview_server(preview_dir: Path, port: int = 8765) -> None:
    """
    Start a background HTTP server that serves an auto-refreshing page showing
    the agent's current view. Open http://localhost:{port} in a browser.
    """
    html = b"""<!DOCTYPE html>
<html>
<head>
  <title>NaVILA Live View</title>
  <style>body{background:#111;margin:0;display:flex;flex-direction:column;align-items:center}
    h2{color:#eee;font-family:monospace;margin:8px}
    img{max-width:100%;border:2px solid #444}</style>
</head>
<body>
  <h2>NaVILA Live View (auto-refresh 0.5s)</h2>
  <img id="frame" src="/current_frame.jpg">
  <script>
    setInterval(function(){
      document.getElementById('frame').src = '/current_frame.jpg?t=' + Date.now();
    }, 500);
  </script>
</body>
</html>"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(preview_dir), **kwargs)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            else:
                super().do_GET()

        def log_message(self, *args):
            pass  # silence server logs

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"[Live Preview] Open http://localhost:{port} in your browser")


def sample_and_pad_images(images: List[Image.Image], num_frames: int = 8, width: int = 640, height: int = 360) -> List[Image.Image]:
    """
    Sample and pad images for NaVILA model input.

    Always keeps the most recent frame as the last element.
    If there are fewer real frames than num_frames, pads the front with black frames.
    If there are more, evenly subsamples the history (excluding the latest frame)
    and appends the latest frame at the end.

    Args:
        images: List of PIL Images (oldest first, most recent last)
        num_frames: Target number of frames
        width: Image width for padding frames
        height: Image height for padding frames

    Returns:
        List of exactly num_frames PIL Images
    """
    if len(images) == 0:
        return [Image.new("RGB", (width, height), color=(0, 0, 0))] * num_frames

    latest_frame = images[-1]
    history = images[:-1]  # everything except the latest

    if len(history) >= num_frames - 1:
        # Subsample history evenly to fill num_frames-1 slots
        indices = np.linspace(0, len(history) - 1, num=num_frames - 1, dtype=int)
        sampled_history = [history[i] for i in indices]
    else:
        # Pad front with black frames, then take all history
        n_pad = (num_frames - 1) - len(history)
        black = Image.new("RGB", (width, height), color=(0, 0, 0))
        sampled_history = [black] * n_pad + list(history)

    return sampled_history + [latest_frame]


@baseline_registry.register_trainer(name="navila_objectnav")
class NaVILAObjectNavTrainer(VERPIRLNavTrainer):
    """
    Trainer for evaluating NaVILA on OVON ObjectNav benchmark.

    Inherits environment setup and metrics from VERPIRLNavTrainer,
    but replaces RL policy with NaVILA's VLM-based action selection.
    """

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """
        Evaluates NaVILA model on ObjectNav episodes.

        Args:
            checkpoint_path: Path to NaVILA checkpoint directory
            writer: Tensorboard writer for logging
            checkpoint_index: Index for logging purposes
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        logger.info(f"Loading NaVILA checkpoint from: {checkpoint_path}")

        # Load NaVILA model
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        local_rank = self.config.habitat_baselines.torch_gpu_id
        device = f"cuda:{local_rank}"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            checkpoint_path, model_name, device_map=device
        )
        model.eval()
        logger.info(f"NaVILA model loaded successfully. Context length: {context_len}")

        # Get config for evaluation
        config = self.config

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

            # Add top_down_map measurement when visualization is needed
            need_viz = (
                len(config.habitat_baselines.eval.video_option) > 0
                or os.environ.get("NAVILA_LIVE_PREVIEW", "") != ""
                or os.environ.get("NAVILA_TRAJ_SAVE_DIR", "") != ""
            )
            if need_viz and not hasattr(config.habitat.task.measurements, "top_down_map"):
                from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
                config.habitat.task.measurements.top_down_map = OmegaConf.structured(
                    TopDownMapMeasurementConfig()
                )

        # Setup video rendering if needed
        if (
            len(config.habitat_baselines.video_render_views) > 0
            and len(self.config.habitat_baselines.eval.video_option) > 0
        ):
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            render_view_uuids = [
                agent_sensors[render_view].uuid
                for render_view in config.habitat_baselines.video_render_views
                if render_view in agent_sensors
            ]
            assert len(render_view_uuids) > 0, (
                "Missing render sensors in agent config: "
                f"{config.habitat_baselines.video_render_views}."
            )
            with read_write(config):
                for render_view_uuid in render_view_uuids:
                    if render_view_uuid not in config.habitat.gym.obs_keys:
                        config.habitat.gym.obs_keys.append(render_view_uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # Load object categories from dataset content files
        import json
        import gzip
        from pathlib import Path

        dataset_path = config.habitat.dataset.data_path
        dataset_dir = Path(dataset_path).parent
        content_dir = dataset_dir / "content"

        logger.info(f"Loading object categories from: {content_dir}")
        object_categories = {}  # Map episode_id -> object_category

        try:
            # Load from all content/*.json.gz files
            content_files = sorted(content_dir.glob("*.json.gz"))
            logger.info(f"Found {len(content_files)} content files")

            for content_file in content_files:
                with gzip.open(content_file, 'rt') as f:
                    content_data = json.load(f)
                    for episode in content_data.get('episodes', []):
                        ep_id = str(episode['episode_id'])
                        # Get object_category from goals or directly
                        if 'goals' in episode and len(episode['goals']) > 0:
                            obj_cat = episode['goals'][0].get('object_category', 'object')
                        elif 'object_category' in episode:
                            obj_cat = episode['object_category']
                        else:
                            obj_cat = 'object'
                        object_categories[ep_id] = obj_cat

            logger.info(f"Loaded {len(object_categories)} episode object categories")
            # Show a few examples
            sample_eps = list(object_categories.items())[:5]
            for ep_id, cat in sample_eps:
                logger.info(f"  Episode {ep_id}: {cat}")
        except Exception as e:
            logger.error(f"Failed to load object categories: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Will use fallback categories")

        # Initialize environments
        self._init_envs(config, is_eval=True)

        # Initialize observation transforms (no actor-critic needed for NaVILA)
        from habitat_baselines.common.obs_transformers import get_active_obs_transforms, apply_obs_transforms_obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)

        # Reset environments
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # Episode tracking
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # Video frames for visualization
        rgb_frames = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        # Live preview server (enabled via NAVILA_LIVE_PREVIEW env var)
        live_preview_dir = os.environ.get("NAVILA_LIVE_PREVIEW", "")
        if live_preview_dir != "":
            preview_path = Path(live_preview_dir)
            preview_path.mkdir(parents=True, exist_ok=True)
            port = int(os.environ.get("NAVILA_LIVE_PORT", "8765"))
            _start_live_preview_server(preview_path, port=port)

        # NaVILA-specific state
        # Unbounded frame history per env — accumulates all frames since episode start.
        # Matches the original NaVILA VLN-CE implementation.
        past_rgbs = [[] for _ in range(self.envs.num_envs)]

        # Rotation queues: when NaVILA requests a turn > 15°, the remaining
        # 15°-step turns are queued here and drained before re-querying the model.
        # Forward moves are never queued (position changes require a fresh query).
        _TURN_STEP_DEG = 15  # degrees per discrete turn action
        rotation_queues = [deque() for _ in range(self.envs.num_envs)]

        # Per-env SPL components for JSON logging (enables offline recomputation)
        _initial_metrics = self.envs.get_metrics()
        _start_dtg = [
            m.get("distance_to_goal", 0.0) for m in _initial_metrics
        ]
        _path_length = [0.0 for _ in range(self.envs.num_envs)]
        _fwd_step = self.config.habitat.simulator.forward_step_size

        # object_categories already loaded from dataset file above

        # Eval episode count
        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        num_successes = 0
        num_total = 0
        json_dict = {}

        # Smart trajectory saving
        # Enable by setting: NAVILA_TRAJ_SAVE_DIR=/path/to/save
        # Tune limits with: NAVILA_TRAJ_MAX_FAIL_PER_CAT (default 4)
        #                   NAVILA_TRAJ_MAX_SUCCESS_PER_CAT (default 1)
        # For val_seen_synonyms (36 categories): 36 × (4+1) = 180 max videos
        _traj_dir = os.environ.get("NAVILA_TRAJ_SAVE_DIR", "")
        _traj_enabled = _traj_dir != ""
        _MAX_FAIL_PER_CAT = int(os.environ.get("NAVILA_TRAJ_MAX_FAIL_PER_CAT", "4"))
        _MAX_SUCCESS_PER_CAT = int(os.environ.get("NAVILA_TRAJ_MAX_SUCCESS_PER_CAT", "1"))
        _traj_fail_counts: Dict[str, int] = defaultdict(int)
        _traj_success_counts: Dict[str, int] = defaultdict(int)
        _traj_frames: List[List] = [[] for _ in range(self.envs.num_envs)]
        if _traj_enabled:
            os.makedirs(_traj_dir, exist_ok=True)
            logger.info(f"[Traj] Smart trajectory saving → {_traj_dir}")
            logger.info(
                f"[Traj] Quota per category: {_MAX_FAIL_PER_CAT} failures, "
                f"{_MAX_SUCCESS_PER_CAT} successes"
            )

        # Main evaluation loop
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            # Collect actions for all environments
            step_data = []
            for env_idx in range(self.envs.num_envs):
                # Drain queued rotations from a prior NaVILA turn > 15°.
                if rotation_queues[env_idx]:
                    action = rotation_queues[env_idx].popleft()
                    step_data.append(action)
                    continue

                # Get object category from pre-loaded dataset
                episode = current_episodes_info[env_idx]
                ep_id = episode.episode_id
                object_category = object_categories.get(ep_id, "object")
                if object_category == "object":
                    logger.warn(f"No category found for episode {ep_id}, using fallback")

                # Generate action from NaVILA model. For turns, queue any remaining
                # steps needed to complete the requested rotation angle.
                with torch.no_grad():
                    action, n_turns = self._get_navila_action(
                        batch,
                        env_idx,
                        past_rgbs[env_idx],
                        object_category,
                        model,
                        tokenizer,
                        image_processor,
                        _TURN_STEP_DEG,
                    )
                if n_turns > 1:
                    action_name = "TURN_LEFT" if action == 2 else "TURN_RIGHT"
                    logger.info(
                        f"Env {env_idx}: Queuing {n_turns - 1} more {action_name} "
                        f"steps to complete {n_turns * _TURN_STEP_DEG}° rotation"
                    )
                    rotation_queues[env_idx].extend([action] * (n_turns - 1))

                step_data.append(action)

            # Track path length: each FORWARD action (1) adds forward_step_size
            for env_idx, act in enumerate(step_data):
                if act == 1:  # FORWARD
                    _path_length[env_idx] += _fwd_step

            # Step all environments
            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            # Update batch
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            # Update episode statistics
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs

            for i in range(n_envs):
                # Check if we should pause this env
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Get target object name — prefer the pre-loaded dataset dict
                # (same source used for model inference) so the category is
                # always available regardless of which sensors/measures are
                # configured.
                ep_id_i = current_episodes_info[i].episode_id
                target_obj = object_categories.get(ep_id_i)
                if target_obj is None:
                    if ObjectGoalSensor.cls_uuid in observations[i]:
                        obj_id = observations[i][ObjectGoalSensor.cls_uuid][0]
                        id_to_name = [
                            "chair",
                            "bed",
                            "potted_plant",
                            "toilet",
                            "tv",
                            "couch",
                        ]
                        target_obj = id_to_name[obj_id]
                    elif OVONObjectGoalID.cls_uuid in infos[i]:
                        obj_id = infos[i][OVONObjectGoalID.cls_uuid]
                        if not hasattr(self, "objectgoal_vocab"):
                            cache = load_pickle(
                                "data/clip_embeddings/ovon_stretch_final_cache.pkl"
                            )
                            self.objectgoal_vocab = sorted(list(cache.keys()))
                        target_obj = self.objectgoal_vocab[obj_id]
                    elif hasattr(current_episodes_info[i], "object_category"):
                        target_obj = current_episodes_info[i].object_category

                # Handle visualization
                need_frame = (
                    len(self.config.habitat_baselines.eval.video_option) > 0
                    or live_preview_dir != ""
                    or _traj_enabled
                )
                if need_frame:
                    # Reconstruct nested top_down_map dict from flattened info keys
                    if "top_down_map.map" in infos[i] and "top_down_map" not in infos[i]:
                        infos[i]["top_down_map"] = {
                            "map": infos[i]["top_down_map.map"],
                            "fog_of_war_mask": infos[i]["top_down_map.fog_of_war_mask"],
                            "agent_map_coord": infos[i]["top_down_map.agent_map_coord"],
                            "agent_angle": infos[i]["top_down_map.agent_angle"],
                        }

                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )

                    # Accumulate traj frames: full-res RGB + top-down map only (no semantic, no text)
                    if _traj_enabled and not dones[i]:
                        rgb_only = {"rgb": observations[i]["rgb"]}
                        traj_frame = observations_to_image(rgb_only, infos[i])
                        _traj_frames[i].append(traj_frame)

                    overlay_dict = self._extract_scalars_from_info(infos[i])
                    if target_obj is not None:
                        overlay_dict["target"] = target_obj
                    frame = overlay_frame(frame, overlay_dict)

                    # Live preview: only write real frames, not the black episode-end frame
                    if live_preview_dir != "" and not dones[i]:
                        Image.fromarray(frame).save(preview_path / "current_frame.jpg")

                    if dones[i] and len(self.config.habitat_baselines.eval.video_option) > 0:
                        # Black frame as visual separator in the recorded video only
                        black_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                        black_frame = overlay_frame(black_frame, overlay_dict)
                        frame = black_frame

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i].append(frame)

                # Episode ended
                if dones[i]:
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    logger.info(
                        f"Success rate: {num_successes/num_total*100:.2f}% "
                        f"({num_successes} out of {num_total})"
                    )

                    # Save episode data to JSON if needed
                    if os.environ.get("OVON_EPISODES_JSON", "") != "" and target_obj is not None:
                        ep_json_dict = {
                            "scene_id": current_episodes_info[i].scene_id,
                            "episode_id": current_episodes_info[i].episode_id,
                            "target": target_obj,
                        }
                        ep_json_dict.update(self._extract_scalars_from_info(infos[i]))

                        # SPL components for offline recomputation
                        ep_json_dict["start_distance_to_goal"] = _start_dtg[i]
                        ep_json_dict["agent_path_length"] = _path_length[i]

                        for key, val in infos[i].items():
                            if FailureModeMeasure.cls_uuid in key:
                                ep_json_dict[key] = val
                        hash_key_str = (
                            f"{current_episodes_info[i].scene_id}"
                            f"_{current_episodes_info[i].episode_id}"
                        )
                        if hash_key_str in json_dict:
                            hash_key_str += "_1"
                        json_dict[hash_key_str] = ep_json_dict

                        # Write incrementally so data is safe if the process crashes
                        _json_path = os.environ.get("OVON_EPISODES_JSON", "")
                        if _json_path:
                            import json as _json
                            with open(_json_path, "w") as _f:
                                _json.dump(json_dict, _f)

                    # Generate video
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        video_metrics = self._extract_scalars_from_info(infos[i])
                        if target_obj is not None:
                            video_metrics[OVONObjectGoalID.cls_uuid] = target_obj

                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=video_metrics,
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )
                        rgb_frames[i] = []

                    # Smart trajectory saving: select by category + outcome quota
                    if _traj_enabled and _traj_frames[i]:
                        is_success = episode_stats.get("success", 0) == 1
                        cat = target_obj or "unknown"
                        dist = episode_stats.get("distance_to_goal", float("inf"))

                        if is_success:
                            count_dict = _traj_success_counts
                            max_quota = _MAX_SUCCESS_PER_CAT
                            mode_tag = "success"
                        else:
                            count_dict = _traj_fail_counts
                            max_quota = _MAX_FAIL_PER_CAT
                            # Classify failure by proximity: close vs far
                            mode_tag = "close_fail" if dist < 0.5 else "far_fail"

                        if count_dict[cat] < max_quota:
                            count_dict[cat] += 1
                            ep_id = current_episodes_info[i].episode_id
                            vid_name = (
                                f"{cat}__{mode_tag}__ep{ep_id}"
                                f"__dtg{dist:.2f}"
                            )
                            generate_video(
                                video_option=["disk"],
                                video_dir=_traj_dir,
                                images=_traj_frames[i],
                                episode_id=vid_name,
                                checkpoint_idx=checkpoint_index,
                                metrics={"success": int(is_success), "dtg": dist},
                                fps=self.config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=[],
                            )
                            logger.info(
                                f"[Traj] Saved {mode_tag} for '{cat}' "
                                f"({count_dict[cat]}/{max_quota}) → {vid_name}"
                            )
                        _traj_frames[i] = []

                    # Handle GFX replay
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

                    # Reset per-env state for next episode
                    past_rgbs[i] = []
                    rotation_queues[i].clear()
                    _path_length[i] = 0.0
                    # Capture start DTG for the next episode (env auto-resets)
                    new_metrics = self.envs.call_at(i, "get_metrics")
                    _start_dtg[i] = new_metrics.get("distance_to_goal", 0.0)

            # Pause completed environments
            (
                self.envs,
                _,  # test_recurrent_hidden_states (not used)
                _,  # not_done_masks (not used)
                current_episode_reward,
                _,  # prev_actions (not used)
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                None,  # test_recurrent_hidden_states
                None,  # not_done_masks
                current_episode_reward,
                None,  # prev_actions
                batch,
                rgb_frames,
            )

            # Also remove paused envs from our tracking lists
            for idx in reversed(envs_to_pause):
                past_rgbs.pop(idx)
                rotation_queues.pop(idx)
                _start_dtg.pop(idx)
                _path_length.pop(idx)
                if _traj_enabled:
                    _traj_frames.pop(idx)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # Aggregate and log statistics
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # Save JSON if needed
        if os.environ.get("OVON_EPISODES_JSON", "") != "":
            import json

            json_file = os.environ["OVON_EPISODES_JSON"]
            with open(json_file, "w") as f:
                json.dump(json_dict, f)
            logger.info(f"Saved episode data to {json_file}")

        self.envs.close()

    def _get_object_category_from_episode(self, episode) -> str:
        """
        Get object category from episode.

        Args:
            episode: Episode object

        Returns:
            Object category string
        """
        # Try direct attribute access (OVONEpisode has this)
        if hasattr(episode, 'object_category') and episode.object_category:
            return episode.object_category

        # Try from goals (goals might be dict or object)
        if hasattr(episode, 'goals') and len(episode.goals) > 0:
            goal = episode.goals[0]
            # Try as object attribute
            if hasattr(goal, 'object_category') and goal.object_category:
                return goal.object_category
            # Try as dict
            if isinstance(goal, dict) and 'object_category' in goal:
                return goal['object_category']

        # Try getting from goals_key attribute (format is scene_category)
        if hasattr(episode, 'goals_key'):
            # goals_key format is typically "{scene_id}_{object_category}"
            parts = episode.goals_key.split('_')
            if len(parts) >= 2:
                # Last part should be the object category
                return '_'.join(parts[1:])  # In case category itself has underscores

        # Fallback - try to infer from episode_id or scene
        logger.warn(f"Could not determine object category for episode {episode.episode_id}")
        logger.warn(f"Episode attributes: {dir(episode)}")
        return "chair"  # Default for debugging

    def _get_navila_action(
        self,
        batch: Dict[str, torch.Tensor],
        env_idx: int,
        past_rgbs: List[Image.Image],
        object_category: str,
        model,
        tokenizer,
        image_processor,
        turn_step_deg: int = 15,
    ):
        """
        Generate an action from NaVILA for one environment step.

        Returns:
            (action, n_turns): action int and number of discrete turn steps needed.
            n_turns is always 1 for forward/stop. For turns, it is
            round(requested_degrees / turn_step_deg), clamped to >= 1.
        """
        # Extract current RGB and convert to PIL Image
        curr_rgb = batch["rgb"][env_idx].cpu().numpy()
        curr_rgb = Image.fromarray(np.uint8(curr_rgb)).convert("RGB")

        # Build frame history
        past_and_current_rgbs = list(past_rgbs) + [curr_rgb]
        num_video_frames = model.config.num_video_frames

        # Sample and pad to required number of frames
        sampled_rgbs = sample_and_pad_images(
            past_and_current_rgbs,
            num_frames=num_video_frames,
            width=curr_rgb.width,
            height=curr_rgb.height,
        )

        # Construct prompt for ObjectNav
        interleaved_images = "<image>\n" * (len(sampled_rgbs) - 1)

        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "Navigate to and find a {object_category}. When you are close to the {object_category}, stop." '
            f"Analyze this series of images to decide your next action. "
            f"Walls, stairs, and elevated obstacles are impassable — do not move forward into them. "
            f"If the path ahead is blocked or the view is not changing, keep turning in the same direction to escape."
        )

        # Build conversation
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images
        images_tensor = process_images(sampled_rgbs, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        # Tokenize
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        # Setup stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.half().to(model.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        logger.info(f"Env {env_idx} - NaVILA output: {outputs}")

        # Parse action and requested rotation degree from text.
        # Turn patterns also capture an optional degree value, e.g. "turn right 45 degree".
        action_patterns = {
            0: re.compile(r"\bstop\b", re.IGNORECASE),
            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
            2: re.compile(r"\bis turn left\b", re.IGNORECASE),
            3: re.compile(r"\bis turn right\b", re.IGNORECASE),
        }
        degree_pattern = re.compile(
            r"turn\s+(?:left|right)\s+(\d+(?:\.\d+)?)\s*degree", re.IGNORECASE
        )

        def map_string_to_action(s):
            for action, pattern in action_patterns.items():
                if pattern.search(s):
                    return action
            return None

        try:
            action = map_string_to_action(outputs)
            if action is None:
                logger.warn(f"Could not parse action from output: {outputs}. Defaulting to FORWARD.")
                action = 1
        except Exception:
            logger.warn(f"Error parsing action from output: {outputs}. Defaulting to FORWARD.")
            action = 1

        # Determine how many discrete turn steps are needed.
        n_turns = 1
        if action in (2, 3):  # turn action
            deg_match = degree_pattern.search(outputs)
            if deg_match:
                requested_deg = float(deg_match.group(1))
                n_turns = max(1, round(requested_deg / turn_step_deg))

        total_deg = n_turns * turn_step_deg
        action_names = {0: "STOP", 1: "FORWARD 25cm", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
        logger.info(
            f"Env {env_idx}: {action_names.get(action, action)}"
            + (f" {total_deg}° ({n_turns} steps)" if action in (2, 3) else " 25cm")
        )

        # Update frame history for next step
        past_rgbs.append(curr_rgb)

        return action, n_turns
