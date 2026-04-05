# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from aliengo_gym import MINI_GYM_ROOT_DIR
from aliengo_gym.envs.base.base_task import BaseTask
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from aliengo_gym.utils.terrain import Terrain
# # from .legged_robot_config import Cfg
from .legged_robot_config import BaseCfg

#from aliengo_gym.utils.terrain_new import Terrain
# from .ll_config import BaseCfg

# import rclpy
# from isaac_bridge.camera_node import CameraPublisher

import zmq
import time
import pickle

class LeggedRobot(BaseTask):
    def __init__(self, cfg: BaseCfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        self.robot_indices = []
        self.guitar_indices = []
        self.camera_handles = []

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

        ##################
        self.vel_x_limit = self.cfg.commands.lin_vel_x
        #self.defer_reset = False
        #self.defer_command_resample = False

        # rclpy.init()
        # self.camera_node = CameraPublisher()

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Store current torques as "last" BEFORE computing new ones
        # self.last_last_torques[:] = self.last_torques[:]
        self.last_torques[:] = self.torques[:]
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            ##################### TORQUE SMOOTHING ############################
            ################## Moving Averages (sliding window) ###############
            # raw_torques = self._compute_torques(self.actions)
            #
            # # Shift history
            # self.torque_history = torch.roll(self.torque_history, shifts=1, dims=0)
            # self.torque_history[0] = raw_torques
            #
            # # Moving average
            # self.torques = torch.mean(self.torque_history, dim=0)

            # EMA (exponential moving average)
            # raw_torques = self._compute_torques(self.actions)
            #
            # beta = 0.7
            # self.smoothed_torques = beta * self.smoothed_torques + (1 - beta) * raw_torques
            #
            # self.torques = self.smoothed_torques

            ###################################################################

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.step_graphics(self.sim)
        # if self.record_now:
            # self.gym.step_graphics(self.sim)
            # self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        #=============== FALL RECOVERY ===============
        fallen = self.root_states[:, 2] < 0.15

        if torch.any(fallen):
            self.root_states[fallen, 2] = 0.3
            self.root_states[fallen, 7:13] = 0.0

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_states)
            )

            # refresh after modifying state
            self.gym.refresh_actor_root_state_tensor(self.sim)

        ###################################################

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # CAMERA + IMU + ROS PUBLISH HERE
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # if self.common_step_counter % 10 == 0:
        if self.common_step_counter % 3 == 0:  # ~10–15 Hz:

            # ALWAYS update graphics first
            self.gym.step_graphics(self.sim)

            # render cameras
            self.gym.render_all_camera_sensors(self.sim)

            env = self.envs[0]
            cam = self.camera_handles[0]

            # CPU image API (SAFE)
            rgb = self.gym.get_camera_image(
                self.sim, env, cam, gymapi.IMAGE_COLOR
            )

            depth = self.gym.get_camera_image(
                self.sim, env, cam, gymapi.IMAGE_DEPTH
            )

            # reshape
            # rgb = rgb.reshape((480, 640, 4))[:, :, :3]
            # depth = -depth.reshape((480, 640))

            H = self.camera_intrinsics["height"]
            W = self.camera_intrinsics["width"]

            rgb = rgb.reshape((H, W, 4))[:, :, :3]
            depth = depth.reshape((H, W))
            depth = np.abs(depth)

            # IMU
            root_states = self.root_states

            pos = root_states[0, 0:3]
            quat = root_states[0, 3:7]
            lin_vel = root_states[0, 7:10]
            ang_vel = root_states[0, 10:13]

            if not hasattr(self, "prev_lin_vel"):
                self.prev_lin_vel = lin_vel

            dt = self.dt
            lin_acc = (lin_vel - self.prev_lin_vel) / dt
            self.prev_lin_vel = lin_vel

            # SEND
            data = {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": self.camera_intrinsics,

                "imu": {
                    "quat": quat.cpu().numpy().tolist(),
                    "ang_vel": ang_vel.cpu().numpy().tolist(),
                    "lin_acc": lin_acc.cpu().numpy().tolist()
                },

                "pose": {
                    "position": pos.cpu().numpy().tolist(),
                    "orientation": quat.cpu().numpy().tolist(),
                    "linear_velocity": lin_vel.cpu().numpy().tolist(),
                    "angular_velocity": ang_vel.cpu().numpy().tolist()
                },

                "timestamp": float(time.time())
            }

            self.zmq_socket.send(pickle.dumps(data))

            # self.camera_node.publish(rgb, depth)
            # rclpy.spin_once(self.camera_node, timeout_sec=0.0)
        # ========================================

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()

        # Compute Energy Consumption
        self.energy_consume = torch.sum(torch.abs(self.dof_vel * self.torques), dim=1).detach().clone()
        self.cot = torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1) / (21.5 * 9.81 * torch.norm(self.base_lin_vel[:, 0:2], dim=1))
        self.dof_acc[:] = (self.dof_vel[:] - self.last_dof_vel[:]) / self.dt

        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # Conditionally reset
        #if not self.defer_reset:
        #    self.reset_idx(env_ids)

        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        # self.last_torques[:] = self.torques[:]
        # self.last_last_torques[:] = self.last_torques[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # self._render_headless()

    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
    #                                dim=1)
    #     self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    #     if self.cfg.rewards.use_terminal_body_height:
    #         self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
    #                                < self.cfg.rewards.terminal_body_height
    #         self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    def check_termination(self):
        """ SLAM mode: detect events but DO NOT reset """

        # detect contact events
        contact_termination = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )

        # detect timeout (not really needed for SLAM)
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length

        # detect low body height (fallen robot)
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(
                self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
            ) < self.cfg.rewards.terminal_body_height
        else:
            self.body_height_buf = torch.zeros_like(contact_termination)

        # DO NOT RESET
        self.reset_buf = torch.zeros_like(contact_termination)

        # OPTIONAL: store for logging/debug
        self.collision_buf = contact_termination
        self.fallen_buf = self.body_height_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # reset robot states
        self._resample_commands(env_ids)
        #if not getattr(self, "defer_command_resample", False):
        #    self._resample_commands(env_ids)
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            # for key in self.episode_sums.keys():
            #     self.extras["train/episode"]['rew_' + key] = torch.mean(
            #         self.episode_sums[key][train_env_ids])
            #     self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())
        if self.cfg.commands.command_curriculum:
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]
            self.extras["train/episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            self.extras["train/episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            self.extras["train/episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            self.extras["train/episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            self.extras["train/episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            self.extras["train/episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            self.extras["train/episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            self.extras["train/episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            self.extras["train/episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
            self.extras["train/episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
            self.extras["train/episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            self.extras["train/episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            self.extras["train/episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            self.extras["train/episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            self.extras["train/episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            self.extras["train/episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            if self.cfg.commands.num_commands > 9:
                self.extras["train/episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                self.extras["train/episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
                                                                           curriculum.weights.shape[0]

            self.extras["train/episode"]["min_action"] = torch.min(self.actions)
            self.extras["train/episode"]["max_action"] = torch.max(self.actions)

            self.extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # base position
        self.root_states[env_ids] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            #self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)

        #self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            #self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                             :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        # if self.cfg.env.observe_command and not self.cfg.env.observe_height_command:
        #     self.obs_buf = torch.cat((self.projected_gravity,
        #                               self.commands[:, :3] * self.commands_scale,
        #                               (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                               self.dof_vel * self.obs_scales.dof_vel,
        #                               self.actions
        #                               ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                                 :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        # if self.cfg.env.observe_desired_contact_states:
        #     self.obs_buf = torch.cat((self.obs_buf,
        #                               self.desired_contact_states), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # print(f"obs shape: {self.obs_buf.shape}")

        # build privileged obs

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        # print(f"priv obs shape: {self.privileged_obs_buf.shape}")
        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target) \

    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        #if not self.defer_command_resample:
        #    env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        #    self._resample_commands(env_ids)

        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base yaws
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []

    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_last_torques = torch.zeros_like(self.torques)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])


        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )


        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_aliengo.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

        # for torque smoothing
        # self.torque_history_len = 3  # N = 3–5 is ideal
        # self.torque_history = torch.zeros(
        #     self.torque_history_len,
        #     self.num_envs,
        #     self.num_dof,
        #     device=self.device
        # )
        # self.smoothed_torques = torch.zeros_like(self.torques)

        # ================= BOUNDARY SETUP =================
        # relative to env origin
        self.boundary_x = torch.tensor([-3.0, 3.0], device=self.device)
        self.boundary_y = torch.tensor([-3.0, 3.0], device=self.device)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                           self.cfg.commands.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from aliengo_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _spawn_wall_marker(self, env_handle, env_id, x, y, z, normal):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(x, y, z)

        # rotate to face inward
        if normal == "x+":
            pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        elif normal == "x-":
            pose.r = gymapi.Quat.from_euler_zyx(np.pi, 0, 0)
        elif normal == "y+":
            pose.r = gymapi.Quat.from_euler_zyx(np.pi/2, 0, 0)
        elif normal == "y-":
            pose.r = gymapi.Quat.from_euler_zyx(-np.pi/2, 0, 0)

        marker_handle = self.gym.create_actor(
            env_handle,
            self.marker_asset,
            pose,
            "marker",
            env_id,
            0,
            0
        )

        return marker_handle

    def sample_terrain_aware_position(self, tile, origin_x, origin_y, existing_positions, radius):

        H, W = tile.shape
        scale = self.cfg.terrain.horizontal_scale

        min_dist = 1.0
        max_tries = 100

        # convert object radius → grid clearance
        clearance_cells = int(radius / scale) + 1

        for _ in range(max_tries):

            xi = np.random.randint(clearance_cells, H - clearance_cells)
            yi = np.random.randint(clearance_cells, W - clearance_cells)

            # must be free space
            if tile[xi, yi] != 0:
                continue

            # strong clearance (prevents penetration)
            if np.any(tile[
                xi-clearance_cells:xi+clearance_cells,
                yi-clearance_cells:yi+clearance_cells
            ] > 0):
                continue

            # prefer near walls but not too close
            near_wall = (
                tile[xi+3, yi] > 0 or tile[xi-3, yi] > 0 or
                tile[xi, yi+3] > 0 or tile[xi, yi-3] > 0
            )

            if not near_wall:
                if np.random.rand() < 0.7:
                    continue

            # convert to world
            # x = origin_x + (xi - H//2) * scale
            # y = origin_y + (yi - W//2) * scale

            x = xi * scale
            y = yi * scale

            # spacing constraint
            if any(np.linalg.norm([x - px, y - py]) < min_dist for px, py in existing_positions):
                continue

            return x, y

        # fallback random safe position
        for _ in range(50):
            xi = np.random.randint(5, H-5)
            yi = np.random.randint(5, W-5)

            if tile[xi, yi] == 0:
                x = origin_x + (xi - H//2) * scale
                y = origin_y + (yi - W//2) * scale

                if not any(np.linalg.norm([x - px, y - py]) < 0.8 for px, py in existing_positions):
                    return x, y

        # final fallback (last resort)
        return origin_x + np.random.uniform(-1, 1), origin_y + np.random.uniform(-1, 1)

    def face_robot_quat(self, x, y, origin_x, origin_y):
        yaw = np.arctan2(origin_y - y, origin_x - x)
        return gymapi.Quat.from_euler_zyx(yaw, 0, 0)


    def upright_quat_facing_robot(self, x, y, origin_x, origin_y):
        yaw = np.arctan2(origin_y - y, origin_x - x)
        return gymapi.Quat.from_euler_zyx(yaw, 0, 1.57)

    def get_terrain_height(self, x, y, tile, origin_x, origin_y):
        scale = self.cfg.terrain.horizontal_scale
        xi = int((x - origin_x) / scale + tile.shape[0] // 2)
        yi = int((y - origin_y) / scale + tile.shape[1] // 2)

        xi = np.clip(xi, 0, tile.shape[0]-1)
        yi = np.clip(yi, 0, tile.shape[1]-1)

        return tile[xi, yi] * self.cfg.terrain.vertical_scale

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        ##################################
        # marker_opts = gymapi.AssetOptions()
        # marker_opts.fix_base_link = True
        # marker_opts.disable_gravity = True
        #
        # # thin wall poster
        # self.marker_asset = self.gym.create_box(
        #     self.sim,
        #     0.02,   # thickness
        #     0.4,    # height
        #     0.4,    # width
        #     marker_opts
        # )
        guitar_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/guitar_new"
        guitar_asset_file = "guitar.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True          # static
        asset_options.disable_gravity = True        # no falling
        asset_options.collapse_fixed_joints = True
        asset_options.use_mesh_materials = True     # use textures if available

        self.guitar_asset = self.gym.load_asset(
            self.sim,
            guitar_asset_root,
            guitar_asset_file,
            asset_options
        )
        assert self.guitar_asset is not None, "Guitar asset NOT loaded"

        shoes_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/shoes"
        shoes_asset_file = "shoes.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.shoes_asset = self.gym.load_asset(
            self.sim,
            shoes_asset_root,
            shoes_asset_file,
            asset_options
        )
        assert self.shoes_asset is not None, "Shoes asset NOT loaded"
        # print("shoes asset loaded")

        mug_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/mug"
        mug_asset_file = "nasa-mug.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.mug_asset = self.gym.load_asset(
            self.sim,
            mug_asset_root,
            mug_asset_file,
            asset_options
        )
        assert self.mug_asset is not None, "Nasa mug asset NOT loaded"

        chair_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/office_chair"
        chair_asset_file = "OfficeChair.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.chair_asset = self.gym.load_asset(
            self.sim,
            chair_asset_root,
            chair_asset_file,
            asset_options
        )
        assert self.chair_asset is not None, "Chair asset NOT loaded"

        bag_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bagpack"
        bag_asset_file = "bagpack.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.bag_asset = self.gym.load_asset(
            self.sim,
            bag_asset_root,
            bag_asset_file,
            asset_options
        )
        assert self.bag_asset is not None, "Bagpack asset NOT loaded"

        person_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/person"
        person_asset_file = "Capoeira.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.person_asset = self.gym.load_asset(
            self.sim,
            person_asset_root,
            person_asset_file,
            asset_options
        )
        assert self.person_asset is not None, "Person asset NOT loaded"

        bicycle_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bicycle"
        bicycle_asset_file = "bicycle.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.bicycle_asset = self.gym.load_asset(
            self.sim,
            bicycle_asset_root,
            bicycle_asset_file,
            asset_options
        )
        assert self.bicycle_asset is not None, "Bicycle asset NOT loaded"

        motorcycle_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/motorcycle"
        motorcycle_asset_file = "Honda CB650R .urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.motorcycle_asset = self.gym.load_asset(
            self.sim,
            motorcycle_asset_root,
            motorcycle_asset_file,
            asset_options
        )
        assert self.motorcycle_asset is not None, "Motorcycle asset NOT loaded"

        firehydrant_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/fire_hydrant"
        firehydrant_asset_file = "model.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.firehydrant_asset = self.gym.load_asset(
            self.sim,
            firehydrant_asset_root,
            firehydrant_asset_file,
            asset_options
        )
        assert self.firehydrant_asset is not None, "Fire Hydrant asset NOT loaded"

        couch_asset_root = "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/couch"
        couch_asset_file = "sofa_low.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.couch_asset = self.gym.load_asset(
            self.sim,
            couch_asset_root,
            couch_asset_file,
            asset_options
        )
        assert self.couch_asset is not None, "Couch asset NOT loaded"

        OBJECT_RADII = {
            "guitar": 0.4,
            "shoes": 0.3,
            "mug": 0.25,
            "chair": 0.8,
            "bag": 0.4,
            "person": 0.35,        # human footprint ~ shoulder width
            "couch": 1.0,          # wide furniture
            "bicycle": 0.8,        # length dominates footprint
            "motorcycle": 1.0,     # similar to bike but bulkier
            "fire_hydrant": 0.25   # small object
        }

        for i in range(self.num_envs):

            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # CAMERA
            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            camera_props = gymapi.CameraProperties()
            camera_props.width = 640
            camera_props.height = 480
            camera_props.enable_tensors = False #True
            camera_props.horizontal_fov = 87.0  # closer to depth sensor

            # CameraInfo (intrinsics)
            W = camera_props.width
            H = camera_props.height
            fov = camera_props.horizontal_fov * np.pi / 180.0

            fx = W / (2 * np.tan(fov / 2))
            fy = fx
            cx = W / 2.0
            cy = H / 2.0

            self.camera_intrinsics = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": W,
                "height": H
            }

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)

            # attach to robot (like base)
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                anymal_handle,  # robot actor
                gymapi.Transform(
                    gymapi.Vec3(0.3, 0.0, 0.2),         # forward, center, height
                    gymapi.Quat.from_euler_zyx(0, 0, 0)
                ),
                gymapi.FOLLOW_TRANSFORM
            )

            self.camera_handles.append(camera_handle)

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # OBJECT PLACEMENT
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            env_origin = self.env_origins[i]

            origin_x = float(env_origin[0].cpu())
            origin_y = float(env_origin[1].cpu())
            origin_z = env_origin[2].item()

            tile = self.terrain.height_field_raw

            final_positions = []

            gx, gy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["guitar"])
            final_positions.append((gx, gy))

            sx, sy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["shoes"])
            final_positions.append((sx, sy))

            mx, my = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["mug"])
            final_positions.append((mx, my))

            cx, cy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["chair"])
            final_positions.append((cx, cy))

            bx, by = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["bag"])
            final_positions.append((bx, by))

            px, py = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["person"])
            final_positions.append((px, py))

            cox, coy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["couch"])
            final_positions.append((cox, coy))

            mcx, mcy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["motorcycle"])
            final_positions.append((mcx, mcy))

            bcx, bcy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["bicycle"])
            final_positions.append((bcx, bcy))

            fx, fy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["fire_hydrant"])
            final_positions.append((fx, fy))


            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # GUITAR
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            guitar_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/guitar_new/textures/guitar_BaseColor.png"
            )

            # gx, gy = get_next_valid()
            gz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(gx, gy, gz + 0.4)
            # pose.p = gymapi.Vec3(gx, gy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(gx, gy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57 + 0.785)

            guitar_handle = self.gym.create_actor(
                env_handle, self.guitar_asset, pose, "guitar", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, guitar_handle))
            num_guitar_bodies = self.gym.get_actor_rigid_body_count(env_handle, guitar_handle)

            for b in range(num_guitar_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    guitar_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    guitar_texture
                )

            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     guitar_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     guitar_texture
            # )

            # self.gym.set_rigid_body_color(
            #     env_handle,
            #     guitar_handle,
            #     0,
            #     gymapi.MESH_VISUAL,
            #     color
            # )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # SHOES
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            shoes_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/shoes/textures/nb574.jpg"
            )

            # sx, sy = get_next_valid()
            sz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(sx, sy, sz - 3)
            # pose.r = self.face_robot_quat(sx, sy, origin_x, origin_y)
            # pose.p = gymapi.Vec3(sx, sy, origin_z + 0.02)
            # pose.r = self.face_robot_quat(sx, sy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

            shoes_handle = self.gym.create_actor(
                env_handle, self.shoes_asset, pose, "shoes", i, 0, 0
            )

            num_shoe_bodies = self.gym.get_actor_rigid_body_count(env_handle, shoes_handle)

            for b in range(num_shoe_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    shoes_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    shoes_texture
                )

            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     shoes_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     shoes_texture
            # )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # NASA MUG
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            mug_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/mug/textures/Mug_D.tga.png"
            )

            # mx, my = get_next_valid()
            mz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(mx, my, mz - 2.5)
            # pose.r = self.face_robot_quat(mx, my, origin_x, origin_y)
            # pose.p = gymapi.Vec3(mx, my, origin_z + 0.02)
            # pose.r = self.face_robot_quat(mx, my, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)

            mug_handle = self.gym.create_actor(
                env_handle, self.mug_asset, pose, "mug", i, 0, 0
            )

            num_mug_bodies = self.gym.get_actor_rigid_body_count(env_handle, mug_handle)

            for b in range(num_mug_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    mug_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    mug_texture
                )

            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     mug_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     mug_texture
            # )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # OFFICE CHAIR
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            chair_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/office_chair/textures/OfficeChair_OfficeChair_Main_BaseColor.png"
            )

            # cx, cy = get_next_valid()
            cz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(cx, cy, cz - 4.0)
            # pose.r = self.face_robot_quat(cx, cy, origin_x, origin_y)
            # pose.p = gymapi.Vec3(cx, cy, origin_z + 0.02)
            # pose.r = self.face_robot_quat(cx, cy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)

            chair_handle = self.gym.create_actor(
                env_handle, self.chair_asset, pose, "chair", i, 0, 0
            )

            num_chair_bodies = self.gym.get_actor_rigid_body_count(env_handle, chair_handle)

            for b in range(num_chair_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    chair_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    chair_texture
                )

            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     chair_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     chair_texture
            # )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # BAGPACK
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            bag_texture = self.gym.create_texture_from_file(
                 self.sim,
                 "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bagpack/textures/Male_Backpack_Metropolis_002_D.png"
             )

            # bx, by = get_next_valid()
            bz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(bx, by, bz - 2.0)
            # pose.r = self.face_robot_quat(bx, by, origin_x, origin_y)
            # pose.p = gymapi.Vec3(bx, by, origin_z + 0.02)
            # pose.r = self.face_robot_quat(bx, by, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)

            bag_handle = self.gym.create_actor(
                env_handle, self.bag_asset, pose, "bagpack", i, 0, 0
            )

            num_bag_bodies = self.gym.get_actor_rigid_body_count(env_handle, bag_handle)

            for b in range(num_bag_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    bag_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    bag_texture
                )

            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     bag_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     bag_texture
            # )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # PERSON
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            person_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/person/textures/Ch38_1001_Diffuse.png"
            )

            # px, py = get_next_valid()
            pz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(px, py, pz + 0.1)
            # pose.p = gymapi.Vec3(gx, gy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(gx, gy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)

            person_handle = self.gym.create_actor(
                env_handle, self.person_asset, pose, "person", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, person_handle))
            num_person_bodies = self.gym.get_actor_rigid_body_count(env_handle, person_handle)

            for b in range(num_person_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    person_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    person_texture
                )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # COUCH
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            couch_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/couch/textures/lambert1_Base_Color.png"
            )

            # cox, coy = get_next_valid()
            coz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(cox, coy, coz + 0.1)
            # pose.p = gymapi.Vec3(cox, coy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(cox, coy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 0)

            couch_handle = self.gym.create_actor(
                env_handle, self.couch_asset, pose, "couch", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, couch_handle))
            num_couch_bodies = self.gym.get_actor_rigid_body_count(env_handle, couch_handle)

            for b in range(num_couch_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    couch_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    couch_texture
                )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # BICYCLE
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            bicycle_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bicycle/textures/bicycle_bicycle_BaseColor.jpeg"
            )

            # bcx, bcy = get_next_valid()
            bcz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(bcx, bcy, bcz + 1.0)
            # pose.p = gymapi.Vec3(bcx, bcy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(bcx, bcy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 0)

            bicycle_handle = self.gym.create_actor(
                env_handle, self.bicycle_asset, pose, "bicycle", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, bicycle_handle))
            num_bicycle_bodies = self.gym.get_actor_rigid_body_count(env_handle, bicycle_handle)

            for b in range(num_bicycle_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    bicycle_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    bicycle_texture
                )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # MOTORCYCLE
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            motorcycle_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bicycle/textures/bicycle_bicycle_BaseColor.jpeg"
            )

            # mcx, mcy = get_next_valid()
            mcz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(mcx, mcy, mcz + 0.3)
            # pose.p = gymapi.Vec3(mcx, mcy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(mcx, mcy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 0)

            motorcycle_handle = self.gym.create_actor(
                env_handle, self.motorcycle_asset, pose, "motorcycle", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, motorcycle_handle))
            num_motorcycle_bodies = self.gym.get_actor_rigid_body_count(env_handle, motorcycle_handle)

            for b in range(num_motorcycle_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    motorcycle_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    motorcycle_texture
                )

            # XXXXXXXXXXXXXXXXXXXXXXXXXX
            # FIRE HYDRANT
            # XXXXXXXXXXXXXXXXXXXXXXXXXX

            fire_hydrant_texture = self.gym.create_texture_from_file(
                self.sim,
                "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/fire_hydrant/textures/Red_Paint_Top_albedo.jpg"
            )

            # fx, fy = get_next_valid()
            fz = env_origin[2].item()

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(fx, fy, fz + 0.3)
            # pose.p = gymapi.Vec3(fx, fy, origin_z + 0.02)
            # pose.r = self.upright_quat_facing_robot(fx, fy, origin_x, origin_y)
            pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 0)

            fire_hydrant_handle = self.gym.create_actor(
                env_handle, self.firehydrant_asset, pose, "fire_hydrant", i, 0, 0
            )

            # print(self.gym.get_actor_rigid_body_names(env_handle, fire_hydrant_handle))
            num_fire_hydrant_bodies = self.gym.get_actor_rigid_body_count(env_handle, fire_hydrant_handle)

            for b in range(num_fire_hydrant_bodies):
                self.gym.set_rigid_body_texture(
                    env_handle,
                    fire_hydrant_handle,
                    b,
                    gymapi.MESH_VISUAL,
                    fire_hydrant_texture
                )

            # STORE GT POSITIONS
            if not hasattr(self, "object_positions"):
                self.object_positions = {}

            self.object_positions[i] = {
                "guitar": (gx, gy),
                "shoes": (sx, sy),
                "mug": (mx, my),
                "chair": (cx, cy),
                "bag": (bx, by),
                "person": (px, py),
                "couch": (cox, coy),
                "bicycle": (bcx, bcy),
                "motorcycle": (mcx, mcy),
                "fire_hydrant": (fx, fy),
            }

            # env_origin = self.env_origins[i]
            #
            # origin_x = float(env_origin[0].cpu())
            # origin_y = float(env_origin[1].cpu())
            #
            # room_size_x = self.cfg.terrain.terrain_length
            # room_size_y = self.cfg.terrain.terrain_width
            #
            #
            # # =========================
            # # GUITAR
            # # =========================
            # texture = self.gym.create_texture_from_file(
            #     self.sim,
            #     "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/guitar_new/textures/guitar_BaseColor.png"
            # )
            #
            # # gx, gy = self.sample_position(origin_x, origin_y, room_size_x, room_size_y)
            # gx, gy = self.sample_non_overlapping(origin_x, origin_y, room_size_x, room_size_y, placed_positions, min_dist=2.0)
            # placed_positions.append((gx, gy))
            #
            # z = env_origin[2].item()
            #
            # pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(gx, gy, z - 1.3)
            # pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57 + 0.785)
            #
            # # pose = gymapi.Transform()
            # # pose.p = gymapi.Vec3(gx, gy, 1.0)   # higher for visibility
            # # pose.r = gymapi.Quat.from_euler_zyx(np.pi, np.pi, np.pi)
            #
            # guitar_handle = self.gym.create_actor(
            #     env_handle,
            #     self.guitar_asset,
            #     pose,
            #     "guitar",
            #     i,
            #     0, # enable collison
            #     0
            # )
            #
            # # rb_states = self.gym.get_actor_rigid_body_states(
            # #     env_handle,
            # #     guitar_handle,
            # #     gymapi.STATE_POS
            # # )
            # # print("INIT Z:", rb_states['pose']['p'][0][2])
            #
            # # color (if texture missing)
            # # color = gymapi.Vec3(0.6, 0.3, 0.1)
            # # self.gym.set_rigid_body_color(
            # #     env_handle,
            # #     guitar_handle,
            # #     0,
            # #     gymapi.MESH_VISUAL,
            # #     color
            # # )
            #
            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     guitar_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     texture
            # )
            #
            # guitar_idx = self.gym.get_actor_index(env_handle, guitar_handle, gymapi.DOMAIN_SIM)
            # self.guitar_indices.append(guitar_idx)
            #
            # # store correct position
            # guitar_pos = np.array([gx, gy])
            #
            #
            # # =========================
            # # SHOES
            # # =========================
            # texture = self.gym.create_texture_from_file(
            #     self.sim,
            #     "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/shoes/textures/nb574.jpg"
            # )
            #
            # # SINGLE FLOWERPOT
            # # sx, sy = self.sample_position(origin_x, origin_y, room_size_x, room_size_y)
            # sx, sy = self.sample_non_overlapping(origin_x, origin_y, room_size_x, room_size_y, placed_positions, min_dist=1.0)
            # placed_positions.append((sx, sy))
            #
            # z = env_origin[2].item()
            #
            # pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(sx, sy, z - 1.2)   # small offset so it sits ON ground
            # pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
            #
            # shoes_handle = self.gym.create_actor(
            #     env_handle,
            #     self.shoes_asset,
            #     pose,
            #     "shoes",
            #     i,
            #     0, # 1(disable collisions)
            #     0
            # )
            #
            # # self.gym.set_actor_rigid_body_properties(
            # #     env_handle,
            # #     shoes_handle,
            # #     self.gym.get_actor_rigid_body_properties(env_handle, pot_handle),
            # #     recomputeInertia=False
            # # )
            #
            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     shoes_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     texture
            # )
            #
            # # # color
            # # self.gym.set_rigid_body_color(
            # #     env_handle,
            # #     shoes_handle,
            # #     0,
            # #     gymapi.MESH_VISUAL,
            # #     gymapi.Vec3(0.8, 0.2, 0.2)
            # # )
            #
            # # =========================
            # # NASA Mug
            # # =========================
            # mug_texture = self.gym.create_texture_from_file(
            #     self.sim,
            #     "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/mug/textures/Mug_D.tga.png"
            # )
            #
            # # mx, my = self.sample_position(origin_x, origin_y, room_size_x, room_size_y)
            # mx, my = self.sample_non_overlapping(origin_x, origin_y, room_size_x, room_size_y, placed_positions, min_dist=1.0)
            # placed_positions.append((mx, my))
            #
            # pose = gymapi.Transform()
            # mz = env_origin[2].item()
            # pose.p = gymapi.Vec3(mx, my, mz-2.0)   # keep high for visibility
            # pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)
            #
            # mug_handle = self.gym.create_actor(
            #     env_handle,
            #     self.mug_asset,
            #     pose,
            #     "mug",
            #     i,
            #     0, # 1(disable collisions)
            #     0
            # )
            #
            # # self.gym.set_actor_rigid_body_properties(
            # #     env_handle,
            # #     shoes_handle,
            # #     self.gym.get_actor_rigid_body_properties(env_handle, pot_handle),
            # #     recomputeInertia=False
            # # )
            #
            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     mug_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     mug_texture
            # )
            #
            # # color
            # # self.gym.set_rigid_body_color(
            # #     env_handle,
            # #     flowerpot_handle,
            # #     0,
            # #     gymapi.MESH_VISUAL,
            # #     gymapi.Vec3(0.8, 0.2, 0.2)
            # # )
            #
            # # =========================
            # # Office Chair
            # # =========================
            # chair_texture = self.gym.create_texture_from_file(
            #     self.sim,
            #     "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/office_chair/textures/OfficeChair_OfficeChair_Main_BaseColor.png"
            # )
            #
            # # mx, my = self.sample_position(origin_x, origin_y, room_size_x, room_size_y)
            # cx, cy = self.sample_non_overlapping(origin_x, origin_y, room_size_x, room_size_y, placed_positions, min_dist=1.0)
            # placed_positions.append((cx, cy))
            #
            # pose = gymapi.Transform()
            # cz = env_origin[2].item()
            # pose.p = gymapi.Vec3(cx, cy, cz-2.0)   # keep high for visibility
            # pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)
            #
            # chair_handle = self.gym.create_actor(
            #     env_handle,
            #     self.chair_asset,
            #     pose,
            #     "chair",
            #     i,
            #     0, # 1(disable collisions)
            #     0
            # )
            #
            # # self.gym.set_actor_rigid_body_properties(
            # #     env_handle,
            # #     chair_handle,
            # #     self.gym.get_actor_rigid_body_properties(env_handle, chair_handle),
            # #     recomputeInertia=False
            # # )
            #
            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     chair_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     chair_texture
            # )
            #
            # # =========================
            # # Bagpack
            # # =========================
            # bag_texture = self.gym.create_texture_from_file(
            #      self.sim,
            #      "/home/anubhav1772/Documents/lab/wtw-aliengo/resources/objects/bagpack/textures/Male_Backpack_Metropolis_002_D.png"
            #  )
            #
            # # bx, by = self.sample_position(origin_x, origin_y, room_size_x, room_size_y)
            # bx, by = self.sample_non_overlapping(origin_x, origin_y, room_size_x, room_size_y, placed_positions, min_dist=1.0)
            # placed_positions.append((bx, by))
            #
            # pose = gymapi.Transform()
            # bz = env_origin[2].item()
            # pose.p = gymapi.Vec3(bx, by, bz-2.0)   # keep high for visibility
            # pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)
            #
            # bag_handle = self.gym.create_actor(
            #     env_handle,
            #     self.bag_asset,
            #     pose,
            #     "bagpack",
            #     i,
            #     0, # 1(disable collisions)
            #     0
            # )
            #
            # # self.gym.set_actor_rigid_body_properties(
            # #     env_handle,
            # #     bag_handle,
            # #     self.gym.get_actor_rigid_body_properties(env_handle, bag_handle),
            # #     recomputeInertia=False
            # # )
            #
            # self.gym.set_rigid_body_texture(
            #     env_handle,
            #     bag_handle,
            #     0,  # body index
            #     gymapi.MESH_VISUAL,
            #     bag_texture
            # )
            #
            # # color
            # # self.gym.set_rigid_body_color(
            # #     env_handle,
            # #     bag_handle,
            # #     0,
            # #     gymapi.MESH_VISUAL,
            # #     gymapi.Vec3(0.8, 0.2, 0.2)
            # # )

            # =========================================================
            # WALL-ALIGNED MARKERS
            # =========================================================
            # num_markers_per_wall = 5
            #
            # for _ in range(num_markers_per_wall):
            #
            #     # pick a random wall side
            #     side = np.random.choice(["left", "right", "top", "bottom"])
            #
            #     z = np.random.uniform(0.4, 1.2)
            #
            #     if side == "left":
            #         x = 0.2
            #         y = np.random.uniform(0.5, 4.5)
            #         normal = "x+"
            #
            #     elif side == "right":
            #         x = 4.8
            #         y = np.random.uniform(0.5, 4.5)
            #         normal = "x-"
            #
            #     elif side == "bottom":
            #         x = np.random.uniform(0.5, 4.5)
            #         y = 0.2
            #         normal = "y+"
            #
            #     else:  # top
            #         x = np.random.uniform(0.5, 4.5)
            #         y = 4.8
            #         normal = "y-"
            #
            #     marker_handle = self._spawn_wall_marker(env_handle, i, x, y, z, normal)
            #
            #     # # random color (for visibility)
            #     # color = gymapi.Vec3(np.random.rand(), np.random.rand(), np.random.rand())
            #     #
            #     # self.gym.set_rigid_body_color(
            #     #     env_handle,
            #     #     marker_handle,
            #     #     0,
            #     #     gymapi.MESH_VISUAL,
            #     #     color
            #     # )
            #     # =========================================================
            #     # APRILTAG-LIKE PATTERN
            #     # =========================================================
            #     pattern_size = 3  # 3x3 grid
            #
            #     base_x, base_y, base_z = x, y, z
            #
            #     cell = 0.12  # size of each square
            #
            #     for px in range(pattern_size):
            #         for py in range(pattern_size):
            #
            #             pose = gymapi.Transform()
            #
            #             offset_x = (px - 1) * cell
            #             offset_y = (py - 1) * cell
            #
            #             if normal in ["x+", "x-"]:
            #                 pose.p = gymapi.Vec3(base_x, base_y + offset_y, base_z + offset_x)
            #             else:
            #                 pose.p = gymapi.Vec3(base_x + offset_x, base_y, base_z + offset_y)
            #
            #             pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
            #
            #             small = self.gym.create_actor(
            #                 env_handle,
            #                 self.marker_asset,
            #                 pose,
            #                 "tag_cell",
            #                 i,
            #                 0,
            #                 0
            #             )
            #
            #             # checker pattern
            #             if (px + py) % 2 == 0:
            #                 color = gymapi.Vec3(0, 0, 0)
            #             else:
            #                 color = gymapi.Vec3(1, 1, 1)
            #
            #             self.gym.set_rigid_body_color(
            #                 env_handle,
            #                 small,
            #                 0,
            #                 gymapi.MESH_VISUAL,
            #                 color
            #             )
            ###############################

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.guitar_indices = torch.tensor(self.guitar_indices, dtype=torch.long, device=self.device)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         self.feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self.video_frames.append(self.video_frame)

        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            if self.eval_cfg is not None:
                bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                             self.root_states[self.num_train_envs, 2]
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                self.video_frames_eval.append(self.video_frame_eval)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self, env_ids, cfg):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.env.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.env.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.env.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.env.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale
