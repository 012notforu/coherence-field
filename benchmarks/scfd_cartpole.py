"""SCFD-driven cart-pole controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

from engine import accel_theta, load_config
from engine.integrators import leapfrog_step
from engine.energy import total_energy_density
from run.common import initialize_state

from observe.features import CartPoleFeatureExtractor, FeatureVector
from observe.policies import LinearPolicy, LinearPolicyConfig
from logic_ternary import CartPoleTernaryConfig, CartPoleTernaryController

def cartpole_step(E: np.ndarray, u: float, cfg: "CartPolePhysics") -> bool:
    g, mc, mp, l, dt, ns = cfg.gravity, cfg.masscart, cfg.masspole, cfg.length, cfg.dt, cfg.n_substeps
    x, x_dot, th, th_dot = E[0], E[1], E[2], E[3]
    for _ in range(ns):
        costh = np.cos(th)
        sinth = np.sin(th)
        temp = (u + mp * l * th_dot ** 2 * sinth) / (mc + mp)
        thacc = (g * sinth - costh * temp) / (l * (4.0 / 3.0 - mp * costh ** 2 / (mc + mp)))
        xacc = temp - mp * l * thacc * costh / (mc + mp)
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        th = th + dt * th_dot
        th_dot = th_dot + dt * thacc
    E[0], E[1], E[2], E[3] = x, x_dot, th, th_dot
    return (abs(x) > cfg.x_threshold) or (abs(th) > cfg.theta_threshold_rad)


@dataclass
class CartPolePhysics:
    dt: float = 0.02
    n_substeps: int = 1
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    x_threshold: float = 2.4
    theta_threshold_rad: float = 12 * np.pi / 180.0


@dataclass
class SCFDControllerConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    micro_steps: int = 40
    micro_steps_calm: int = 16
    encode_gain: float = 0.05
    encode_width: int = 3
    decay: float = 0.98
    smooth_lambda: float = 0.25
    deadzone_angle: float = np.deg2rad(1.0)
    deadzone_ang_vel: float = 0.1
    action_clip: float = 10.0
    action_delta_clip: float = 2.0
    reset_noise: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.02, 0.05], dtype=np.float32)
    )
    feature_momentum: float = 0.05
    deadzone_feature_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    )
    policy_weights: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 1.5, 0.8, 0.0, 6.0, 2.0, 0.5, 0.2, -0.1, 0.0], dtype=np.float32)
    )
    policy_bias: float = 0.0
    blend_linear_weight: float = 1.0
    blend_ternary_weight: float = 0.5
    ternary_force_scale: float = 7.5
    ternary_smooth_lambda: float = 1.0





class SCFDCartPoleController:
    def __init__(self, cfg: SCFDControllerConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.sim_cfg = load_config(cfg.scfd_cfg_path)
        self.dx = self.sim_cfg.grid.spacing
        self.physics = CartPolePhysics()
        self.grid_shape = self.sim_cfg.grid.shape
        self.mid_col = self.grid_shape[1] // 2
        self.feature_extractor = CartPoleFeatureExtractor(
            self.sim_cfg,
            momentum=cfg.feature_momentum,
            standardize=True,
            deadzone_scale=cfg.deadzone_feature_scale,
        )
        policy_cfg = LinearPolicyConfig(
            action_clip=cfg.action_clip,
            action_delta_clip=cfg.action_delta_clip,
            smooth_lambda=cfg.smooth_lambda,
        )
        self.policy = LinearPolicy(
            dim=self.feature_extractor.dimension,
            config=policy_cfg,
            weights=cfg.policy_weights.astype(np.float32),
            bias=float(cfg.policy_bias),
        )
        ternary_cfg = CartPoleTernaryConfig(
            force_scale=cfg.ternary_force_scale,
            smooth_lambda=cfg.ternary_smooth_lambda,
            action_clip=cfg.action_clip,
            deadzone_angle=cfg.deadzone_angle,
            deadzone_ang_vel=cfg.deadzone_ang_vel,
        )
        self.ternary_controller = CartPoleTernaryController(self.feature_extractor, ternary_cfg)
        self.last_feature_vector: FeatureVector | None = None
        self.last_policy_info: dict[str, float] | None = None
        self.last_ternary_info: dict[str, float] | None = None
        self.last_blend_components: Dict[str, float] = {'linear': 0.0, 'ternary': 0.0}
        self.last_action: float = 0.0
        self.reset()

    def reset(self) -> None:
        seed = int(self.rng.integers(0, 2**32 - 1))
        state = initialize_state(self.sim_cfg, seed)
        self.theta = state["theta"].astype(np.float32)
        self.theta_dot = state["theta_dot"].astype(np.float32)
        self.obs_filter = np.zeros(4, dtype=np.float32)
        self.feature_extractor.reset()
        self.policy.reset()
        self.ternary_controller.reset()
        self.last_ternary_info = None
        self.last_blend_components = {'linear': 0.0, 'ternary': 0.0}
        self.last_action = 0.0
        self.last_feature_vector = None
        self.last_policy_info = None

    def _in_deadzone(self, obs: np.ndarray) -> bool:
        return (abs(obs[2]) < self.cfg.deadzone_angle) and (abs(obs[3]) < self.cfg.deadzone_ang_vel)

    def _encode_observation(self, obs: np.ndarray, gain_scale: float) -> None:
        obs = obs.astype(np.float32)
        self.obs_filter = self.cfg.decay * self.obs_filter + (1.0 - self.cfg.decay) * obs
        h, w = self.grid_shape
        centers = np.linspace(0, h - 1, 4, dtype=np.int32)
        half = max(1, self.cfg.encode_width // 2)
        for i, center in enumerate(centers):
            row_start = max(center - 2, 0)
            row_end = min(center + 3, h)
            value = gain_scale * self.cfg.encode_gain * self.obs_filter[i]
            # inject antisymmetrically around center column
            left_start = max(self.mid_col - self.cfg.encode_width, 0)
            left_end = self.mid_col
            right_start = self.mid_col
            right_end = min(self.mid_col + self.cfg.encode_width, w)
            self.theta[row_start:row_end, left_start:left_end] -= value
            self.theta[row_start:row_end, right_start:right_end] += value

    def _evolve_field(self, deadzone: bool) -> Dict[str, float]:
        steps = self.cfg.micro_steps_calm if deadzone else self.cfg.micro_steps
        steps = max(1, steps)
        metrics = {"rms_step": 0.0}
        for _ in range(steps):
            self.theta, self.theta_dot, _, info = leapfrog_step(
                self.theta,
                self.theta_dot,
                lambda f: accel_theta(f, self.sim_cfg.physics, dx=self.dx),
                self.sim_cfg.integration.dt,
                max_step=None,
            )
            metrics = info
        return metrics

    def _blend_actions(self, linear_action: float, ternary_action: float) -> float:
        blend = (
            self.cfg.blend_linear_weight * float(linear_action)
            + self.cfg.blend_ternary_weight * float(ternary_action)
        )
        blend = float(np.clip(blend, -self.cfg.action_clip, self.cfg.action_clip))
        self.last_blend_components = {'linear': float(linear_action), 'ternary': float(ternary_action)}
        self.last_action = blend
        return blend

    def _compute_action(self, obs: np.ndarray, deadzone: bool) -> float:
        feature_vector = self.feature_extractor.extract(
            self.theta,
            self.theta_dot,
            obs,
            prev_action=self.policy.prev_action,
        )
        features = feature_vector.normalized
        self.last_feature_vector = feature_vector
        linear_action, linear_info = self.policy.act(
            features,
            deadzone=deadzone,
            deadzone_scale=self.feature_extractor.deadzone_scale,
        )
        self.last_policy_info = linear_info
        ternary_action = 0.0
        ternary_info: dict[str, float] | None = None
        if self.cfg.blend_ternary_weight != 0.0:
            ternary_action, ternary_info = self.ternary_controller.compute_action(
                self.theta,
                self.theta_dot,
                obs,
                feature_vector=feature_vector,
            )
        self.last_ternary_info = ternary_info
        action = self._blend_actions(linear_action, ternary_action)
        return action




    def run_episode(self, steps: int = 1000) -> Dict[str, float]:
        self.reset()
        E = self.rng.normal(0.0, self.cfg.reset_noise, size=4).astype(np.float32)
        alive = True
        max_steps = 0
        last_info: Dict[str, float] = {"rms_step": 0.0}
        while alive and max_steps < steps:
            deadzone = self._in_deadzone(E)
            gain_scale = 0.2 if deadzone else 1.0
            self._encode_observation(E, gain_scale)
            last_info = self._evolve_field(deadzone)
            action = self._compute_action(E, deadzone)
            alive = not cartpole_step(E, action, self.physics)
            max_steps += 1
        return {
            "steps": max_steps,
            "rms_step": last_info.get("rms_step", 0.0),
            "action_last": self.last_action,
        }

    def generate_visualization(
        self,
        steps: int = 800,
        out_dir: str | Path = "scfd_viz",
        save_video: bool = True,
        video_format: str = "auto",
    ) -> Dict[str, object]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.reset()
        fields = np.zeros((steps, *self.grid_shape), dtype=np.float32)
        env_hist = np.zeros((steps, 4), dtype=np.float32)
        action_hist = np.zeros(steps, dtype=np.float32)
        energy_hist = np.zeros(steps, dtype=np.float32)
        E = self.rng.normal(0.0, self.cfg.reset_noise, size=4).astype(np.float32)
        alive = True
        frame_count = 0
        for t in range(steps):
            if not alive:
                break
            deadzone = self._in_deadzone(E)
            gain_scale = 0.2 if deadzone else 1.0
            self._encode_observation(E, gain_scale)
            self._evolve_field(deadzone)
            action = self._compute_action(E, deadzone)
            fields[t] = self.theta
            env_hist[t] = E
            action_hist[t] = action
            energy_hist[t] = float(np.mean(total_energy_density(self.theta, self.theta_dot, self.sim_cfg.physics)))
            alive = not cartpole_step(E, action, self.physics)
            frame_count = t + 1
        fields = fields[:frame_count]
        env_hist = env_hist[:frame_count]
        action_hist = action_hist[:frame_count]
        energy_hist = energy_hist[:frame_count]
        center_raster = fields[:, self.grid_shape[0] // 2, :]
        raster_path = out_path / "scfd_field_raster.png"
        if frame_count > 0:
            plt.imsave(raster_path, center_raster, cmap="coolwarm")
        else:
            plt.imsave(raster_path, np.zeros((1, self.grid_shape[1]), dtype=np.float32), cmap="coolwarm")
        np.savez_compressed(out_path / "scfd_rollout.npz", field=fields, env=env_hist, action=action_hist, energy=energy_hist)
        video_path: Path | None = None
        if save_video and frame_count > 1:
            format_key = video_format.lower()
            if format_key not in {"auto", "mp4", "gif"}:
                raise ValueError(f"Unsupported video_format '{video_format}' (expected auto/mp4/gif)")
            v = float(np.max(np.abs(fields))) or 1.0
            fig, (ax_field, ax_cart) = plt.subplots(1, 2, figsize=(10, 4))
            im = ax_field.imshow(fields[0], cmap="coolwarm", vmin=-v, vmax=v, animated=True)
            ax_field.set_title("SCFD field")
            ax_field.axis("off")
            ax_cart.set_xlim(-self.physics.x_threshold * 1.2, self.physics.x_threshold * 1.2)
            ax_cart.set_ylim(-0.5, 1.2)
            ax_cart.set_xlabel("x")
            ax_cart.set_ylabel("height")
            cart_w, cart_h = 0.3, 0.2
            y_cart = 0.0
            cart = plt.Rectangle((0 - cart_w / 2, y_cart - cart_h / 2), cart_w, cart_h, fill=False)
            ax_cart.add_patch(cart)
            pole_line, = ax_cart.plot([], [], lw=2)
            force_txt = ax_cart.text(0.02, 0.92, "", transform=ax_cart.transAxes)

            def init():
                cart.set_xy((0 - cart_w / 2, y_cart - cart_h / 2))
                pole_line.set_data([], [])
                force_txt.set_text("")
                return im, cart, pole_line, force_txt

            def animate(i):
                im.set_array(fields[i])
                x, x_dot, th, th_dot = env_hist[i]
                cart.set_x(x - cart_w / 2)
                pivot = np.array([x, y_cart + cart_h / 2])
                tip = pivot + np.array([np.sin(th) * self.physics.length * 2.0, np.cos(th) * self.physics.length * 2.0])
                pole_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
                force_txt.set_text(f"u={action_hist[i]: .2f}")
                return im, cart, pole_line, force_txt

            anim = animation.FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=frame_count,
                interval=self.physics.dt * 1000.0,
                blit=True,
            )
            format_order: list[tuple[str, str]]
            if format_key == "mp4":
                format_order = [("mp4", "ffmpeg")]
            elif format_key == "gif":
                format_order = [("gif", "pillow")]
            else:
                format_order = [("mp4", "ffmpeg"), ("gif", "pillow")]
            for suffix, writer_name in format_order:
                candidate = out_path / f"scfd_cartpole.{suffix}"
                try:
                    anim.save(candidate, writer=writer_name, fps=max(int(1 / self.physics.dt), 10))
                    video_path = candidate
                    break
                except Exception:
                    video_path = None
            if video_path is None:
                fallback = out_path / "scfd_cartpole.gif"
                try:
                    anim.save(fallback, writer="pillow", fps=max(int(1 / self.physics.dt), 10))
                    video_path = fallback
                except Exception:
                    video_path = None
            plt.close(fig)
        return {
            "frames": frame_count,
            "raster": str(raster_path),
            "rollout": str(out_path / "scfd_rollout.npz"),
            "video": str(video_path) if video_path else None,
        }
