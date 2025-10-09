"""Emergent Model cart-pole benchmark (ported from EM33cart notebook)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap


@dataclass
class EMCartConfig:
    B: int = 512
    K: int = 8
    Np: int = 6
    Vars: int = 4
    T: int = 24
    force_scale: float = 9.0
    force_bias_b: float = 3.0
    force_noise_std: float = 0.15
    u_max: float = 10.0
    dt: float = 0.02
    n_substeps: int = 1
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    x_threshold: float = 2.4
    theta_threshold_rad: float = 12 * np.pi / 180.0
    GA_interval: int = 10
    p_rule_mut: float = 0.02
    p_state_mut: float = 0.005
    crossover_rate: float = 0.01
    obs_min: np.ndarray = field(default_factory=lambda: np.array([-2.4, -3.0, -0.2095, -3.5], dtype=np.float32))
    obs_max: np.ndarray = field(default_factory=lambda: np.array([ 2.4,  3.0,  0.2095,  3.5], dtype=np.float32))
    init_state_std: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.02, 0.05], dtype=np.float32))
    n_control_steps: int = 10_000

    def __post_init__(self) -> None:
        self.N = self.Vars * self.K * (1 + self.Np)


class EMCartPoleController:
    def __init__(self, cfg: EMCartConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.best_agent: tuple[np.ndarray, np.ndarray] | None = None
        self.best_steps: int = 0
        self._init_layout()
        self.reset()

    def _init_layout(self) -> None:
        cfg = self.cfg
        K, Np, Vars = cfg.K, cfg.Np, cfg.Vars
        Lvar = K * (1 + Np)
        N = Vars * Lvar
        region_starts = np.array([v * Lvar for v in range(Vars)], dtype=np.int64)
        self.I_idx = region_starts[:, None] + np.arange(K)[None, :] * (1 + Np)
        P_offsets = 1 + np.arange(Np, dtype=np.int64)
        self.P_idx = region_starts[:, None, None] + np.arange(K)[None, :, None] * (1 + Np) + P_offsets[None, None, :]
        self.L_idx = (self.I_idx - 1) % N
        self.R_idx = (self.I_idx + 1) % N
        self.I_flat = self.I_idx.reshape(-1)
        self.P_mask = np.zeros(N, dtype=bool)
        self.P_mask[self.P_idx.reshape(-1)] = True

    def reset(self) -> None:
        cfg = self.cfg
        B, N = cfg.B, cfg.N
        rng = self.rng
        self.R = rng.integers(0, 3, size=(B, 27), dtype=np.int8)
        self.R[:, 0] = 0
        self.S = np.zeros((B, N), dtype=np.int8)
        mut_mask = (rng.random((B, N)) < 0.02) & self.P_mask[None, :]
        self.S[mut_mask] = rng.integers(1, 3, size=mut_mask.sum(), dtype=np.int8)
        self.E = rng.normal(0.0, cfg.init_state_std, size=(B, 4)).astype(np.float32)
        self.alive = np.ones(B, dtype=bool)
        self.steps_alive = np.zeros(B, dtype=np.int32)
        self.stats = {"sum_L": 0.0, "sum_R": 0.0, "sum_u": 0.0, "n": 0}
        self.best_agent = None
        self.best_steps = 0

    def _discretize(self, E: np.ndarray) -> np.ndarray:
        z = (E - self.cfg.obs_min[None, :]) / (self.cfg.obs_max - self.cfg.obs_min)[None, :]
        z = np.clip(z, 0.0, 0.999999)
        return (z * self.cfg.K).astype(np.int32)

    def _encode(self, S: np.ndarray, bin_idx: np.ndarray, alive: np.ndarray) -> None:
        ai = np.where(alive)[0]
        S[np.ix_(ai, self.I_flat)] = 0
        for vi in range(self.cfg.Vars):
            pos = self.I_idx[vi, bin_idx[ai, vi]]
            S[ai, pos] = 2

    def _evolve(self, S: np.ndarray, R: np.ndarray) -> np.ndarray:
        for _ in range(self.cfg.T):
            left = np.roll(S, 1, axis=1)
            center = S
            right = np.roll(S, -1, axis=1)
            idx = (left * 9 + center * 3 + right).astype(np.int16)
            S = R[np.arange(S.shape[0])[:, None], idx]
        return S

    def _decode(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = (S == 2)
        Lsum = np.zeros(S.shape[0], dtype=np.float32)
        Rsum = np.zeros(S.shape[0], dtype=np.float32)
        for vi in range(self.cfg.Vars):
            Lsum += A[:, self.L_idx[vi]].sum(axis=1)
            Rsum += A[:, self.R_idx[vi]].sum(axis=1)
        u = self.cfg.force_scale * ((Rsum - Lsum) / (Rsum + Lsum + self.cfg.force_bias_b))
        u += self.rng.normal(0.0, self.cfg.force_noise_std, size=u.shape)
        u = np.clip(u, -self.cfg.u_max, self.cfg.u_max)
        return u.astype(np.float32), Lsum, Rsum

    def _cartpole_step(self, E: np.ndarray, u: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        g, mc, mp, l, dt, ns = cfg.gravity, cfg.masscart, cfg.masspole, cfg.length, cfg.dt, cfg.n_substeps
        x, x_dot, th, th_dot = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
        for _ in range(ns):
            force = u
            costh = np.cos(th)
            sinth = np.sin(th)
            temp = (force + mp * l * th_dot ** 2 * sinth) / (mc + mp)
            thacc = (g * sinth - costh * temp) / (l * (4.0 / 3.0 - mp * costh ** 2 / (mc + mp)))
            xacc = temp - mp * l * thacc * costh / (mc + mp)
            x = x + dt * x_dot
            x_dot = x_dot + dt * xacc
            th = th + dt * th_dot
            th_dot = th_dot + dt * thacc
        E[:, 0], E[:, 1], E[:, 2], E[:, 3] = x, x_dot, th, th_dot
        dead = (np.abs(E[:, 0]) > cfg.x_threshold) | (np.abs(E[:, 2]) > cfg.theta_threshold_rad)
        dead = dead & self.alive
        return dead

    def _ga_refill(self, dead_idx: np.ndarray) -> None:
        cfg = self.cfg
        if dead_idx.size == 0:
            return
        B, N = self.S.shape
        rng = self.rng
        parents = np.where(self.alive)[0]
        if parents.size == 0:
            parents = np.arange(B)
        pA = rng.choice(parents, size=dead_idx.size, replace=True)
        pB = rng.choice(parents, size=dead_idx.size, replace=True)
        R_new = np.where(rng.random((dead_idx.size, 27)) < cfg.crossover_rate, self.R[pA], self.R[pB]).copy()
        R_new[:, 0] = 0
        if cfg.p_rule_mut > 0.0:
            mut_r = rng.random((dead_idx.size, 27)) < cfg.p_rule_mut
            mut_r[:, 0] = False
            cand = rng.integers(0, 3, size=(dead_idx.size, 27), dtype=np.int8)
            R_new[mut_r] = cand[mut_r]
        S_new = np.where(rng.random((dead_idx.size, N)) < cfg.crossover_rate, self.S[pA], self.S[pB]).copy()
        if cfg.p_state_mut > 0.0:
            mut_s = (rng.random((dead_idx.size, N)) < cfg.p_state_mut) & self.P_mask[None, :]
            candS = rng.integers(0, 3, size=(dead_idx.size, N), dtype=np.int8)
            S_new[mut_s] = candS[mut_s]
        self.S[dead_idx] = S_new
        self.R[dead_idx] = R_new
        self.E[dead_idx] = rng.normal(0.0, cfg.init_state_std, size=(dead_idx.size, 4)).astype(np.float32)
        self.alive[dead_idx] = True
        self.steps_alive[dead_idx] = 0

    def control_step(self) -> None:
        alive = self.alive
        bin_idx = self._discretize(self.E)
        self._encode(self.S, bin_idx, alive)
        self.S[:] = self._evolve(self.S, self.R)
        u, Lsum, Rsum = self._decode(self.S)
        a = alive.astype(np.float32)
        self.stats["sum_L"] += float((Lsum * a).sum())
        self.stats["sum_R"] += float((Rsum * a).sum())
        self.stats["sum_u"] += float((np.abs(u) * a).sum())
        self.stats["n"] += int(a.sum())
        dead_now = self._cartpole_step(self.E, u)
        alive[dead_now] = False
        self.steps_alive[alive] += 1
        if dead_now.any():
            self._ga_refill(np.where(dead_now)[0])

    def run(self, steps: int | None = None) -> Dict[str, float]:
        steps = steps or self.cfg.n_control_steps
        best_steps = 0
        best_pair: tuple[np.ndarray, np.ndarray] | None = None
        for t in range(steps):
            self.control_step()
            idx = int(np.argmax(self.steps_alive))
            current = int(self.steps_alive[idx])
            if current > best_steps:
                best_steps = current
                best_pair = (self.R[idx].copy(), self.S[idx].copy())
        if best_pair is not None:
            self.best_steps = best_steps
            self.best_agent = best_pair
        alive_count = int(self.alive.sum())
        mean_alive = float(self.steps_alive[self.alive].mean()) if alive_count else 0.0
        avg_L = self.stats["sum_L"] / self.stats["n"] if self.stats["n"] else 0.0
        avg_R = self.stats["sum_R"] / self.stats["n"] if self.stats["n"] else 0.0
        avg_abs_u = self.stats["sum_u"] / self.stats["n"] if self.stats["n"] else 0.0
        return {
            "best_steps": best_steps,
            "alive": alive_count,
            "mean_steps_alive": mean_alive,
            "avg_L": avg_L,
            "avg_R": avg_R,
            "avg_abs_u": avg_abs_u,
        }

    def generate_visualization(self, horizon: int = 800, out_dir: str | Path = "em_viz", save_video: bool = True) -> Dict[str, object]:
        if self.best_agent is None:
            raise RuntimeError("Call run() before generating visualization.")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        R_vec, S_vec = self.best_agent
        R1 = R_vec[None, :].copy()
        S1 = S_vec[None, :].copy()
        E1 = self.rng.normal(0.0, self.cfg.init_state_std, size=(1, 4)).astype(np.float32)
        alive = np.array([True], dtype=bool)

        steps = min(horizon, self.cfg.n_control_steps)
        M = np.zeros((steps, self.cfg.N), dtype=np.uint8)
        E_hist = np.zeros((steps, 4), dtype=np.float32)
        u_hist = np.zeros(steps, dtype=np.float32)

        frame_count = 0
        for t in range(steps):
            if not alive[0]:
                break
            bins = self._discretize(E1)
            self._encode(S1, bins, alive)
            S1[:] = self._evolve(S1, R1)
            u, _, _ = self._decode(S1)

            x, x_dot, th, th_dot = E1[0]
            for _ in range(self.cfg.n_substeps):
                force = float(u[0])
                costh = np.cos(th)
                sinth = np.sin(th)
                temp = (force + self.cfg.masspole * self.cfg.length * th_dot ** 2 * sinth) / (self.cfg.masscart + self.cfg.masspole)
                thacc = (self.cfg.gravity * sinth - costh * temp) / (self.cfg.length * (4.0 / 3.0 - self.cfg.masspole * costh ** 2 / (self.cfg.masscart + self.cfg.masspole)))
                xacc = temp - self.cfg.masspole * self.cfg.length * thacc * costh / (self.cfg.masscart + self.cfg.masspole)
                x = x + self.cfg.dt * x_dot
                x_dot = x_dot + self.cfg.dt * xacc
                th = th + self.cfg.dt * th_dot
                th_dot = th_dot + self.cfg.dt * thacc
            E1[0] = np.array([x, x_dot, th, th_dot], dtype=np.float32)
            dead = (abs(x) > self.cfg.x_threshold) or (abs(th) > self.cfg.theta_threshold_rad)

            M[frame_count] = S1[0]
            E_hist[frame_count] = E1[0]
            u_hist[frame_count] = float(u[0])
            frame_count += 1
            if dead:
                alive[0] = False
                break

        M = M[:frame_count]
        E_hist = E_hist[:frame_count]
        u_hist = u_hist[:frame_count]

        cmap = ListedColormap([[1, 1, 1], [1, 0, 0], [0, 0, 1]])
        raster_path = out_path / "em_ca_raster.png"
        if frame_count > 0:
            plt.imsave(raster_path, M, cmap=cmap, vmin=0, vmax=2)
        else:
            plt.imsave(raster_path, np.zeros((1, self.cfg.N), dtype=np.uint8), cmap=cmap, vmin=0, vmax=2)

        np.savez_compressed(out_path / "em_rollout.npz", M=M, E=E_hist, u=u_hist)

        video_path: Path | None = None
        if save_video and frame_count > 1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlim(-self.cfg.x_threshold * 1.2, self.cfg.x_threshold * 1.2)
            ax.set_ylim(-0.5, 1.2)
            ax.set_xlabel("x")
            ax.set_ylabel("height")
            cart_w, cart_h = 0.3, 0.2
            y_cart = 0.0
            cart = plt.Rectangle((0 - cart_w / 2, y_cart - cart_h / 2), cart_w, cart_h, fill=False)
            ax.add_patch(cart)
            pole_line, = ax.plot([], [], lw=2)
            force_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes)

            def init():
                cart.set_xy((0 - cart_w / 2, y_cart - cart_h / 2))
                pole_line.set_data([], [])
                force_txt.set_text("")
                return cart, pole_line, force_txt

            def animate(i):
                x, x_dot, th, th_dot = E_hist[i]
                cart.set_x(x - cart_w / 2)
                pivot = np.array([x, y_cart + cart_h / 2])
                tip = pivot + np.array([np.sin(th) * self.cfg.length * 2.0, np.cos(th) * self.cfg.length * 2.0])
                pole_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
                force_txt.set_text(f"u={u_hist[i]: .2f}")
                return cart, pole_line, force_txt

            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_count, interval=self.cfg.dt * 1000.0, blit=True)
            video_path = out_path / "em_cartpole.mp4"
            try:
                anim.save(video_path, writer="ffmpeg", fps=max(int(1 / self.cfg.dt), 10))
            except Exception:
                video_path = video_path.with_suffix(".gif")
                anim.save(video_path, writer="pillow", fps=max(int(1 / self.cfg.dt), 10))
            plt.close(fig)

        return {
            "frames": frame_count,
            "raster": str(raster_path),
            "rollout": str(out_path / "em_rollout.npz"),
            "video": str(video_path) if video_path else None,
        }
