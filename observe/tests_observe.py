import importlib.util

import numpy as np
import pytest

from engine.params import load_config

from observe.adapter import FieldAdapter
from observe.trackers import SpectrumTracker, SymbolTracker
from observe.prototypes import PrototypeBank
from observe.sym_lm import NGramLanguageModel
from observe.controller import GentleController
from observe.features import CartPoleFeatureExtractor
from observe.policies import LinearPolicy, LinearPolicyConfig


cfg = load_config("cfg/defaults.yaml")
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_field_adapter_normalizes_mean_zero():
    adapter = FieldAdapter(cfg)
    theta = np.ones(cfg.grid.shape)
    theta_dot = np.zeros_like(theta)
    obs = adapter.prepare_observation(theta, theta_dot)
    mean = obs["normalized"].mean()
    assert abs(mean) < 1e-6


def test_spectrum_tracker_width_positive():
    tracker = SpectrumTracker(window=8)
    for _ in range(8):
        field = np.random.default_rng().normal(size=cfg.grid.shape)
        tracker.update(field)
    assert tracker.width >= 0.0


def test_symbol_tracker_entropy_single_symbol():
    tracker = SymbolTracker(history_window=32)
    metrics = tracker.update([1] * 10)
    assert metrics["perplexity"] == pytest.approx(1.0, rel=1e-6)


def test_ngram_model_perplexity_lower_for_seen_sequence():
    lm = NGramLanguageModel(order=2)
    sequence = [1, 2, 1, 2, 1, 2]
    lm.update(sequence)
    perplex_seen = lm.perplexity(sequence)
    perplex_unseen = lm.perplexity([3, 4, 5])
    assert perplex_seen <= perplex_unseen


def test_prototype_bank_updates_and_scores():
    bank = PrototypeBank(capacity=4)
    patch = np.ones((4, 4))
    bank.update(1, patch)
    score = bank.score(1, patch)
    assert score < 1e-6


def test_controller_clamps_adjustments():
    controller = GentleController(max_step=0.05)
    metrics = {"spectrum_width": 10.0, "perplexity": 10.0, "horizon": 0.1, "energy_drift": 0.0}
    decision = controller.step(metrics)
    assert all(abs(v) <= 0.05 for v in decision.nudges.values())
    assert not decision.safe_mode


def test_controller_enters_safe_mode():
    controller = GentleController(max_step=0.05)
    metrics = {"energy_drift": 0.02}
    decision = controller.step(metrics)
    assert decision.safe_mode
    assert all(v == 0.0 for v in decision.nudges.values())


def test_feature_extractor_standardizes_features():
    extractor = CartPoleFeatureExtractor(cfg, momentum=0.2)
    theta = np.zeros(cfg.grid.shape, dtype=np.float32)
    theta_dot = np.zeros_like(theta)
    env = np.array([0.0, 0.0, 0.01, 0.0], dtype=np.float32)
    result = extractor.extract(theta, theta_dot, env)
    assert result.raw.shape == (extractor.dimension,)
    assert result.normalized.shape == (extractor.dimension,)
    assert np.allclose(result.normalized, 0.0, atol=1e-6)
    theta[:, extractor.mid_col:] += 0.2
    env_next = np.array([0.05, -0.1, -0.02, 0.15], dtype=np.float32)
    result_next = extractor.extract(theta, theta_dot, env_next, prev_action=0.3)
    assert np.isfinite(result_next.normalized).all()
    assert not np.allclose(result_next.normalized, 0.0)
    scaled = extractor.deadzone_scaled(result_next.normalized)
    assert scaled.shape == result_next.normalized.shape


def test_linear_policy_respects_clamp_and_smoothing():
    linear = LinearPolicy(dim=2, config=LinearPolicyConfig(action_clip=1.0, action_delta_clip=5.0, smooth_lambda=1.0), weights=[1.0, 0.0])
    action, info = linear.act([2.0, 0.0])
    assert action == pytest.approx(1.0)
    assert info['raw'] == pytest.approx(2.0)
    smooth_policy = LinearPolicy(dim=1, config=LinearPolicyConfig(action_clip=2.0, action_delta_clip=0.5, smooth_lambda=0.5), weights=[2.0])
    act1, info1 = smooth_policy.act([1.0])
    assert act1 == pytest.approx(0.5)
    act2, info2 = smooth_policy.act([1.0])
    assert act2 == pytest.approx(1.0)
    smooth_policy.reset()
    act3, info3 = smooth_policy.act([1.0], deadzone=True, deadzone_scale=[0.1])
    assert info3['effective_features'][0] == pytest.approx(0.1)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_autoencoder_predictor_roundtrip():
    torch = pytest.importorskip("torch")
    from observe.ae_predictor import AutoEncoderPredictor

    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 16)).astype(np.float32)
    predictor = AutoEncoderPredictor(input_dim=16, latent_dim=4, hidden_dim=8, lr=1e-2)
    loss = predictor.step(batch)
    assert loss >= 0.0
    recon, recon_loss = predictor.reconstruct(batch)
    assert recon.shape == batch.shape
    assert recon_loss >= 0.0
