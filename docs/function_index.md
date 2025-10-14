# Function Index

This document lists top-level functions and class methods across the primary SCFD modules. Tests and generated assets are excluded.

## benchmarks/cartpole_stress_test.py
### Module Functions
- `main()`
### Class Methods
- `CartPoleState`: add_noise(), get_noisy_observation(), to_array()
- `StressTestCartPole`: __init__(), _save_stress_test_results(), compute_control_force(), compute_performance_metrics(), generate_position_setpoints(), initialize_scfd_field(), reset_metrics(), reset_state(), run_stress_test(), setup_efe_tuner(), step_physics(), update_efe_parameters(), update_physics_drift(), update_scfd_field()

## benchmarks/em_cartpole.py
### Class Methods
- `EMCartConfig`: __post_init__()
- `EMCartPoleController`: __init__(), _cartpole_step(), _decode(), _discretize(), _encode(), _evolve(), _ga_refill(), _init_layout(), control_step(), generate_visualization(), reset(), run()

## benchmarks/flow_constriction.py
### Class Methods
- `FlowConstrictionSimulator`: __init__(), _apply_boundaries(), _build_geometry(), _derivatives(), generate_visualization(), reset(), run(), step()

## benchmarks/flow_redundant.py
### Class Methods
- `FlowRedundantSimulator`: __init__(), _build_actuators(), _build_monitor(), _derivatives(), generate_visualization(), reset(), run(), step()

## benchmarks/flow_regime_sweep.py
### Class Methods
- `FlowRegimeSweep`: __init__(), _run_single(), aggregate_metrics(), run(), save_visualizations()

## benchmarks/heat_diffusion_anisotropic.py
### Module Functions
- `synthetic_anisotropic_temperature()`
### Class Methods
- `HeatAnisotropicSimulator`: __init__(), _anisotropic_diffusion(), _build_tensor(), generate_visualization(), reset(), step()

## benchmarks/heat_diffusion_arc.py
### Module Functions
- `apply_arc_transform()`
- `build_transform_cycle()`
### Class Methods
- `HeatDiffusionArcSimulator`: __init__(), _advance_transform(), generate_visualization(), reset(), run(), step()

## benchmarks/heat_diffusion_inverse.py
### Module Functions
- `synthetic_source_map()`
### Class Methods
- `HeatInverseSimulator`: __init__(), _forward_diffuse(), generate_visualization(), reset(), run(), step()

## benchmarks/heat_diffusion_mobile.py
### Module Functions
- `synthetic_mobile_target()`
### Class Methods
- `HeatMobileSimulator`: __init__(), _current_waypoint(), _heater_mask(), _target_field(), generate_visualization(), reset(), run(), step()

## benchmarks/heat_diffusion_obstacle.py
### Module Functions
- `synthetic_obstacle_target()`
### Class Methods
- `HeatObstacleSimulator`: __init__(), _apply_budget(), _build_obstacle_mask(), _corner_mask(), generate_visualization(), reset(), step()

## benchmarks/heat_diffusion_periodic.py
### Module Functions
- `synthetic_periodic_temperature()`
### Class Methods
- `HeatPeriodicSimulator`: __init__(), _apply_budget(), _wrap_error(), generate_visualization(), reset(), step()

## benchmarks/heat_diffusion_routing.py
### Module Functions
- `generate_blob_pattern()`
### Class Methods
- `HeatDiffusionRoutingSimulator`: __init__(), _build_collision_mask(), generate_visualization(), reset(), run(), step()

## benchmarks/heat_front_tracking.py
### Module Functions
- `generate_ring()`
### Class Methods
- `HeatFrontTrackingSimulator`: __init__(), generate_visualization(), reset(), run(), step()

## benchmarks/heat_parameter_id.py
### Class Methods
- `HeatParameterIDSimulator`: __init__(), _build_alpha_map(), generate_visualization(), reset(), run(), step()

## benchmarks/image_completion.py
### Module Functions
- `create_test_images()`
### Class Methods
- `ImageCompletionAgent`: __init__(), detect_image_failure_pattern(), sense_image_environment()
- `ImageCompletionSystem`: __init__(), _apply_damage(), _create_damage_mask(), _create_image_completion_agents(), _save_reconstruction_results(), _setup_image_completion_compositions(), reconstruct_image()

## benchmarks/maze_solving.py
### Module Functions
- `generate_maze()`
### Class Methods
- `AdaptiveMazeSolver`: __init__(), _adjacent_wall_count(), _apply_vector_to_config(), _blend_configs(), _collect_coverage_vectors(), _collect_maze_vectors(), _collect_precision_vectors(), _create_environment_observation(), _create_scfd_controller(), _detect_failure_pattern(), _distance_to_goal(), _load_vector_payload(), _make_move(), _manhattan_distance(), _select_next_vector(), _select_support_vectors(), _should_switch_vector(), _vector_to_guidance_field(), generate_visualization(), reset(), run_generation(), solve_adaptively()
- `MazeSCFDController`: __init__(), step()

## benchmarks/multi_agent_grid.py
### Class Methods
- `VectorTag`: __post_init__()
- `VectorRegistry`: __init__(), _initialize_tags(), get_vectors_by_tag(), get_vectors_for_pattern(), update_performance()
- `GridCellAgent`: __init__(), _log_field_interactions(), apply_vector_with_physics(), detect_failure_pattern(), select_vector_by_pattern(), sense_local_environment(), should_activate(), step(), update_exploration_mode()
- `MultiAgentGridSystem`: __init__(), _initialize_agents(), run_simulation(), step()

## benchmarks/multi_agent_maze.py
### Class Methods
- `MazeGridCellAgent`: __init__(), _apply_meta_vector(), _is_valid_position(), _manhattan_distance(), _softmax(), apply_maze_vector(), compute_movement_direction(), compute_movement_utility(), detect_maze_failure_pattern(), select_maze_vector(), sense_maze_environment(), step_maze_solving()
- `MultiAgentMazeSolver`: __init__(), _create_goal_gradient_field(), _initialize_maze_agents(), execute_agent_movement(), solve_maze(), visualize_solution()

## benchmarks/run_cartpole.py
### Module Functions
- `_load_scfd_config()`
- `_parse_weights()`
- `main()`
- `parse_args()`
- `run_em()`
- `run_scfd()`

## benchmarks/run_flow_constriction.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_flow_regime.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_heat_anisotropic.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_heat_obstacle.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_heat_periodic.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_wave_cavity.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/run_wave_partial.py
### Module Functions
- `_parse_args()`
- `main()`

## benchmarks/vector_composition.py
### Class Methods
- `CompositionRule`: __post_init__()
- `VectorComposer`: __init__(), _adapt_weights(), _adaptive_composition(), _apply_weight_adjustments(), _conditional_composition(), _enforce_cache_invariants(), _initialize_default_compositions(), _initialize_metropolis_cache(), _linear_composition(), _on_parameter_update(), _on_registry_callback(), _refresh_cache_if_stale(), _score_composition_rule(), _sequential_composition(), _update_composition_performance(), add_composition_rule(), compose_vectors(), extract_scfd_features_for_sensing(), get_acceptance_bounds(), get_composition_analytics(), get_metropolis_temperature(), get_scfd_parameters_snapshot(), get_scfd_telemetry_summary(), record_scfd_telemetry(), select_composition()
- `CompositeAgent`: __init__(), _compute_local_novelty(), _infer_domain_from_context(), _monitor_acceptance_patterns(), _reset_safety_alert(), _trigger_kill_switch(), _trigger_safety_alert(), get_acceptance_telemetry(), reset_kill_switch(), step_with_composition()

## benchmarks/wave_field_cavity.py
### Module Functions
- `standing_mode_target()`
### Class Methods
- `WaveCavityController`: __init__(), _smooth(), reset(), step()
- `WaveCavitySimulator`: __init__(), _apply_boundaries(), _build_boundary_mask(), generate_visualization(), reset(), run(), step()

## benchmarks/wave_field_mode_switch.py
### Class Methods
- `WaveModeSwitchSimulator`: __init__(), _build_masks(), _update_target(), generate_visualization(), reset(), run(), step()

## benchmarks/wave_field_partial.py
### Module Functions
- `random_sensor_mask()`
### Class Methods
- `WavePartialController`: __init__(), _smooth(), reset(), step()
- `WavePartialSimulator`: __init__(), _apply_boundaries(), _build_boundary_mask(), generate_visualization(), reset(), run(), step()

## em_baseline/diagnostics_em.py
### Module Functions
- `compute_symbol_metrics()`
- `spectrum_metrics()`

## em_baseline/encode_decode.py
### Module Functions
- `decode_output()`
- `encode_input()`

## em_baseline/halting.py
### Class Methods
- `HaltingController`: __post_init__(), should_halt()

## em_baseline/optimizer.py
### Class Methods
- `RandomStateOptimizer`: evolve()

## em_baseline/runner.py
### Module Functions
- `run_em_episode()`

## em_baseline/transition_f.py
### Class Methods
- `LeniaTransition`: __post_init__(), _activate(), _build_kernel(), step()

## engine/diagnostics.py
### Module Functions
- `coherence_metrics()`
- `compute_energy_report()`
- `edge_metrics()`
- `energy_drift()`
- `impulse_response()`
- `predictability_horizon()`
- `radial_spectrum()`

## engine/energy.py
### Module Functions
- `_compute_local_scfd_energy_change()`
- `_reflect()`
- `_sigmoid()`
- `coherence_energy_density()`
- `coherence_p_to_m()`
- `compute_local_entropy_change()`
- `compute_scfd_coherence()`
- `compute_scfd_curvature()`
- `compute_scfd_energy_from_grid()`
- `compute_scfd_gradient()`
- `compute_scfd_lagrangian()`
- `compute_total_energy()`
- `cross_gradient_energy_density()`
- `curvature_energy_density()`
- `extract_scfd_features()`
- `get_scfd_telemetry()`
- `kinetic_energy_density()`
- `metropolis_accept()`
- `metropolis_accept_prob()`
- `potential_energy_density()`
- `run_scfd_boundary_tests()`
- `run_scfd_validation_probes()`
- `run_scfd_validation_probes()`
- `scfd_accept_update()`
- `scfd_accept_update_domain_aware()`
- `scfd_accept_update_hard()`
- `scfd_accept_update_soft()`
- `scfd_accept_update_weighted()`
- `scfd_acceptance_weight()`
- `scfd_catastrophic_guard()`
- `scfd_preflight_checks()`
- `total_energy_density()`
- `validate_scfd_integration()`
### Class Methods
- `SCFDAcceptanceTracker`: __init__(), _check_kill_switch(), add_proposal(), get_acceptance_rate(), get_stats(), reset()

## engine/integrators.py
### Module Functions
- `leapfrog_step()`
- `suggest_cfl_dt()`

## engine/ops.py
### Module Functions
- `_as_step()`
- `bilaplacian()`
- `divergence()`
- `grad()`
- `hessian_action()`
- `laplacian()`
- `norm_sq_grad()`

## engine/params.py
### Module Functions
- `_read_yaml()`
- `load_config()`
- `validate_near_critical()`
### Class Methods
- `GridSpec`: size()
- `PotentialConfig`: derivative(), energy()
- `HeterogeneityConfig`: _gaussian_kernel(), generate_field()
- `PhysicsParams`: coherence_exponent(), coherence_linear_term(), coherence_margin(), wave_speed_sq()
- `SimulationConfig`: startup_summary()

## engine/pde_core.py
### Module Functions
- `accel_CK()`
- `accel_theta()`
- `coherence_force()`

## engine/scheduler.py
### Class Methods
- `AsyncScheduler`: __init__(), _bump(), _poisson_mask(), _shuffle_mask(), ks_statistic(), sample_mask()

## engine/symbolizer.py
### Module Functions
- `_bandpass()`
- `extract_symbols()`

## engine/tests_engine.py
### Module Functions
- `_physics()`
- `test_async_scheduler_mask_shape_and_rate()`
- `test_cross_variational_honesty()`
- `test_edge_metrics_flat_field()`
- `test_energy_drift_small_series()`
- `test_hessian_action_periodic()`
- `test_impulse_response_static_dynamics()`
- `test_laplacian_matches_divergence_of_gradient()`
- `test_leapfrog_roundtrip()`
- `test_longrun_energy_drift()`
- `test_metropolis_accept_monotonic()`
- `test_ops_green_identity()`
- `test_suggest_cfl_dt_bounds()`
- `test_symbolizer_read_only()`
- `test_validate_near_critical_margin_bounds()`
- `test_validate_near_critical_passes()`
- `test_validate_near_critical_wave_speed_failure()`
- `test_variational_honesty_single_field()`

## observe/adapter.py
### Class Methods
- `FieldAdapter`: denormalize(), flatten(), normalize(), prepare_observation(), stack(), update_stats()

## observe/ae_predictor.py
### Class Methods
- `AutoEncoderPredictor`: __post_init__(), _to_tensor(), reconstruct(), regime_change_score(), step(), update_predictor()

## observe/controller.py
### Module Functions
- `_soft_over()`
- `controller_score()`
### Class Methods
- `MetadataDrivenController`: __init__(), _check_constraints(), _compute_adjustment(), _default_parameter_registry(), _default_policy_table(), _default_signal_registry(), _evaluate_policy_rules(), _generate_parameter_adjustments(), _get_cooldown_weight(), _parameter_on_cooldown(), _record_nudge(), _record_nudge_weighted(), _rule_triggered(), _update_signal_state(), get_controller_weight(), get_status(), step()
- `GentleController`: __post_init__(), _clamp(), _update_ema(), _validate_nudges(), get_sensing_status(), step()

## observe/prototypes.py
### Class Methods
- `PrototypeBank`: ranked(), score(), update()

## observe/sym_lm.py
### Class Methods
- `NGramLanguageModel`: __post_init__(), distribution(), perplexity(), reset(), update()

## observe/tests_observe.py
### Module Functions
- `test_autoencoder_predictor_roundtrip()`
- `test_controller_clamps_adjustments()`
- `test_controller_enters_safe_mode()`
- `test_feature_extractor_standardizes_features()`
- `test_field_adapter_normalizes_mean_zero()`
- `test_linear_policy_respects_clamp_and_smoothing()`
- `test_ngram_model_perplexity_lower_for_seen_sequence()`
- `test_prototype_bank_updates_and_scores()`
- `test_spectrum_tracker_width_positive()`
- `test_symbol_tracker_entropy_single_symbol()`

## observe/trackers.py
### Class Methods
- `SpectrumTracker`: __post_init__(), update(), width()
- `SymbolTracker`: __post_init__(), most_common(), update()
- `HorizonTracker`: __post_init__(), update()

## orchestrator/example.py
### Module Functions
- `choose_engine()`
- `main()`
- `run_episode()`

## orchestrator/pipeline.py
### Module Functions
- `_classify_target_array()`
- `_default_tuple()`
- `_infer_grid_shape()`
- `_infer_physics()`
- `_infer_target_kind()`
- `_infer_transform_cycle()`
- `_infer_vector_metadata()`
- `build_plan()`
- `load_vector_registry()`
- `plan_for_environment()`
- `probe_environment()`
### Class Methods
- `ProbeReport`: transform_cycle_length()
- `Plan`: summary()

## run/common.py
### Module Functions
- `apply_gentle_gate()`
- `compute_edge_diagnostics()`
- `finalize_plots()`
- `high_frequency_fraction()`
- `initialize_state()`
- `load_simulation_config()`
- `setup_logger()`
- `store_spectrum()`
- `summarize_energy()`
### Class Methods
- `ImpulseProbe`: measure()

## run/compare_scfd_vs_em.py
### Module Functions
- `_compute_em_spectrum()`
- `_filter_scalars()`
- `_parse_args()`
- `_run_em()`
- `_run_scfd()`
- `_write_summary()`
- `main()`

## run/latency_profile.py
### Module Functions
- `_parse_args()`
- `_profile_steps()`
- `main()`
- `profile_flow_redundant()`
- `profile_heat_diffusion()`
- `profile_wave_mode_switch()`

## run/main_conservative.py
### Module Functions
- `_parse_args()`
- `_validate_cfl()`
- `main()`

## run/main_hybrid.py
### Module Functions
- `_parse_args()`
- `_validate_cfl()`
- `main()`

## run/main_ml_loop.py
### Module Functions
- `_maybe_autoencoder()`
- `_parse_args()`
- `main()`

## run/mnist_energy_alignment.py
### Module Functions
- `analyze_samples()`
- `main()`
- `parse_args()`
- `single_example()`

## run/mnist_energy_core.py
### Module Functions
- `_device()`
- `_physics_params()`
- `activation_map()`
- `compute_scfd_metrics()`
- `energy_reward()`
- `expected_energy()`
- `get_mnist_dataloaders()`
- `load_energy_params()`
- `load_or_train_cnn()`
- `sample_activation_fields()`
- `train_mnist_model()`
### Class Methods
- `SimpleMNISTCNN`: __init__(), forward()
- `EnergyParams`: from_vector()

## run/mnist_energy_monitor.py
### Module Functions
- `main()`
- `parse_args()`

## run/robustness_battery.py
### Module Functions
- `_parse_args()`
- `_run_scenario()`
- `_summarize_flow()`
- `_summarize_heat()`
- `_summarize_wave()`
- `evaluate_flow_redundant()`
- `evaluate_heat_diffusion()`
- `evaluate_wave_mode_switch()`
- `main()`

## run/run_heat_arc.py
### Module Functions
- `_parse_args()`
- `main()`

## run/run_heat_front.py
### Module Functions
- `_parse_args()`
- `main()`

## run/run_heat_param_id.py
### Module Functions
- `_parse_args()`
- `main()`

## run/run_heat_routing.py
### Module Functions
- `_coerce_centers()`
- `_parse_args()`
- `_parse_centers()`
- `main()`

## run/train_cma_energy_alignment.py
### Module Functions
- `evaluate_vector()`
- `main()`
- `parse_args()`
- `run_cma()`
- `save_vector()`
- `write_manifest()`

## run/train_cma_flow_constriction.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_flow_redundant.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_flow_regime.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_gray_scott.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_anisotropic.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_arc.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_metadata_from_params()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_front.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_metadata_from_params()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_inverse.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_mobile.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_parse_path()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_obstacle.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_param_id.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_metadata_from_params()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_periodic.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_heat_routing.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_metadata_from_params()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_maze_explore.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_evaluate_vector()`
- `_parse_args()`
- `_parse_densities()`
- `_parse_shape()`
- `_prepare_solver()`
- `_save_vector()`
- `_sigmoid()`
- `_simulate_episode()`
- `_vector_to_metadata()`
- `_write_history_header()`
- `main()`

## run/train_cma_scfd.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_metadata()`
- `_config_from_vector()`
- `_evaluate()`
- `_metadata_from_config()`
- `_parse_args()`
- `_save_vector()`
- `_training_metadata()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_wave_cavity.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_wave_mode_switch.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## run/train_cma_wave_partial.py
### Module Functions
- `_append_history()`
- `_clip()`
- `_config_from_vector()`
- `_evaluate()`
- `_parse_args()`
- `_save_vector()`
- `_vector_from_config()`
- `_write_history_header()`
- `main()`

## scripts/meta_smoke.py
### Module Functions
- `_filter_params()`
- `_latest_vector()`
- `_mean()`
- `_resolve_cfg_path()`
- `evaluate_cartpole()`
- `evaluate_flow()`
- `evaluate_gray()`
- `evaluate_heat()`
- `evaluate_wave()`
- `main()`

## utils/active_inference.py
### Module Functions
- `create_efe_movement_wrapper()`
### Class Methods
- `ActiveInferenceEvaluator`: __init__(), _compute_ambiguity(), _compute_gradient_variance(), _compute_local_entropy(), _compute_prediction_error(), _compute_risk(), _predict_composition_failure_risk(), _valid_position(), compute_movement_efe(), is_exploration_mode(), select_action()

## utils/efe_controller.py
### Module Functions
- `wrap_agent_with_efe()`
- `wrap_system_with_efe()`
### Class Methods
- `EFEController`: __init__(), _compute_local_ambiguity(), _efe_composition_step(), _efe_enhanced_step(), _efe_should_activate(), _efe_standard_agent_step(), _estimate_composition_ambiguity(), _estimate_composition_risk(), _evaluate_composition_options(), _should_attempt_composition(), _should_explore(), _should_use_efe(), _standard_step(), _update_local_context(), get_efe_stats(), step_with_efe()

## utils/efe_tuner.py
### Module Functions
- `create_image_completion_test_function()`
- `create_realtime_tuner_for_composition()`
### Class Methods
- `EFETunerConfig`: __post_init__()
- `EFEParameterTuner`: __init__(), _apply_lexicographic_constraints(), _apply_trust_region_with_floors(), _auto_revert_parameters(), _can_update_now(), _compute_metadata_driven_updates(), _compute_realtime_ambiguity(), _compute_realtime_risk(), _compute_robustness(), _estimate_parameter_sensitivity(), _tournament_select(), _update_metric_windows(), add_parameter_range(), crossover(), enable_realtime_tuning(), evaluate_fitness(), evolve_population(), generate_individual(), get_realtime_status(), mutate_individual(), optimize(), setup_image_completion_tuning(), update_realtime_parameters()

## utils/event_bus.py
### Module Functions
- `get_event_bus()`
- `publish_on_change()`
- `reset_event_bus()`
### Class Methods
- `Event`: __post_init__()
- `ParameterUpdateEvent`: __post_init__()
- `SignalThresholdEvent`: __post_init__()
- `EFEOptimizationEvent`: __post_init__()
- `EventSubscription`: __init__(), get_handler(), is_alive()
- `EventBus`: __init__(), _cleanup_dead_subscriptions(), get_stats(), publish(), publish_efe_optimization(), publish_parameter_update(), publish_signal_threshold(), subscribe(), unsubscribe(), unsubscribe_component()
- `EventBusComponent`: __del__(), __init__(), cleanup_subscriptions(), publish(), subscribe()

## utils/io.py
### Module Functions
- `backup_file()`

## utils/logging.py
### Module Functions
- `create_run_directory()`
### Class Methods
- `PhysicsContext`: to_dict()
- `VectorInvocation`: to_dict()
- `OutcomeMetrics`: to_dict()
- `CellLogEntry`: to_dict()
- `RunLogger`: __post_init__(), compute_neighbor_summary(), compute_physics_context(), dump_config(), log_cell_action(), log_csv(), log_interaction(), log_step(), log_vector_stats_summary()

## utils/maze_controller.py
### Module Functions
- `_bfs_path_exists()`
- `create_simple_maze_solvable()`
- `create_solvable_maze()`
- `test_maze_soft_acceptance()`
### Class Methods
- `SimpleMazeAgent`: __init__(), current_position(), get_path(), move_to()
- `SimpleMazeField`: __init__(), get_field_copy(), is_valid_position()
- `MazeController`: __init__(), _check_guardrails(), _compute_proposal(), _execute_maze_step(), _get_param_value(), _is_path_possible(), _log_progress(), _set_domain_mode(), _set_param_value(), get_acceptance_telemetry(), get_telemetry_summary(), reset_kill_switch(), solve_maze()
- `MazeExplorationStateFixed`: __init__(), get_low_visit_neighbor()

## utils/maze_exploration.py
### Module Functions
- `_coerce_pos()`
- `choose_vector_by_context()`
- `maze_step_with_soft_acceptance()`
- `scfd_accept_update_maze_aware()`
### Class Methods
- `MazeExplorationState`: __init__(), get_low_visit_neighbor(), get_telemetry(), is_stalled(), novelty(), record_accept_reason(), reset_stall_counter(), update_visit()

## utils/numerics.py
### Module Functions
- `ema_update()`
- `fft2d()`
- `gaussian_kernel()`

## utils/parameter_registry.py
### Module Functions
- `get_global_registry()`
- `get_parameter_value()`
- `get_role_values()`
- `register_parameter()`
- `set_snapshot_mode()`
- `update_parameter()`
### Class Methods
- `ParameterConstraint`: _check_trust_region_violation(), get_effective_trust_region(), validate()
- `ParameterEntry`: __init__(), _process_value_by_dtype(), update_value()
- `ParameterRegistry`: __init__(), _compute_role_delta(), _initialize_core_parameters(), _initialize_group_invariants(), _update_weights_role(), add_event_callback(), get_all_values(), get_coupling_group(), get_diagnostics(), get_parameter(), get_parameters_by_role(), get_values_by_role(), load_state(), register_parameter(), remove_event_callback(), save_state(), update_parameter(), update_role()

## utils/provenance.py
### Module Functions
- `_git()`
- `write_meta()`

## utils/registry_efe_tuner.py
### Module Functions
- `get_global_registry_efe_tuner()`
### Class Methods
- `RegistryEFETuner`: __init__(), _ab_test_parameters(), _apply_parameter_set(), _compute_current_efe(), _evaluate_parameter_set(), _generate_joint_candidates(), _generate_role_candidates(), _initialize_role_trust_regions(), _optimize_full_joint(), _optimize_hierarchical(), _optimize_role_block_coordinate(), _optimize_single_role(), _passes_early_fail_constraints(), _select_multiple_target_roles(), _select_target_role(), _update_trust_regions_failure(), _update_trust_regions_success(), get_diagnostics(), record_sensing_decision(), run_strategic_optimization(), should_optimize()

## utils/sensing_router.py
### Module Functions
- `get_global_sensing_router()`
### Class Methods
- `SensingRouter`: __init__(), _calculate_adjustment(), _check_constraint_violations(), _evaluate_condition(), _get_constraint_weight(), _initialize_policy_rules(), _initialize_signal_catalog(), _normalize_signal(), _update_constraint_margins(), apply_role_adjustments(), evaluate_policies(), get_diagnostics(), process_signals(), register_policy_rule(), register_signal()

## utils/trap_free_navigation.py
### Class Methods
- `TrapFreeNavigator`: __init__(), _compute_distance_field(), _get_distance(), _sigmoid(), combine_guidance(), compute_progress(), get_diagnostics(), get_goal_direction(), get_navigation_weights(), is_stuck(), set_maze_and_goal(), update_state()

## utils/unified_metadata.py
### Class Methods
- `VectorMetrics`: __post_init__()
- `CompositionMetrics`: __post_init__()
- `UnifiedMetadataManager`: __init__(), _create_empty_metadata(), _create_test_run_summary(), _import_pathway_logs(), _import_vector_stats(), _load_or_create_metadata(), _update_composition_metrics(), _update_vector_metrics(), export_vector_data(), get_composition_analytics(), get_system_summary(), get_vector_analytics(), import_run_data(), log_move(), register_composition(), register_vector(), save()

## utils/vector_adapters.py
### Module Functions
- `apply_vector()`
- `get_global_vector_manager()`
### Class Methods
- `VectorAdapter`: __init__(), _check_invariants(), _evaluate_invariant(), apply_vector(), can_handle()
- `PolicyWeightsAdapter`: __init__(), _scale_to_param_range(), apply_vector()
- `CompositionWeightsAdapter`: __init__(), apply_vector()
- `RegionalParamsAdapter`: __init__(), apply_vector()
- `VectorTypeDetector`: __init__(), _create_cartpole_manifest(), _create_composition_manifest(), _create_policy_manifest(), _create_regional_manifest(), _fingerprint_vector(), _parse_manifest(), detect_vector_type()
- `VectorApplicationManager`: __init__(), apply_vector_from_path(), get_supported_types()

## utils/vector_adapters_v2.py
### Module Functions
- `apply_vector_by_id()`
- `get_global_vector_adapter()`
### Class Methods
- `ManifestVectorAdapter`: __init__(), _check_manifest_invariants(), _create_parameter_from_manifest(), _evaluate_manifest_invariant(), _normalize_weight_groups(), _scale_value_using_manifest(), apply_vector_by_id(), apply_vector_with_manifest()

## utils/vector_controller_bridge.py
### Module Functions
- `apply_bridge_to_composition_context()`
- `create_bridge_for_system()`
### Class Methods
- `PolicyVector`: applies_to_context(), applies_to_vector(), is_active()
- `VectorPerformanceTracker`: update_performance()
- `VectorControllerBridge`: __init__(), _apply_composition_weight_policy(), _apply_performance_scaling_policy(), _apply_safety_enforcement_policy(), _apply_selection_bias_policy(), _cleanup_expired_policies(), _create_performance_policies(), _create_policy_from_rule(), _create_safety_policy(), _extract_target_roles(), _log_policy_creation(), _on_composition_result(), _on_parameter_update(), _setup_event_subscriptions(), apply_policies_to_composition(), get_bridge_status(), integrate_with_vector_system(), process_sensing_decision(), update_vector_performance()

## utils/vector_registry.py
### Module Functions
- `get_global_vector_registry()`
- `get_vector_manifest()`
- `query_vectors()`
### Class Methods
- `VectorManifest`: validate()
- `VectorQuery`: __init__(), custom(), domain(), forbid_tags(), require_tags(), with_context()
- `VectorRegistry`: __init__(), _evaluate_predicate(), _explain_selection(), _matches_query(), _score_candidate(), get_manifest(), get_performance_summary(), get_vector_path(), list_vectors(), load_and_register_all(), mark_used(), query_candidates(), register_vector(), set_default_scorer(), update_performance()

## utils/viz.py
### Module Functions
- `_prepare_path()`
- `plot_energy_breakdown()`
- `plot_horizon()`
- `plot_impulse_response()`
- `plot_spectrum()`
