
Lmod is automatically replacing "intel/24.0" with "gcc/15.1.0".


Lmod is automatically replacing "impi/21.11" with "openmpi/5.0.8".


Lmod is automatically replacing "gcc/15.1.0" with "nvidia/25.3".


Lmod is automatically replacing "nvidia/25.3" with "opencilk/2.1.0".

2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124
2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124
2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124
2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu1242026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124

2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu1242026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124

2026-06-03 15:36:19,197 - root - INFO - Torch version: 2.6.0+cu124
2026-06-03 15:36:29,283 - root - INFO - rank 2, begin data loader init2026-06-03 15:36:29,283 - root - INFO - rank 1, begin data loader init
2026-06-03 15:36:29,284 - root - INFO - rank 3, begin data loader init
2026-06-03 15:36:29,283 - root - INFO - No checkpoint found. Starting fresh training run.2026-06-03 15:36:29,283 - root - INFO - rank 2, begin data loader init
2026-06-03 15:36:29,283 - root - INFO - rank 1, begin data loader init

2026-06-03 15:36:29,284 - root - INFO - rank 3, begin data loader init

2026-06-03 15:36:29,283 - root - INFO - No checkpoint found. Starting fresh training run.
2026-06-03 15:36:29,286 - root - INFO - --------------- Versions ---------------
2026-06-03 15:36:29,286 - root - INFO - --------------- Versions ---------------
2026-06-03 15:36:29,287 - root - INFO - Torch: 2.6.0+cu124
2026-06-03 15:36:29,287 - root - INFO - Torch: 2.6.0+cu124
2026-06-03 15:36:29,287 - root - INFO - ----------------------------------------
2026-06-03 15:36:29,287 - root - INFO - ----------------------------------------
2026-06-03 15:36:29,287 - root - INFO - ------------------ Configuration ------------------
2026-06-03 15:36:29,287 - root - INFO - ------------------ Configuration ------------------
2026-06-03 15:36:29,287 - root - INFO - Configuration file: /work/11095/jwan4/PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml
2026-06-03 15:36:29,287 - root - INFO - Configuration file: /work/11095/jwan4/PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml
2026-06-03 15:36:29,287 - root - INFO - Configuration name: SFNO
2026-06-03 15:36:29,287 - root - INFO - Configuration name: SFNO
2026-06-03 15:36:29,287 - root - INFO - nettype sfno_plasim
2026-06-03 15:36:29,287 - root - INFO - nettype sfno_plasim
2026-06-03 15:36:29,287 - root - INFO - scheduler LinearWarmupCosineAnnealingLR
2026-06-03 15:36:29,287 - root - INFO - scheduler LinearWarmupCosineAnnealingLR
2026-06-03 15:36:29,287 - root - INFO - num_warmup_epochs 5
2026-06-03 15:36:29,287 - root - INFO - num_warmup_epochs 5
2026-06-03 15:36:29,287 - root - INFO - warmup_start_lr 1e-08
2026-06-03 15:36:29,287 - root - INFO - warmup_start_lr 1e-08
2026-06-03 15:36:29,288 - root - INFO - eta_min 1e-08
2026-06-03 15:36:29,288 - root - INFO - eta_min 1e-08
2026-06-03 15:36:29,288 - root - INFO - loss raw_l2
2026-06-03 15:36:29,288 - root - INFO - loss raw_l2
2026-06-03 15:36:29,288 - root - INFO - lr 0.0001
2026-06-03 15:36:29,288 - root - INFO - lr 0.0001
2026-06-03 15:36:29,288 - root - INFO - checkpoint_save_interval 1
2026-06-03 15:36:29,288 - root - INFO - checkpoint_save_interval 1
2026-06-03 15:36:29,288 - root - INFO - max_checkpoints_to_keep 1000
2026-06-03 15:36:29,288 - root - INFO - max_checkpoints_to_keep 1000
2026-06-03 15:36:29,288 - root - INFO - use_ema True
2026-06-03 15:36:29,288 - root - INFO - use_ema True
2026-06-03 15:36:29,288 - root - INFO - ema_decay 0.999
2026-06-03 15:36:29,288 - root - INFO - ema_decay 0.999
2026-06-03 15:36:29,288 - root - INFO - ema_warmup_epochs 6
2026-06-03 15:36:29,288 - root - INFO - ema_warmup_epochs 6
2026-06-03 15:36:29,288 - root - INFO - curriculum_learning False
2026-06-03 15:36:29,288 - root - INFO - curriculum_learning False
2026-06-03 15:36:29,288 - root - INFO - ensemble_validation False
2026-06-03 15:36:29,288 - root - INFO - ensemble_validation False
2026-06-03 15:36:29,288 - root - INFO - balanced_learning False
2026-06-03 15:36:29,288 - root - INFO - balanced_learning False
2026-06-03 15:36:29,288 - root - INFO - spectral_transform sht
2026-06-03 15:36:29,288 - root - INFO - spectral_transform sht
2026-06-03 15:36:29,289 - root - INFO - filter_type linear
2026-06-03 15:36:29,289 - root - INFO - filter_type linear
2026-06-03 15:36:29,289 - root - INFO - operator_type dhconv
2026-06-03 15:36:29,289 - root - INFO - operator_type dhconv
2026-06-03 15:36:29,289 - root - INFO - scale_factor 1
2026-06-03 15:36:29,289 - root - INFO - scale_factor 1
2026-06-03 15:36:29,289 - root - INFO - embed_dim 444
2026-06-03 15:36:29,289 - root - INFO - embed_dim 444
2026-06-03 15:36:29,289 - root - INFO - num_layers 12
2026-06-03 15:36:29,289 - root - INFO - num_layers 12
2026-06-03 15:36:29,289 - root - INFO - use_mlp True
2026-06-03 15:36:29,289 - root - INFO - use_mlp True
2026-06-03 15:36:29,289 - root - INFO - mlp_ratio 2.0
2026-06-03 15:36:29,289 - root - INFO - mlp_ratio 2.0
2026-06-03 15:36:29,289 - root - INFO - activation_function gelu
2026-06-03 15:36:29,289 - root - INFO - activation_function gelu
2026-06-03 15:36:29,289 - root - INFO - encoder_layers 1
2026-06-03 15:36:29,289 - root - INFO - encoder_layers 1
2026-06-03 15:36:29,289 - root - INFO - pos_embed True
2026-06-03 15:36:29,289 - root - INFO - pos_embed True
2026-06-03 15:36:29,289 - root - INFO - drop_rate 0.0
2026-06-03 15:36:29,289 - root - INFO - drop_rate 0.0
2026-06-03 15:36:29,289 - root - INFO - drop_path_rate 0.0
2026-06-03 15:36:29,289 - root - INFO - drop_path_rate 0.0
2026-06-03 15:36:29,290 - root - INFO - num_blocks 16
2026-06-03 15:36:29,290 - root - INFO - num_blocks 16
2026-06-03 15:36:29,290 - root - INFO - sparsity_threshold 0.0
2026-06-03 15:36:29,290 - root - INFO - sparsity_threshold 0.0
2026-06-03 15:36:29,290 - root - INFO - normalization_layer instance_norm
2026-06-03 15:36:29,290 - root - INFO - normalization_layer instance_norm
2026-06-03 15:36:29,290 - root - INFO - hard_thresholding_fraction 1.0
2026-06-03 15:36:29,290 - root - INFO - hard_thresholding_fraction 1.0
2026-06-03 15:36:29,290 - root - INFO - use_complex_kernels True
2026-06-03 15:36:29,290 - root - INFO - use_complex_kernels True
2026-06-03 15:36:29,290 - root - INFO - big_skip True
2026-06-03 15:36:29,290 - root - INFO - big_skip True
2026-06-03 15:36:29,290 - root - INFO - rank 1.0
2026-06-03 15:36:29,290 - root - INFO - rank 1.0
2026-06-03 15:36:29,290 - root - INFO - factorization None
2026-06-03 15:36:29,290 - root - INFO - factorization None
2026-06-03 15:36:29,290 - root - INFO - separable False
2026-06-03 15:36:29,290 - root - INFO - separable False
2026-06-03 15:36:29,290 - root - INFO - complex_network True
2026-06-03 15:36:29,290 - root - INFO - complex_network True
2026-06-03 15:36:29,290 - root - INFO - complex_activation real
2026-06-03 15:36:29,290 - root - INFO - complex_activation real
2026-06-03 15:36:29,290 - root - INFO - spectral_layers 3
2026-06-03 15:36:29,290 - root - INFO - spectral_layers 3
2026-06-03 15:36:29,291 - root - INFO - checkpointing 3
2026-06-03 15:36:29,291 - root - INFO - checkpointing 3
2026-06-03 15:36:29,291 - root - INFO - sync_norm True
2026-06-03 15:36:29,291 - root - INFO - sync_norm True
2026-06-03 15:36:29,291 - root - INFO - data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data
2026-06-03 15:36:29,291 - root - INFO - data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data
2026-06-03 15:36:29,291 - root - INFO - bias_data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/bias
2026-06-03 15:36:29,291 - root - INFO - bias_data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/bias
2026-06-03 15:36:29,291 - root - INFO - upper_air_variables ['T', 'U', 'V', 'Z3', 'RELHUM']
2026-06-03 15:36:29,291 - root - INFO - upper_air_variables ['T', 'U', 'V', 'Z3', 'RELHUM']
2026-06-03 15:36:29,291 - root - INFO - surface_variables ['TREFHT', 'U10', 'RHREFHT', 'PS', 'PSL', 'TMQ']
2026-06-03 15:36:29,291 - root - INFO - surface_variables ['TREFHT', 'U10', 'RHREFHT', 'PS', 'PSL', 'TMQ']
2026-06-03 15:36:29,291 - root - INFO - diagnostic_variables ['FSNTOA', 'FSNT', 'PRECT']
2026-06-03 15:36:29,291 - root - INFO - diagnostic_variables ['FSNTOA', 'FSNT', 'PRECT']
2026-06-03 15:36:29,291 - root - INFO - land_variables ['SOILWATER_10CM', 'TSOI_10CM']
2026-06-03 15:36:29,291 - root - INFO - land_variables ['SOILWATER_10CM', 'TSOI_10CM']
2026-06-03 15:36:29,291 - root - INFO - ocean_variables []
2026-06-03 15:36:29,291 - root - INFO - ocean_variables []
2026-06-03 15:36:29,291 - root - INFO - mask_output False
2026-06-03 15:36:29,291 - root - INFO - mask_output False
2026-06-03 15:36:29,291 - root - INFO - constant_boundary_variables ['PCT_GLACIER', 'PFTDATA_MASK', 'PCT_NATVEG', 'TOPO']
2026-06-03 15:36:29,291 - root - INFO - constant_boundary_variables ['PCT_GLACIER', 'PFTDATA_MASK', 'PCT_NATVEG', 'TOPO']
2026-06-03 15:36:29,292 - root - INFO - varying_boundary_variables ['SST', 'ICE', 'sol_in']
2026-06-03 15:36:29,292 - root - INFO - varying_boundary_variables ['SST', 'ICE', 'sol_in']
2026-06-03 15:36:29,292 - root - INFO - train_year_start 2015
2026-06-03 15:36:29,292 - root - INFO - train_year_start 2015
2026-06-03 15:36:29,292 - root - INFO - train_year_end 2040
2026-06-03 15:36:29,292 - root - INFO - train_year_end 2040
2026-06-03 15:36:29,292 - root - INFO - val_year_start 2045
2026-06-03 15:36:29,292 - root - INFO - val_year_start 2045
2026-06-03 15:36:29,292 - root - INFO - val_year_end 2050
2026-06-03 15:36:29,292 - root - INFO - val_year_end 2050
2026-06-03 15:36:29,292 - root - INFO - long_validation True
2026-06-03 15:36:29,292 - root - INFO - long_validation True
2026-06-03 15:36:29,292 - root - INFO - long_val_year_start 2045
2026-06-03 15:36:29,292 - root - INFO - long_val_year_start 2045
2026-06-03 15:36:29,292 - root - INFO - long_rollout_years 5
2026-06-03 15:36:29,292 - root - INFO - long_rollout_years 5
2026-06-03 15:36:29,292 - root - INFO - epochs_per_long_validation 1
2026-06-03 15:36:29,292 - root - INFO - epochs_per_long_validation 1
2026-06-03 15:36:29,292 - root - INFO - mask_fill {'SOILWATER_10CM': 0.0, 'TSOI_10CM': 270.0, 'PCT_GLACIER': 0.0, 'PFTDATA_MASK': 0.0, 'PCT_NATVEG': 0.0, 'TOPO': 0.0, 'SST': 270.0, 'ICE': 0.0}
2026-06-03 15:36:29,292 - root - INFO - mask_fill {'SOILWATER_10CM': 0.0, 'TSOI_10CM': 270.0, 'PCT_GLACIER': 0.0, 'PFTDATA_MASK': 0.0, 'PCT_NATVEG': 0.0, 'TOPO': 0.0, 'SST': 270.0, 'ICE': 0.0}
2026-06-03 15:36:29,292 - root - INFO - data_timedelta_hours 6
2026-06-03 15:36:29,292 - root - INFO - data_timedelta_hours 6
2026-06-03 15:36:29,293 - root - INFO - surface_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - surface_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - surface_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - surface_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - surface_ff_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - surface_ff_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_ff_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - upper_air_ff_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - boundary_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - boundary_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - boundary_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - boundary_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - diagnostic_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - diagnostic_mean data_2015-2050_mean.nc
2026-06-03 15:36:29,293 - root - INFO - diagnostic_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - diagnostic_std data_2015-2050_std_corr.nc
2026-06-03 15:36:29,293 - root - INFO - climatology_file climatology.nc
2026-06-03 15:36:29,293 - root - INFO - climatology_file climatology.nc
2026-06-03 15:36:29,293 - root - INFO - calendar 365_day
2026-06-03 15:36:29,293 - root - INFO - calendar 365_day
2026-06-03 15:36:29,293 - root - INFO - timedelta_hours 6
2026-06-03 15:36:29,293 - root - INFO - timedelta_hours 6
2026-06-03 15:36:29,294 - root - INFO - has_year_zero True
2026-06-03 15:36:29,294 - root - INFO - has_year_zero True
2026-06-03 15:36:29,294 - root - INFO - num_levels 18
2026-06-03 15:36:29,294 - root - INFO - num_levels 18
2026-06-03 15:36:29,294 - root - INFO - use_sigma_levels True
2026-06-03 15:36:29,294 - root - INFO - use_sigma_levels True
2026-06-03 15:36:29,294 - root - INFO - levels [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
2026-06-03 15:36:29,294 - root - INFO - levels [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
2026-06-03 15:36:29,294 - root - INFO - sigma_levels [4.714998332947841, 10.655023096474308, 19.235455601758737, 28.79458853709195, 50.11779996521295, 69.59908688413749, 96.46377266572703, 145.04282239200347, 200.99889546355382, 256.72368590525895, 302.21364012188303, 385.999023919911, 492.46857402252755, 608.6437744215842, 713.7046383204334, 849.6612491105952, 925.5197481473349, 998.4964394917621]
2026-06-03 15:36:29,294 - root - INFO - sigma_levels [4.714998332947841, 10.655023096474308, 19.235455601758737, 28.79458853709195, 50.11779996521295, 69.59908688413749, 96.46377266572703, 145.04282239200347, 200.99889546355382, 256.72368590525895, 302.21364012188303, 385.999023919911, 492.46857402252755, 608.6437744215842, 713.7046383204334, 849.6612491105952, 925.5197481473349, 998.4964394917621]
2026-06-03 15:36:29,294 - root - INFO - horizontal_resolution [180, 360]
2026-06-03 15:36:29,294 - root - INFO - horizontal_resolution [180, 360]
2026-06-03 15:36:29,294 - root - INFO - depths [2, 6, 6, 2]
2026-06-03 15:36:29,294 - root - INFO - depths [2, 6, 6, 2]
2026-06-03 15:36:29,294 - root - INFO - predict_delta False
2026-06-03 15:36:29,294 - root - INFO - predict_delta False
2026-06-03 15:36:29,294 - root - INFO - patch_size [2, 2, 2]
2026-06-03 15:36:29,294 - root - INFO - patch_size [2, 2, 2]
2026-06-03 15:36:29,294 - root - INFO - updown_scale_factor 2
2026-06-03 15:36:29,294 - root - INFO - updown_scale_factor 2
2026-06-03 15:36:29,294 - root - INFO - window_size [2, 2, 4]
2026-06-03 15:36:29,294 - root - INFO - window_size [2, 2, 4]
2026-06-03 15:36:29,294 - root - INFO - epsilon_factor 0.01
2026-06-03 15:36:29,294 - root - INFO - epsilon_factor 0.01
2026-06-03 15:36:29,295 - root - INFO - perturbation_type gaussian_noise
2026-06-03 15:36:29,295 - root - INFO - perturbation_type gaussian_noise
2026-06-03 15:36:29,295 - root - INFO - upper_air_boundary False
2026-06-03 15:36:29,295 - root - INFO - upper_air_boundary False
2026-06-03 15:36:29,295 - root - INFO - subpixel_deconv True
2026-06-03 15:36:29,295 - root - INFO - subpixel_deconv True
2026-06-03 15:36:29,295 - root - INFO - recovery_head True
2026-06-03 15:36:29,295 - root - INFO - recovery_head True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_head False
2026-06-03 15:36:29,295 - root - INFO - diagnostic_head False
2026-06-03 15:36:29,295 - root - INFO - vertical_windowing False
2026-06-03 15:36:29,295 - root - INFO - vertical_windowing False
2026-06-03 15:36:29,295 - root - INFO - train_year_to_year False
2026-06-03 15:36:29,295 - root - INFO - train_year_to_year False
2026-06-03 15:36:29,295 - root - INFO - polar_pad False
2026-06-03 15:36:29,295 - root - INFO - polar_pad False
2026-06-03 15:36:29,295 - root - INFO - grid_has_poles False
2026-06-03 15:36:29,295 - root - INFO - grid_has_poles False
2026-06-03 15:36:29,295 - root - INFO - diagnostic_logs True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_logs True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_acc True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_acc True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_gif True
2026-06-03 15:36:29,295 - root - INFO - diagnostic_gif True
2026-06-03 15:36:29,296 - root - INFO - diagnostic_spectra True
2026-06-03 15:36:29,296 - root - INFO - diagnostic_spectra True
2026-06-03 15:36:29,296 - root - INFO - diagnostic_acc_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_acc_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_gif_var_dict {'Z3': [492.46857402252755], 'U': [492.46857402252755, 256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_gif_var_dict {'Z3': [492.46857402252755], 'U': [492.46857402252755, 256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_spectrum_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_spectrum_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_bias_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - diagnostic_bias_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-03 15:36:29,296 - root - INFO - forecast_lead_times [1, 12, 20, 40, 60]
2026-06-03 15:36:29,296 - root - INFO - forecast_lead_times [1, 12, 20, 40, 60]
2026-06-03 15:36:29,296 - root - INFO - lev lev
2026-06-03 15:36:29,296 - root - INFO - lev lev
2026-06-03 15:36:29,296 - root - INFO - num_inferences 128
2026-06-03 15:36:29,296 - root - INFO - num_inferences 128
2026-06-03 15:36:29,296 - root - INFO - use_reentrant False
2026-06-03 15:36:29,296 - root - INFO - use_reentrant False
2026-06-03 15:36:29,296 - root - INFO - lat [-89.5, -88.5, -87.5, -86.5, -85.5, -84.5, -83.5, -82.5, -81.5, -80.5, -79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5, -71.5, -70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5, -62.5, -61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5, -53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5, -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5, -35.5, -34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5, -26.5, -25.5, -24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5]
2026-06-03 15:36:29,296 - root - INFO - lat [-89.5, -88.5, -87.5, -86.5, -85.5, -84.5, -83.5, -82.5, -81.5, -80.5, -79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5, -71.5, -70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5, -62.5, -61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5, -53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5, -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5, -35.5, -34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5, -26.5, -25.5, -24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5]
2026-06-03 15:36:29,297 - root - INFO - lon [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5, 135.5, 136.5, 137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5, 144.5, 145.5, 146.5, 147.5, 148.5, 149.5, 150.5, 151.5, 152.5, 153.5, 154.5, 155.5, 156.5, 157.5, 158.5, 159.5, 160.5, 161.5, 162.5, 163.5, 164.5, 165.5, 166.5, 167.5, 168.5, 169.5, 170.5, 171.5, 172.5, 173.5, 174.5, 175.5, 176.5, 177.5, 178.5, 179.5, 180.5, 181.5, 182.5, 183.5, 184.5, 185.5, 186.5, 187.5, 188.5, 189.5, 190.5, 191.5, 192.5, 193.5, 194.5, 195.5, 196.5, 197.5, 198.5, 199.5, 200.5, 201.5, 202.5, 203.5, 204.5, 205.5, 206.5, 207.5, 208.5, 209.5, 210.5, 211.5, 212.5, 213.5, 214.5, 215.5, 216.5, 217.5, 218.5, 219.5, 220.5, 221.5, 222.5, 223.5, 224.5, 225.5, 226.5, 227.5, 228.5, 229.5, 230.5, 231.5, 232.5, 233.5, 234.5, 235.5, 236.5, 237.5, 238.5, 239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 247.5, 248.5, 249.5, 250.5, 251.5, 252.5, 253.5, 254.5, 255.5, 256.5, 257.5, 258.5, 259.5, 260.5, 261.5, 262.5, 263.5, 264.5, 265.5, 266.5, 267.5, 268.5, 269.5, 270.5, 271.5, 272.5, 273.5, 274.5, 275.5, 276.5, 277.5, 278.5, 279.5, 280.5, 281.5, 282.5, 283.5, 284.5, 285.5, 286.5, 287.5, 288.5, 289.5, 290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5, 297.5, 298.5, 299.5, 300.5, 301.5, 302.5, 303.5, 304.5, 305.5, 306.5, 307.5, 308.5, 309.5, 310.5, 311.5, 312.5, 313.5, 314.5, 315.5, 316.5, 317.5, 318.5, 319.5, 320.5, 321.5, 322.5, 323.5, 324.5, 325.5, 326.5, 327.5, 328.5, 329.5, 330.5, 331.5, 332.5, 333.5, 334.5, 335.5, 336.5, 337.5, 338.5, 339.5, 340.5, 341.5, 342.5, 343.5, 344.5, 345.5, 346.5, 347.5, 348.5, 349.5, 350.5, 351.5, 352.5, 353.5, 354.5, 355.5, 356.5, 357.5, 358.5, 359.5]
2026-06-03 15:36:29,297 - root - INFO - lon [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5, 135.5, 136.5, 137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5, 144.5, 145.5, 146.5, 147.5, 148.5, 149.5, 150.5, 151.5, 152.5, 153.5, 154.5, 155.5, 156.5, 157.5, 158.5, 159.5, 160.5, 161.5, 162.5, 163.5, 164.5, 165.5, 166.5, 167.5, 168.5, 169.5, 170.5, 171.5, 172.5, 173.5, 174.5, 175.5, 176.5, 177.5, 178.5, 179.5, 180.5, 181.5, 182.5, 183.5, 184.5, 185.5, 186.5, 187.5, 188.5, 189.5, 190.5, 191.5, 192.5, 193.5, 194.5, 195.5, 196.5, 197.5, 198.5, 199.5, 200.5, 201.5, 202.5, 203.5, 204.5, 205.5, 206.5, 207.5, 208.5, 209.5, 210.5, 211.5, 212.5, 213.5, 214.5, 215.5, 216.5, 217.5, 218.5, 219.5, 220.5, 221.5, 222.5, 223.5, 224.5, 225.5, 226.5, 227.5, 228.5, 229.5, 230.5, 231.5, 232.5, 233.5, 234.5, 235.5, 236.5, 237.5, 238.5, 239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 247.5, 248.5, 249.5, 250.5, 251.5, 252.5, 253.5, 254.5, 255.5, 256.5, 257.5, 258.5, 259.5, 260.5, 261.5, 262.5, 263.5, 264.5, 265.5, 266.5, 267.5, 268.5, 269.5, 270.5, 271.5, 272.5, 273.5, 274.5, 275.5, 276.5, 277.5, 278.5, 279.5, 280.5, 281.5, 282.5, 283.5, 284.5, 285.5, 286.5, 287.5, 288.5, 289.5, 290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5, 297.5, 298.5, 299.5, 300.5, 301.5, 302.5, 303.5, 304.5, 305.5, 306.5, 307.5, 308.5, 309.5, 310.5, 311.5, 312.5, 313.5, 314.5, 315.5, 316.5, 317.5, 318.5, 319.5, 320.5, 321.5, 322.5, 323.5, 324.5, 325.5, 326.5, 327.5, 328.5, 329.5, 330.5, 331.5, 332.5, 333.5, 334.5, 335.5, 336.5, 337.5, 338.5, 339.5, 340.5, 341.5, 342.5, 343.5, 344.5, 345.5, 346.5, 347.5, 348.5, 349.5, 350.5, 351.5, 352.5, 353.5, 354.5, 355.5, 356.5, 357.5, 358.5, 359.5]
2026-06-03 15:36:29,297 - root - INFO - loglr -13
2026-06-03 15:36:29,297 - root - INFO - loglr -13
2026-06-03 15:36:29,297 - root - INFO - weight_decay 3e-06
2026-06-03 15:36:29,297 - root - INFO - weight_decay 3e-06
2026-06-03 15:36:29,297 - root - INFO - num_data_workers 4
2026-06-03 15:36:29,297 - root - INFO - num_data_workers 4
2026-06-03 15:36:29,297 - root - INFO - batch_size 4
2026-06-03 15:36:29,297 - root - INFO - batch_size 4
2026-06-03 15:36:29,297 - root - INFO - max_epochs 100
2026-06-03 15:36:29,297 - root - INFO - max_epochs 100
2026-06-03 15:36:29,297 - root - INFO - no_leap_year 2020
2026-06-03 15:36:29,297 - root - INFO - no_leap_year 2020
2026-06-03 15:36:29,297 - root - INFO - leap_year 2020
2026-06-03 15:36:29,297 - root - INFO - leap_year 2020
2026-06-03 15:36:29,297 - root - INFO - log_to_screen True
2026-06-03 15:36:29,297 - root - INFO - log_to_screen True
2026-06-03 15:36:29,297 - root - INFO - log_to_wandb True
2026-06-03 15:36:29,297 - root - INFO - log_to_wandb True
2026-06-03 15:36:29,297 - root - INFO - save_checkpoint True
2026-06-03 15:36:29,297 - root - INFO - save_checkpoint True
2026-06-03 15:36:29,297 - root - INFO - save_forecasts True
2026-06-03 15:36:29,297 - root - INFO - save_forecasts True
2026-06-03 15:36:29,298 - root - INFO - optimizer_type AdamW
2026-06-03 15:36:29,298 - root - INFO - optimizer_type AdamW
2026-06-03 15:36:29,298 - root - INFO - plot_animations False
2026-06-03 15:36:29,298 - root - INFO - plot_animations False
2026-06-03 15:36:29,298 - root - INFO - group plasim
2026-06-03 15:36:29,298 - root - INFO - group plasim
2026-06-03 15:36:29,298 - root - INFO - exp_dir /work/11095/jwan4/PanguWeather/v2.0/results
2026-06-03 15:36:29,298 - root - INFO - exp_dir /work/11095/jwan4/PanguWeather/v2.0/results
2026-06-03 15:36:29,298 - root - INFO - enable_fp8 False
2026-06-03 15:36:29,298 - root - INFO - enable_fp8 False
2026-06-03 15:36:29,298 - root - INFO - fresh_start False
2026-06-03 15:36:29,298 - root - INFO - fresh_start False
2026-06-03 15:36:29,298 - root - INFO - use_transformer_engine False
2026-06-03 15:36:29,298 - root - INFO - use_transformer_engine False
2026-06-03 15:36:29,298 - root - INFO - early_stopping False
2026-06-03 15:36:29,298 - root - INFO - early_stopping False
2026-06-03 15:36:29,298 - root - INFO - entity jesswan-university-of-chicago
2026-06-03 15:36:29,298 - root - INFO - entity jesswan-university-of-chicago
2026-06-03 15:36:29,298 - root - INFO - project E3SM-SRM-SFNO
2026-06-03 15:36:29,298 - root - INFO - project E3SM-SRM-SFNO
2026-06-03 15:36:29,298 - root - INFO - name E3SM-SRM-SFNO-CTL_SST0051_REST0101-0013
2026-06-03 15:36:29,298 - root - INFO - name E3SM-SRM-SFNO-CTL_SST0051_REST0101-0013
2026-06-03 15:36:29,298 - root - INFO - enable_amp True
2026-06-03 15:36:29,298 - root - INFO - enable_amp True
2026-06-03 15:36:29,299 - root - INFO - amp_dtype bfloat16
2026-06-03 15:36:29,299 - root - INFO - amp_dtype bfloat16
2026-06-03 15:36:29,299 - root - INFO - use_zero_optimizer True
2026-06-03 15:36:29,299 - root - INFO - use_zero_optimizer True
2026-06-03 15:36:29,299 - root - INFO - vae_loss False
2026-06-03 15:36:29,299 - root - INFO - vae_loss False
2026-06-03 15:36:29,299 - root - INFO - mode train
2026-06-03 15:36:29,299 - root - INFO - mode train
2026-06-03 15:36:29,299 - root - INFO - test_iterations 30
2026-06-03 15:36:29,299 - root - INFO - test_iterations 30
2026-06-03 15:36:29,299 - root - INFO - run_iter 1
2026-06-03 15:36:29,299 - root - INFO - run_iter 1
2026-06-03 15:36:29,299 - root - INFO - use_legacy_model False
2026-06-03 15:36:29,299 - root - INFO - use_legacy_model False
2026-06-03 15:36:29,299 - root - INFO - has_diagnostic True
2026-06-03 15:36:29,299 - root - INFO - has_diagnostic True
2026-06-03 15:36:29,299 - root - INFO - num_ensemble_members 1
2026-06-03 15:36:29,299 - root - INFO - num_ensemble_members 1
2026-06-03 15:36:29,299 - root - INFO - just_validate False
2026-06-03 15:36:29,299 - root - INFO - just_validate False
2026-06-03 15:36:29,299 - root - INFO - validation_epochs []
2026-06-03 15:36:29,299 - root - INFO - validation_epochs []
2026-06-03 15:36:29,299 - root - INFO - validate_before_train False
2026-06-03 15:36:29,299 - root - INFO - validate_before_train False
2026-06-03 15:36:29,300 - root - INFO - debug False
2026-06-03 15:36:29,300 - root - INFO - debug False
2026-06-03 15:36:29,300 - root - INFO - world_size 4
2026-06-03 15:36:29,300 - root - INFO - world_size 4
2026-06-03 15:36:29,300 - root - INFO - global_batch_size 16
2026-06-03 15:36:29,300 - root - INFO - global_batch_size 16
2026-06-03 15:36:29,300 - root - INFO - seed 0
2026-06-03 15:36:29,300 - root - INFO - seed 0
2026-06-03 15:36:29,300 - root - INFO - experiment_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013
2026-06-03 15:36:29,300 - root - INFO - experiment_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013
2026-06-03 15:36:29,300 - root - INFO - checkpoint_dir_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints
2026-06-03 15:36:29,300 - root - INFO - checkpoint_dir_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints
2026-06-03 15:36:29,300 - root - INFO - checkpoint_dir_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints
2026-06-03 15:36:29,300 - root - INFO - checkpoint_dir_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints
2026-06-03 15:36:29,300 - root - INFO - plots_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots
2026-06-03 15:36:29,300 - root - INFO - plots_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots
2026-06-03 15:36:29,300 - root - INFO - spectra_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/spectra
2026-06-03 15:36:29,300 - root - INFO - spectra_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/spectra
2026-06-03 15:36:29,300 - root - INFO - acc_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/acc
2026-06-03 15:36:29,300 - root - INFO - acc_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/acc
2026-06-03 15:36:29,300 - root - INFO - gif_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/gif
2026-06-03 15:36:29,300 - root - INFO - gif_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/gif
2026-06-03 15:36:29,300 - root - INFO - bias_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/bias
2026-06-03 15:36:29,300 - root - INFO - bias_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/bias
2026-06-03 15:36:29,300 - root - INFO - validation_data_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/validation_data
2026-06-03 15:36:29,300 - root - INFO - validation_data_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/validation_data
2026-06-03 15:36:29,301 - root - INFO - checkpoint_path_globstr_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_epoch_*.tar
2026-06-03 15:36:29,301 - root - INFO - checkpoint_path_globstr_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_epoch_*.tar
2026-06-03 15:36:29,301 - root - INFO - checkpoint_path_globstr_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_epoch_*.tar
2026-06-03 15:36:29,301 - root - INFO - checkpoint_path_globstr_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_epoch_*.tar
2026-06-03 15:36:29,301 - root - INFO - best_checkpoint_path_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/best_ckpt.tar
2026-06-03 15:36:29,301 - root - INFO - best_checkpoint_path_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/best_ckpt.tar
2026-06-03 15:36:29,301 - root - INFO - best_checkpoint_path_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/best_ckpt.tar
2026-06-03 15:36:29,301 - root - INFO - best_checkpoint_path_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/best_ckpt.tar
2026-06-03 15:36:29,301 - root - INFO - latest_checkpoint_path_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_latest.tar
2026-06-03 15:36:29,301 - root - INFO - latest_checkpoint_path_save /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_latest.tar
2026-06-03 15:36:29,301 - root - INFO - latest_checkpoint_path_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_latest.tar
2026-06-03 15:36:29,301 - root - INFO - latest_checkpoint_path_load /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/checkpoints/ckpt_latest.tar
2026-06-03 15:36:29,301 - root - INFO - plot_save_interval 10
2026-06-03 15:36:29,301 - root - INFO - plot_save_interval 10
2026-06-03 15:36:29,301 - root - INFO - max_plots_to_keep 5
2026-06-03 15:36:29,301 - root - INFO - max_plots_to_keep 5
2026-06-03 15:36:29,301 - root - INFO - resuming False
2026-06-03 15:36:29,301 - root - INFO - resuming False
2026-06-03 15:36:29,301 - root - INFO - finetuning False
2026-06-03 15:36:29,301 - root - INFO - finetuning False
2026-06-03 15:36:29,301 - root - INFO - local_rank 0
2026-06-03 15:36:29,301 - root - INFO - local_rank 0
2026-06-03 15:36:29,301 - root - INFO - ---------------------------------------------------
2026-06-03 15:36:29,301 - root - INFO - ---------------------------------------------------
2026-06-03 15:36:29,318 - root - INFO - Initialized wandb_step: 0
2026-06-03 15:36:29,318 - root - INFO - Initialized wandb_step: 0
2026-06-03 15:36:29,318 - root - INFO - rank 0, begin data loader init
2026-06-03 15:36:29,318 - root - INFO - rank 0, begin data loader init
2026-06-03 15:38:47,256 - root - INFO - Params2026-06-03 15:38:47,256 - root - INFO - Params
2026-06-03 15:38:47,256 - root - INFO - Params2026-06-03 15:38:47,256 - root - INFO - Params2026-06-03 15:38:47,256 - root - INFO - rank 0, data loader initialized


2026-06-03 15:38:47,256 - root - INFO - Params
2026-06-03 15:38:47,256 - root - INFO - Params

2026-06-03 15:38:47,256 - root - INFO - rank 0, data loader initialized
2026-06-03 15:38:47,259 - root - INFO - Output directories under: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013
2026-06-03 15:38:47,259 - root - INFO - Output directories under: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013
2026-06-03 15:38:47,259 - root - INFO -   Spectra: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/spectra
2026-06-03 15:38:47,259 - root - INFO -   Spectra: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/spectra
2026-06-03 15:38:47,259 - root - INFO -   GIFs: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/gif
2026-06-03 15:38:47,259 - root - INFO -   GIFs: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/gif
2026-06-03 15:38:47,259 - root - INFO -   ACC plots: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/acc
2026-06-03 15:38:47,259 - root - INFO -   ACC plots: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/acc
2026-06-03 15:38:47,259 - root - INFO -   Bias plots: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/bias
2026-06-03 15:38:47,259 - root - INFO -   Bias plots: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0013/plots/bias
2026-06-03 15:38:47,259 - root - INFO - WandB resume mode: never
2026-06-03 15:38:47,259 - root - INFO - WandB resume mode: never
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home1/11095/jwan4/.netrc.
wandb: Currently logged in as: jesswan (jesswan-university-of-chicago) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.27.0
wandb: Run data is saved locally in /work2/11095/jwan4/PanguWeather/v2.0/wandb/run-20260603_153847-jn7cnljo
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run E3SM-SRM-SFNO-CTL_SST0051_REST0101-0013-1
wandb: ⭐️ View project at https://wandb.ai/jesswan-university-of-chicago/E3SM-SRM-SFNO
wandb: 🚀 View run at https://wandb.ai/jesswan-university-of-chicago/E3SM-SRM-SFNO/runs/jn7cnljo
wandb: Detected [huggingface_hub.inference] in use.
wandb: Use W&B Weave for improved LLM call tracing. Install Weave with `pip install weave` then add `import weave` to the top of your script.
wandb: For more information, check out the docs at: https://weave-docs.wandb.ai
2026-06-03 15:38:49,368 - root - INFO - WandB initialized with config: <utils.YParams.YParams object at 0x14cd3ddbdf90>
2026-06-03 15:38:49,368 - root - INFO - WandB initialized with config: <utils.YParams.YParams object at 0x14cd3ddbdf90>
2026-06-03 15:38:49,423 - root - INFO - Params
2026-06-03 15:38:49,423 - root - INFO - Params
2026-06-03 15:38:53,636 - root - INFO - Rank 2 Loaded EMA with decay = 0.999
2026-06-03 15:38:53,636 - root - INFO - Rank 2 Loaded EMA with decay = 0.999
2026-06-03 15:38:53,639 - root - INFO - Rank 1 Loaded EMA with decay = 0.999
2026-06-03 15:38:53,639 - root - INFO - Rank 1 Loaded EMA with decay = 0.999
2026-06-03 15:38:53,648 - root - INFO - Rank 3 Loaded EMA with decay = 0.999
2026-06-03 15:38:53,648 - root - INFO - Rank 3 Loaded EMA with decay = 0.999
2026-06-03 15:38:54,810 - root - INFO - Rank 0 Loaded EMA with decay = 0.999
2026-06-03 15:38:54,810 - root - INFO - Rank 0 Loaded EMA with decay = 0.999
2026-06-03 15:38:55,715 - root - INFO - Using ZeroRedundancyOptimizer (Stage 1) wrapping AdamW across 4 ranks.2026-06-03 15:38:55,715 - root - INFO - Losses is setup2026-06-03 15:38:55,715 - root - INFO - Losses is setup


2026-06-03 15:38:55,715 - root - INFO - Losses is setup
2026-06-03 15:38:55,715 - root - INFO - Losses is setup
2026-06-03 15:38:55,715 - root - INFO - Using ZeroRedundancyOptimizer (Stage 1) wrapping AdamW across 4 ranks.2026-06-03 15:38:55,715 - root - INFO - Losses is setup

2026-06-03 15:38:55,715 - root - INFO - Starting fresh training run2026-06-03 15:38:55,715 - root - INFO - Starting fresh training run

2026-06-03 15:38:55,715 - root - INFO - Losses is setup2026-06-03 15:38:55,715 - root - INFO - Starting fresh training run
2026-06-03 15:38:55,715 - root - INFO - Starting fresh training run

2026-06-03 15:38:55,716 - root - INFO - Starting fresh training run
2026-06-03 15:38:55,716 - root - INFO - Losses is setup2026-06-03 15:38:55,716 - root - INFO - Starting fresh training run

2026-06-03 15:38:55,716 - root - INFO - Losses is setup2026-06-03 15:38:55,716 - root - INFO - Expected total batches: 22812026-06-03 15:38:55,716 - root - INFO - Expected total batches: 2281

2026-06-03 15:38:55,716 - root - INFO - Expected total batches: 22812026-06-03 15:38:55,716 - root - INFO - Expected total batches: 2281


2026-06-03 15:38:55,716 - root - INFO - Expected total batches: 2281
2026-06-03 15:38:55,716 - root - INFO - Starting fresh training run2026-06-03 15:38:55,716 - root - INFO - Expected total batches: 2281

2026-06-03 15:38:55,716 - root - INFO - Starting fresh training run
2026-06-03 15:38:55,717 - root - INFO - Number of trainable model parameters: 892808076
2026-06-03 15:38:55,717 - root - INFO - Number of trainable model parameters: 892808076
2026-06-03 15:38:55,718 - root - INFO - Starting Training Loop...
2026-06-03 15:38:55,718 - root - INFO - Starting Training Loop...
2026-06-03 15:38:55,718 - root - INFO - Starting epoch 1/100
2026-06-03 15:38:55,718 - root - INFO - Starting epoch 1/100
2026-06-03 15:38:55,719 - root - INFO - Expected total batches: 2281
2026-06-03 15:38:55,719 - root - INFO - Expected total batches: 2281
  0%|                              | 0/2281 [00:00<?, ?it/s]Epoch [1/100], Year 2015, Loss: 0.8915:   0%|                              | 0/2281 [03:58<?, ?it/s]Epoch [1/100], Year 2015, Loss: 0.8915:   0%|                              | 1/2281 [03:58<151:11:11, 238.72s/it]Epoch [1/100], Year 2015, Loss: 0.5684:   0%|                              | 1/2281 [07:25<151:11:11, 238.72s/it]Epoch [1/100], Year 2015, Loss: 0.5684:   0%|                              | 2/2281 [07:25<139:14:48, 219.96s/it]Epoch [1/100], Year 2015, Loss: 0.3035:   0%|                              | 2/2281 [10:52<139:14:48, 219.96s/it]Epoch [1/100], Year 2015, Loss: 0.3035:   0%|                              | 3/2281 [10:52<135:23:33, 213.97s/it]Epoch [1/100], Year 2015, Loss: 0.1802:   0%|                              | 3/2281 [14:19<135:23:33, 213.97s/it]Epoch [1/100], Year 2015, Loss: 0.1802:   0%|                              | 4/2281 [14:19<133:33:10, 211.15s/it]Epoch [1/100], Year 2015, Loss: 0.1133:   0%|                              | 4/2281 [17:57<133:33:10, 211.15s/it]Epoch [1/100], Year 2015, Loss: 0.1133:   0%|                              | 5/2281 [17:57<135:12:25, 213.86s/it]Epoch [1/100], Year 2015, Loss: 0.0781:   0%|                              | 5/2281 [21:24<135:12:25, 213.86s/it]Epoch [1/100], Year 2015, Loss: 0.0781:   0%|                              | 6/2281 [21:24<133:38:16, 211.47s/it]Epoch [1/100], Year 2015, Loss: 0.0620:   0%|                              | 6/2281 [24:51<133:38:16, 211.47s/it]Epoch [1/100], Year 2015, Loss: 0.0620:   0%|                              | 7/2281 [24:51<132:37:16, 209.95s/it]slurmstepd: error: *** JOB 3178772 ON c561-002 CANCELLED AT 2026-06-03T16:04:59 DUE TO TIME LIMIT ***
