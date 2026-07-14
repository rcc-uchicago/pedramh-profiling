
Lmod is automatically replacing "intel/24.0" with "gcc/15.1.0".


Lmod is automatically replacing "impi/21.11" with "openmpi/5.0.8".


Lmod is automatically replacing "gcc/15.1.0" with "nvidia/25.3".


Lmod is automatically replacing "nvidia/25.3" with "opencilk/2.1.0".

2026-06-16 17:08:19,546 - root - INFO - Resuming from existing checkpoint.
2026-06-16 17:08:19,549 - root - INFO - --------------- Versions ---------------
2026-06-16 17:08:19,550 - root - INFO - Torch: 2.6.0+cu124
2026-06-16 17:08:19,550 - root - INFO - ----------------------------------------
2026-06-16 17:08:19,550 - root - INFO - ------------------ Configuration ------------------
2026-06-16 17:08:19,550 - root - INFO - Configuration file: /work/11095/jwan4/PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml
2026-06-16 17:08:19,550 - root - INFO - Configuration name: SFNO
2026-06-16 17:08:19,550 - root - INFO - nettype sfno_plasim
2026-06-16 17:08:19,550 - root - INFO - scheduler LinearWarmupCosineAnnealingLR
2026-06-16 17:08:19,550 - root - INFO - num_warmup_epochs 5
2026-06-16 17:08:19,550 - root - INFO - warmup_start_lr 1e-08
2026-06-16 17:08:19,550 - root - INFO - eta_min 1e-08
2026-06-16 17:08:19,550 - root - INFO - loss raw_l2
2026-06-16 17:08:19,550 - root - INFO - lr 0.0001
2026-06-16 17:08:19,550 - root - INFO - checkpoint_save_interval 1
2026-06-16 17:08:19,550 - root - INFO - max_checkpoints_to_keep 1000
2026-06-16 17:08:19,551 - root - INFO - use_ema True
2026-06-16 17:08:19,551 - root - INFO - ema_decay 0.999
2026-06-16 17:08:19,551 - root - INFO - ema_warmup_epochs 6
2026-06-16 17:08:19,551 - root - INFO - curriculum_learning False
2026-06-16 17:08:19,551 - root - INFO - ensemble_validation False
2026-06-16 17:08:19,551 - root - INFO - balanced_learning False
2026-06-16 17:08:19,551 - root - INFO - spectral_transform sht
2026-06-16 17:08:19,551 - root - INFO - filter_type linear
2026-06-16 17:08:19,551 - root - INFO - operator_type dhconv
2026-06-16 17:08:19,551 - root - INFO - scale_factor 1
2026-06-16 17:08:19,551 - root - INFO - embed_dim 444
2026-06-16 17:08:19,551 - root - INFO - num_layers 12
2026-06-16 17:08:19,551 - root - INFO - use_mlp True
2026-06-16 17:08:19,551 - root - INFO - mlp_ratio 2.0
2026-06-16 17:08:19,551 - root - INFO - activation_function gelu
2026-06-16 17:08:19,551 - root - INFO - encoder_layers 1
2026-06-16 17:08:19,551 - root - INFO - pos_embed True
2026-06-16 17:08:19,551 - root - INFO - drop_rate 0.0
2026-06-16 17:08:19,551 - root - INFO - drop_path_rate 0.0
2026-06-16 17:08:19,552 - root - INFO - num_blocks 16
2026-06-16 17:08:19,552 - root - INFO - sparsity_threshold 0.0
2026-06-16 17:08:19,552 - root - INFO - normalization_layer instance_norm
2026-06-16 17:08:19,552 - root - INFO - hard_thresholding_fraction 1.0
2026-06-16 17:08:19,552 - root - INFO - use_complex_kernels True
2026-06-16 17:08:19,552 - root - INFO - big_skip True
2026-06-16 17:08:19,552 - root - INFO - rank 1.0
2026-06-16 17:08:19,552 - root - INFO - factorization None
2026-06-16 17:08:19,552 - root - INFO - separable False
2026-06-16 17:08:19,552 - root - INFO - complex_network True
2026-06-16 17:08:19,552 - root - INFO - complex_activation real
2026-06-16 17:08:19,552 - root - INFO - spectral_layers 3
2026-06-16 17:08:19,552 - root - INFO - checkpointing 2
2026-06-16 17:08:19,552 - root - INFO - sync_norm True
2026-06-16 17:08:19,552 - root - INFO - data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data
2026-06-16 17:08:19,552 - root - INFO - bias_data_dir /scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/bias
2026-06-16 17:08:19,552 - root - INFO - upper_air_variables ['T', 'U', 'V', 'Z3', 'RELHUM']
2026-06-16 17:08:19,552 - root - INFO - surface_variables ['TREFHT', 'U10', 'RHREFHT', 'PS', 'PSL', 'TMQ']
2026-06-16 17:08:19,552 - root - INFO - diagnostic_variables ['FSNTOA', 'FSNT', 'PRECT']
2026-06-16 17:08:19,553 - root - INFO - land_variables ['SOILWATER_10CM', 'TSOI_10CM']
2026-06-16 17:08:19,553 - root - INFO - ocean_variables []
2026-06-16 17:08:19,553 - root - INFO - mask_output False
2026-06-16 17:08:19,553 - root - INFO - constant_boundary_variables ['PCT_GLACIER', 'PFTDATA_MASK', 'PCT_NATVEG', 'TOPO']
2026-06-16 17:08:19,553 - root - INFO - varying_boundary_variables ['SST', 'ICE', 'sol_in']
2026-06-16 17:08:19,553 - root - INFO - train_year_start 2015
2026-06-16 17:08:19,553 - root - INFO - train_year_end 2040
2026-06-16 17:08:19,553 - root - INFO - val_year_start 2045
2026-06-16 17:08:19,553 - root - INFO - val_year_end 2050
2026-06-16 17:08:19,553 - root - INFO - long_validation True
2026-06-16 17:08:19,553 - root - INFO - long_val_year_start 2045
2026-06-16 17:08:19,553 - root - INFO - long_rollout_years 5
2026-06-16 17:08:19,553 - root - INFO - epochs_per_long_validation 1
2026-06-16 17:08:19,553 - root - INFO - mask_fill {'SOILWATER_10CM': 0.0, 'TSOI_10CM': 270.0, 'PCT_GLACIER': 0.0, 'PFTDATA_MASK': 0.0, 'PCT_NATVEG': 0.0, 'TOPO': 0.0, 'SST': 270.0, 'ICE': 0.0}
2026-06-16 17:08:19,553 - root - INFO - data_timedelta_hours 6
2026-06-16 17:08:19,553 - root - INFO - surface_mean data_2015-2050_mean.nc
2026-06-16 17:08:19,553 - root - INFO - surface_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,553 - root - INFO - surface_ff_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,554 - root - INFO - upper_air_mean data_2015-2050_mean.nc
2026-06-16 17:08:19,554 - root - INFO - upper_air_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,554 - root - INFO - upper_air_ff_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,554 - root - INFO - boundary_mean data_2015-2050_mean.nc
2026-06-16 17:08:19,554 - root - INFO - boundary_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,554 - root - INFO - diagnostic_mean data_2015-2050_mean.nc
2026-06-16 17:08:19,554 - root - INFO - diagnostic_std data_2015-2050_std_corr.nc
2026-06-16 17:08:19,554 - root - INFO - climatology_file climatology.nc
2026-06-16 17:08:19,554 - root - INFO - calendar 365_day
2026-06-16 17:08:19,554 - root - INFO - timedelta_hours 6
2026-06-16 17:08:19,554 - root - INFO - has_year_zero True
2026-06-16 17:08:19,554 - root - INFO - num_levels 18
2026-06-16 17:08:19,554 - root - INFO - use_sigma_levels True
2026-06-16 17:08:19,554 - root - INFO - levels [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
2026-06-16 17:08:19,554 - root - INFO - sigma_levels [4.714998332947841, 10.655023096474308, 19.235455601758737, 28.79458853709195, 50.11779996521295, 69.59908688413749, 96.46377266572703, 145.04282239200347, 200.99889546355382, 256.72368590525895, 302.21364012188303, 385.999023919911, 492.46857402252755, 608.6437744215842, 713.7046383204334, 849.6612491105952, 925.5197481473349, 998.4964394917621]
2026-06-16 17:08:19,554 - root - INFO - horizontal_resolution [180, 360]
2026-06-16 17:08:19,554 - root - INFO - depths [2, 6, 6, 2]
2026-06-16 17:08:19,554 - root - INFO - predict_delta False
2026-06-16 17:08:19,554 - root - INFO - patch_size [2, 2, 2]
2026-06-16 17:08:19,555 - root - INFO - updown_scale_factor 2
2026-06-16 17:08:19,555 - root - INFO - window_size [2, 2, 4]
2026-06-16 17:08:19,555 - root - INFO - epsilon_factor 0.01
2026-06-16 17:08:19,555 - root - INFO - perturbation_type gaussian_noise
2026-06-16 17:08:19,555 - root - INFO - upper_air_boundary False
2026-06-16 17:08:19,555 - root - INFO - subpixel_deconv True
2026-06-16 17:08:19,555 - root - INFO - recovery_head True
2026-06-16 17:08:19,555 - root - INFO - diagnostic_head False
2026-06-16 17:08:19,555 - root - INFO - vertical_windowing False
2026-06-16 17:08:19,555 - root - INFO - train_year_to_year False
2026-06-16 17:08:19,555 - root - INFO - polar_pad False
2026-06-16 17:08:19,555 - root - INFO - grid_has_poles False
2026-06-16 17:08:19,555 - root - INFO - diagnostic_logs True
2026-06-16 17:08:19,555 - root - INFO - diagnostic_acc True
2026-06-16 17:08:19,555 - root - INFO - diagnostic_gif True
2026-06-16 17:08:19,555 - root - INFO - diagnostic_spectra True
2026-06-16 17:08:19,555 - root - INFO - diagnostic_acc_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-16 17:08:19,555 - root - INFO - diagnostic_gif_var_dict {'Z3': [492.46857402252755], 'U': [492.46857402252755, 256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-16 17:08:19,555 - root - INFO - diagnostic_spectrum_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-16 17:08:19,556 - root - INFO - diagnostic_bias_var_dict {'Z3': [492.46857402252755], 'U': [256.72368590525895], 'T': [849.6612491105952], 'TREFHT': []}
2026-06-16 17:08:19,556 - root - INFO - forecast_lead_times [1, 12, 20, 40, 60]
2026-06-16 17:08:19,556 - root - INFO - lev lev
2026-06-16 17:08:19,556 - root - INFO - num_inferences 128
2026-06-16 17:08:19,556 - root - INFO - use_reentrant False
2026-06-16 17:08:19,556 - root - INFO - lat [-89.5, -88.5, -87.5, -86.5, -85.5, -84.5, -83.5, -82.5, -81.5, -80.5, -79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5, -71.5, -70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5, -62.5, -61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5, -53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5, -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5, -35.5, -34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5, -26.5, -25.5, -24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5]
2026-06-16 17:08:19,556 - root - INFO - lon [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5, 135.5, 136.5, 137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5, 144.5, 145.5, 146.5, 147.5, 148.5, 149.5, 150.5, 151.5, 152.5, 153.5, 154.5, 155.5, 156.5, 157.5, 158.5, 159.5, 160.5, 161.5, 162.5, 163.5, 164.5, 165.5, 166.5, 167.5, 168.5, 169.5, 170.5, 171.5, 172.5, 173.5, 174.5, 175.5, 176.5, 177.5, 178.5, 179.5, 180.5, 181.5, 182.5, 183.5, 184.5, 185.5, 186.5, 187.5, 188.5, 189.5, 190.5, 191.5, 192.5, 193.5, 194.5, 195.5, 196.5, 197.5, 198.5, 199.5, 200.5, 201.5, 202.5, 203.5, 204.5, 205.5, 206.5, 207.5, 208.5, 209.5, 210.5, 211.5, 212.5, 213.5, 214.5, 215.5, 216.5, 217.5, 218.5, 219.5, 220.5, 221.5, 222.5, 223.5, 224.5, 225.5, 226.5, 227.5, 228.5, 229.5, 230.5, 231.5, 232.5, 233.5, 234.5, 235.5, 236.5, 237.5, 238.5, 239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 247.5, 248.5, 249.5, 250.5, 251.5, 252.5, 253.5, 254.5, 255.5, 256.5, 257.5, 258.5, 259.5, 260.5, 261.5, 262.5, 263.5, 264.5, 265.5, 266.5, 267.5, 268.5, 269.5, 270.5, 271.5, 272.5, 273.5, 274.5, 275.5, 276.5, 277.5, 278.5, 279.5, 280.5, 281.5, 282.5, 283.5, 284.5, 285.5, 286.5, 287.5, 288.5, 289.5, 290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5, 297.5, 298.5, 299.5, 300.5, 301.5, 302.5, 303.5, 304.5, 305.5, 306.5, 307.5, 308.5, 309.5, 310.5, 311.5, 312.5, 313.5, 314.5, 315.5, 316.5, 317.5, 318.5, 319.5, 320.5, 321.5, 322.5, 323.5, 324.5, 325.5, 326.5, 327.5, 328.5, 329.5, 330.5, 331.5, 332.5, 333.5, 334.5, 335.5, 336.5, 337.5, 338.5, 339.5, 340.5, 341.5, 342.5, 343.5, 344.5, 345.5, 346.5, 347.5, 348.5, 349.5, 350.5, 351.5, 352.5, 353.5, 354.5, 355.5, 356.5, 357.5, 358.5, 359.5]
2026-06-16 17:08:19,556 - root - INFO - loglr -13
2026-06-16 17:08:19,556 - root - INFO - weight_decay 3e-06
2026-06-16 17:08:19,556 - root - INFO - num_data_workers 4
2026-06-16 17:08:19,556 - root - INFO - batch_size 1
2026-06-16 17:08:19,556 - root - INFO - max_epochs 100
2026-06-16 17:08:19,556 - root - INFO - no_leap_year 2020
2026-06-16 17:08:19,556 - root - INFO - leap_year 2020
2026-06-16 17:08:19,556 - root - INFO - log_to_screen True
2026-06-16 17:08:19,556 - root - INFO - log_to_wandb True
2026-06-16 17:08:19,557 - root - INFO - save_checkpoint True
2026-06-16 17:08:19,557 - root - INFO - save_forecasts True
2026-06-16 17:08:19,557 - root - INFO - optimizer_type AdamW
2026-06-16 17:08:19,557 - root - INFO - plot_animations False
2026-06-16 17:08:19,557 - root - INFO - group plasim
2026-06-16 17:08:19,557 - root - INFO - exp_dir /work/11095/jwan4/PanguWeather/v2.0/results
2026-06-16 17:08:19,557 - root - INFO - enable_fp8 False
2026-06-16 17:08:19,557 - root - INFO - fresh_start False
2026-06-16 17:08:19,557 - root - INFO - use_transformer_engine False
2026-06-16 17:08:19,557 - root - INFO - early_stopping False
2026-06-16 17:08:19,557 - root - INFO - entity jesswan-university-of-chicago
2026-06-16 17:08:19,557 - root - INFO - project E3SM-SRM-SFNO
2026-06-16 17:08:19,557 - root - INFO - name E3SM-SRM-SFNO-CTL_SST0051_REST0101-0016
2026-06-16 17:08:19,557 - root - INFO - use_legacy_model False
2026-06-16 17:08:19,557 - root - INFO - save_basenames ['/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001']
2026-06-16 17:08:19,557 - root - INFO - output_dirs ['/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016']
2026-06-16 17:08:19,557 - root - INFO - ensemble_inference_hours 336
2026-06-16 17:08:19,557 - root - INFO - num_ensemble_members 1
2026-06-16 17:08:19,557 - root - INFO - run_iter 1
2026-06-16 17:08:19,557 - root - INFO - has_diagnostic True
2026-06-16 17:08:19,558 - root - INFO - init_nc_filepaths ['/scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/sigma_data/2045_Combined_EAM_ELM.nc']
2026-06-16 17:08:19,558 - root - INFO - ensemble_members_per_pred 1
2026-06-16 17:08:19,558 - root - INFO - world_size 4
2026-06-16 17:08:19,558 - root - INFO - global_batch_size 4
2026-06-16 17:08:19,558 - root - INFO - experiment_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016
2026-06-16 17:08:19,558 - root - INFO - checkpoint_dir /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints
2026-06-16 17:08:19,558 - root - INFO - best_checkpoint_path /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/best_ckpt.tar
2026-06-16 17:08:19,558 - root - INFO - latest_checkpoint_path /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/ckpt_latest.tar
2026-06-16 17:08:19,558 - root - INFO - checkpoint_path_globstr /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/ckpt_epoch_*.tar
2026-06-16 17:08:19,558 - root - INFO - resuming True
2026-06-16 17:08:19,558 - root - INFO - local_rank 0
2026-06-16 17:08:19,558 - root - INFO - enable_amp True
2026-06-16 17:08:19,558 - root - INFO - ---------------------------------------------------
2026-06-16 17:08:19,592 - root - INFO - rank 0, begin data loader init
2026-06-16 17:08:20,541 - root - INFO - rank 1, begin data loader init
2026-06-16 17:08:20,542 - root - INFO - rank 2, begin data loader init
2026-06-16 17:08:20,545 - root - INFO - rank 3, begin data loader init
Loading boundary data:   0%|          | 0/56 [00:00<?, ?it/s]Loading boundary data:   4%|3         | 2/56 [00:00<00:04, 11.86it/s]Loading boundary data:   7%|7         | 4/56 [00:00<00:05,  8.70it/s]Loading boundary data:   9%|8         | 5/56 [00:00<00:07,  7.14it/s]Loading boundary data:  14%|#4        | 8/56 [00:00<00:05,  8.86it/s]Loading boundary data:  18%|#7        | 10/56 [00:01<00:04, 10.33it/s]Loading boundary data:  21%|##1       | 12/56 [00:01<00:04, 10.73it/s]Loading boundary data:  25%|##5       | 14/56 [00:01<00:03, 11.27it/s]Loading boundary data:  29%|##8       | 16/56 [00:01<00:04,  8.01it/s]Loading boundary data:  32%|###2      | 18/56 [00:02<00:04,  8.23it/s]Loading boundary data:  34%|###3      | 19/56 [00:02<00:04,  7.50it/s]Loading boundary data:  36%|###5      | 20/56 [00:02<00:05,  6.73it/s]Loading boundary data:  38%|###7      | 21/56 [00:02<00:04,  7.18it/s]Loading boundary data:  39%|###9      | 22/56 [00:02<00:05,  6.60it/s]Loading boundary data:  41%|####1     | 23/56 [00:02<00:05,  6.55it/s]Loading boundary data:  43%|####2     | 24/56 [00:03<00:05,  6.28it/s]Loading boundary data:  45%|####4     | 25/56 [00:03<00:05,  5.84it/s]Loading boundary data:  46%|####6     | 26/56 [00:03<00:05,  5.97it/s]Loading boundary data:  48%|####8     | 27/56 [00:03<00:05,  5.55it/s]Loading boundary data:  50%|#####     | 28/56 [00:03<00:05,  5.36it/s]Loading boundary data:  52%|#####1    | 29/56 [00:03<00:04,  5.69it/s]Loading boundary data:  55%|#####5    | 31/56 [00:04<00:03,  8.12it/s]Loading boundary data:  61%|######    | 34/56 [00:04<00:01, 11.24it/s]Loading boundary data:  64%|######4   | 36/56 [00:04<00:01, 12.23it/s]Loading boundary data:  70%|######9   | 39/56 [00:04<00:01, 11.67it/s]Loading boundary data:  73%|#######3  | 41/56 [00:04<00:01, 12.61it/s]Loading boundary data:  77%|#######6  | 43/56 [00:04<00:00, 13.88it/s]Loading boundary data:  80%|########  | 45/56 [00:05<00:01, 10.48it/s]Loading boundary data:  84%|########3 | 47/56 [00:05<00:00, 10.60it/s]Loading boundary data:  88%|########7 | 49/56 [00:05<00:00, 10.24it/s]Loading boundary data:  91%|#########1| 51/56 [00:05<00:00,  9.64it/s]Loading boundary data:  95%|#########4| 53/56 [00:05<00:00, 10.89it/s]Loading boundary data:  98%|#########8| 55/56 [00:06<00:00,  9.98it/s]Loading boundary data: 100%|##########| 56/56 [00:06<00:00,  8.87it/s]
2026-06-16 17:08:31,736 - root - INFO - rank 2, data loader initialized2026-06-16 17:08:31,736 - root - INFO - rank 3, data loader initialized
2026-06-16 17:08:31,736 - root - INFO - rank 0, data loader initialized

2026-06-16 17:08:31,736 - root - INFO - rank 1, data loader initialized
2026-06-16 17:08:43,730 - root - INFO - Number of trainable model parameters: 8928080762026-06-16 17:08:43,730 - root - INFO - Number of trainable model parameters: 892808076

2026-06-16 17:08:43,730 - root - INFO - Number of trainable model parameters: 8928080762026-06-16 17:08:43,730 - root - INFO - Number of trainable model parameters: 892808076

2026-06-16 17:09:12,986 - root - INFO - Using EMA state from checkpoint (preferred for inference)2026-06-16 17:09:12,986 - root - INFO - Using EMA state from checkpoint (preferred for inference)2026-06-16 17:09:12,986 - root - INFO - Using EMA state from checkpoint (preferred for inference)

2026-06-16 17:09:12,986 - root - INFO - Using EMA state from checkpoint (preferred for inference)

2026-06-16 17:09:12,988 - root - WARNING - Direct load failed: Error(s) in loading state_dict for DistributedDataParallel:
	Missing key(s) in state_dict: "module.pos_embed", "module.encoder.0.weight", "module.encoder.0.bias", "module.encoder.2.weight", "module.blocks.0.norm0.weight", "module.blocks.0.norm0.bias", "module.blocks.0.filter.filter.weight", "module.blocks.0.filter.filter.bias", "module.blocks.0.inner_skip.weight", "module.blocks.0.inner_skip.bias", "module.blocks.0.norm1.weight", "module.blocks.0.norm1.bias", "module.blocks.0.mlp.fwd.0.weight", "module.blocks.0.mlp.fwd.0.bias", "module.blocks.0.mlp.fwd.2.weight", "module.blocks.0.mlp.fwd.2.bias", "module.blocks.1.norm0.weight", "module.blocks.1.norm0.bias", "module.blocks.1.filter.filter.weight", "module.blocks.1.filter.filter.bias", "module.blocks.1.inner_skip.weight", "module.blocks.1.inner_skip.bias", "module.blocks.1.norm1.weight", "module.blocks.1.norm1.bias", "module.blocks.1.mlp.fwd.0.weight", "module.blocks.1.mlp.fwd.0.bias", "module.blocks.1.mlp.fwd.2.weight", "module.blocks.1.mlp.fwd.2.bias", "module.blocks.2.norm0.weight", "module.blocks.2.norm0.bias", "module.blocks.2.filter.filter.weight", "module.blocks.2.filter.filter.bias", "module.blocks.2.inner_skip.weight", "module.blocks.2.inner_skip.bias", "module.blocks.2.norm1.weight", "module.blocks.2.norm1.bias", "module.blocks.2.mlp.fwd.0.weight", "module.blocks.2.mlp.fwd.0.bias", "module.blocks.2.mlp.fwd.2.weight", "module.blocks.2.mlp.fwd.2.bias", "module.blocks.3.norm0.weight", "module.blocks.3.norm0.bias", "module.blocks.3.filter.filter.weight", "module.blocks.3.filter.filter.bias", "module.blocks.3.inner_skip.weight", "module.blocks.3.inner_skip.bias", "module.blocks.3.norm1.weight", "module.blocks.3.norm1.bias", "module.blocks.3.mlp.fwd.0.weight", "module.blocks.3.mlp.fwd.0.bias", "module.blocks.3.mlp.fwd.2.weight", "module.blocks.3.mlp.fwd.2.bias", "module.blocks.4.norm0.weight", "module.blocks.4.norm0.bias", "module.blocks.4.filter.filter.weight", "module.blocks.4.filter.filter.bias", "module.blocks.4.inner_skip.weight", "module.blocks.4.inner_skip.bias", "module.blocks.4.norm1.weight", "module.blocks.4.norm1.bias", "module.blocks.4.mlp.fwd.0.weight", "module.blocks.4.mlp.fwd.0.bias", "module.blocks.4.mlp.fwd.2.weight", "module.blocks.4.mlp.fwd.2.bias", "module.blocks.5.norm0.weight", "module.blocks.5.norm0.bias", "module.blocks.5.filter.filter.weight", "module.blocks.5.filter.filter.bias", "module.blocks.5.inner_skip.weight", "module.blocks.5.inner_skip.bias", "module.blocks.5.norm1.weight", "module.blocks.5.norm1.bias", "module.blocks.5.mlp.fwd.0.weight", "module.blocks.5.mlp.fwd.0.bias", "module.blocks.5.mlp.fwd.2.weight", "module.blocks.5.mlp.fwd.2.bias", "module.blocks.6.norm0.weight", "module.blocks.6.norm0.bias", "module.blocks.6.filter.filter.weight", "module.blocks.6.filter.filter.bias", "module.blocks.6.inner_skip.weight", "module.blocks.6.inner_skip.bias", "module.blocks.6.norm1.weight", "module.blocks.6.norm1.bias", "module.blocks.6.mlp.fwd.0.weight", "module.blocks.6.mlp.fwd.0.bias", "module.blocks.6.mlp.fwd.2.weight", "module.blocks.6.mlp.fwd.2.bias", "module.blocks.7.norm0.weight", "module.blocks.7.norm0.bias", "module.blocks.7.filter.filter.weight", "module.blocks.7.filter.filter.bias", "module.blocks.7.inner_skip.weight", "module.blocks.7.inner_skip.bias", "module.blocks.7.norm1.weight", "module.blocks.7.norm1.bias", "module.blocks.7.mlp.fwd.0.weight", "module.blocks.7.mlp.fwd.0.bias", "module.blocks.7.mlp.fwd.2.weight", "module.blocks.7.mlp.fwd.2.bias", "module.blocks.8.norm0.weight", "module.blocks.8.norm0.bias", "module.blocks.8.filter.filter.weight", "module.blocks.8.filter.filter.bias", "module.blocks.8.inner_skip.weight", "module.blocks.8.inner_skip.bias", "module.blocks.8.norm1.weight", "module.blocks.8.norm1.bias", "module.blocks.8.mlp.fwd.0.weight", "module.blocks.8.mlp.fwd.0.bias", "module.blocks.8.mlp.fwd.2.weight", "module.blocks.8.mlp.fwd.2.bias", "module.blocks.9.norm0.weight", "module.blocks.9.norm0.bias", "module.blocks.9.filter.filter.weight", "module.blocks.9.filter.filter.bias", "module.blocks.9.inner_skip.weight", "module.blocks.9.inner_skip.bias", "module.blocks.9.norm1.weight", "module.blocks.9.norm1.bias", "module.blocks.9.mlp.fwd.0.weight", "module.blocks.9.mlp.fwd.0.bias", "module.blocks.9.mlp.fwd.2.weight", "module.blocks.9.mlp.fwd.2.bias", "module.blocks.10.norm0.weight", "module.blocks.10.norm0.bias", "module.blocks.10.filter.filter.weight", "module.blocks.10.filter.filter.bias", "module.blocks.10.inner_skip.weight", "module.blocks.10.inner_skip.bias", "module.blocks.10.norm1.weight", "module.blocks.10.norm1.bias", "module.blocks.10.mlp.fwd.0.weight", "module.blocks.10.mlp.fwd.0.bias", "module.blocks.10.mlp.fwd.2.weight", "module.blocks.10.mlp.fwd.2.bias", "module.blocks.11.norm0.weight", "module.blocks.11.norm0.bias", "module.blocks.11.filter.filter.weight", "module.blocks.11.filter.filter.bias", "module.blocks.11.inner_skip.weight", "module.blocks.11.inner_skip.bias", "module.blocks.11.norm1.weight", "module.blocks.11.norm1.bias", "module.blocks.11.mlp.fwd.0.weight", "module.blocks.11.mlp.fwd.0.bias", "module.blocks.11.mlp.fwd.2.weight", "module.blocks.11.mlp.fwd.2.bias", "module.decoder.0.weight", "module.decoder.0.bias", "module.decoder.2.weight". 
	Unexpected key(s) in state_dict: "pos_embed", "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "blocks.0.norm0.weight", "blocks.0.norm0.bias", "blocks.0.filter.filter.weight", "blocks.0.filter.filter.bias", "blocks.0.inner_skip.weight", "blocks.0.inner_skip.bias", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.mlp.fwd.0.weight", "blocks.0.mlp.fwd.0.bias", "blocks.0.mlp.fwd.2.weight", "blocks.0.mlp.fwd.2.bias", "blocks.1.norm0.weight", "blocks.1.norm0.bias", "blocks.1.filter.filter.weight", "blocks.1.filter.filter.bias", "blocks.1.inner_skip.weight", "blocks.1.inner_skip.bias", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.mlp.fwd.0.weight", "blocks.1.mlp.fwd.0.bias", "blocks.1.mlp.fwd.2.weight", "blocks.1.mlp.fwd.2.bias", "blocks.2.norm0.weight", "blocks.2.norm0.bias", "blocks.2.filter.filter.weight", "blocks.2.filter.filter.bias", "blocks.2.inner_skip.weight", "blocks.2.inner_skip.bias", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.mlp.fwd.0.weight", "blocks.2.mlp.fwd.0.bias", "blocks.2.mlp.fwd.2.weight", "blocks.2.mlp.fwd.2.bias", "blocks.3.norm0.weight", "blocks.3.norm0.bias", "blocks.3.filter.filter.weight", "blocks.3.filter.filter.bias", "blocks.3.inner_skip.weight", "blocks.3.inner_skip.bias", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.mlp.fwd.0.weight", "blocks.3.mlp.fwd.0.bias", "blocks.3.mlp.fwd.2.weight", "blocks.3.mlp.fwd.2.bias", "blocks.4.norm0.weight", "blocks.4.norm0.bias", "blocks.4.filter.filter.weight", "blocks.4.filter.filter.bias", "blocks.4.inner_skip.weight", "blocks.4.inner_skip.bias", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.mlp.fwd.0.weight", "blocks.4.mlp.fwd.0.bias", "blocks.4.mlp.fwd.2.weight", "blocks.4.mlp.fwd.2.bias", "blocks.5.norm0.weight", "blocks.5.norm0.bias", "blocks.5.filter.filter.weight", "blocks.5.filter.filter.bias", "blocks.5.inner_skip.weight", "blocks.5.inner_skip.bias", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.mlp.fwd.0.weight", "blocks.5.mlp.fwd.0.bias", "blocks.5.mlp.fwd.2.weight", "blocks.5.mlp.fwd.2.bias", "blocks.6.norm0.weight", "blocks.6.norm0.bias", "blocks.6.filter.filter.weight", "blocks.6.filter.filter.bias", "blocks.6.inner_skip.weight", "blocks.6.inner_skip.bias", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.mlp.fwd.0.weight", "blocks.6.mlp.fwd.0.bias", "blocks.6.mlp.fwd.2.weight", "blocks.6.mlp.fwd.2.bias", "blocks.7.norm0.weight", "blocks.7.norm0.bias", "blocks.7.filter.filter.weight", "blocks.7.filter.filter.bias", "blocks.7.inner_skip.weight", "blocks.7.inner_skip.bias", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.mlp.fwd.0.weight", "blocks.7.mlp.fwd.0.bias", "blocks.7.mlp.fwd.2.weight", "blocks.7.mlp.fwd.2.bias", "blocks.8.norm0.weight", "blocks.8.norm0.bias", "blocks.8.filter.filter.weight", "blocks.8.filter.filter.bias", "blocks.8.inner_skip.weight", "blocks.8.inner_skip.bias", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.mlp.fwd.0.weight", "blocks.8.mlp.fwd.0.bias", "blocks.8.mlp.fwd.2.weight", "blocks.8.mlp.fwd.2.bias", "blocks.9.norm0.weight", "blocks.9.norm0.bias", "blocks.9.filter.filter.weight", "blocks.9.filter.filter.bias", "blocks.9.inner_skip.weight", "blocks.9.inner_skip.bias", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.mlp.fwd.0.weight", "blocks.9.mlp.fwd.0.bias", "blocks.9.mlp.fwd.2.weight", "blocks.9.mlp.fwd.2.bias", "blocks.10.norm0.weight", "blocks.10.norm0.bias", "blocks.10.filter.filter.weight", "blocks.10.filter.filter.bias", "blocks.10.inner_skip.weight", "blocks.10.inner_skip.bias", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.mlp.fwd.0.weight", "blocks.10.mlp.fwd.0.bias", "blocks.10.mlp.fwd.2.weight", "blocks.10.mlp.fwd.2.bias", "blocks.11.norm0.weight", "blocks.11.norm0.bias", "blocks.11.filter.filter.weight", "blocks.11.filter.filter.bias", "blocks.11.inner_skip.weight", "blocks.11.inner_skip.bias", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.mlp.fwd.0.weight", "blocks.11.mlp.fwd.0.bias", "blocks.11.mlp.fwd.2.weight", "blocks.11.mlp.fwd.2.bias", "decoder.0.weight", "decoder.0.bias", "decoder.2.weight". . Attempting to fix "module." prefix mismatch...
2026-06-16 17:09:12,989 - root - INFO - Added "module." prefix to checkpoint keys
2026-06-16 17:09:12,989 - root - WARNING - Direct load failed: Error(s) in loading state_dict for DistributedDataParallel:
	Missing key(s) in state_dict: "module.pos_embed", "module.encoder.0.weight", "module.encoder.0.bias", "module.encoder.2.weight", "module.blocks.0.norm0.weight", "module.blocks.0.norm0.bias", "module.blocks.0.filter.filter.weight", "module.blocks.0.filter.filter.bias", "module.blocks.0.inner_skip.weight", "module.blocks.0.inner_skip.bias", "module.blocks.0.norm1.weight", "module.blocks.0.norm1.bias", "module.blocks.0.mlp.fwd.0.weight", "module.blocks.0.mlp.fwd.0.bias", "module.blocks.0.mlp.fwd.2.weight", "module.blocks.0.mlp.fwd.2.bias", "module.blocks.1.norm0.weight", "module.blocks.1.norm0.bias", "module.blocks.1.filter.filter.weight", "module.blocks.1.filter.filter.bias", "module.blocks.1.inner_skip.weight", "module.blocks.1.inner_skip.bias", "module.blocks.1.norm1.weight", "module.blocks.1.norm1.bias", "module.blocks.1.mlp.fwd.0.weight", "module.blocks.1.mlp.fwd.0.bias", "module.blocks.1.mlp.fwd.2.weight", "module.blocks.1.mlp.fwd.2.bias", "module.blocks.2.norm0.weight", "module.blocks.2.norm0.bias", "module.blocks.2.filter.filter.weight", "module.blocks.2.filter.filter.bias", "module.blocks.2.inner_skip.weight", "module.blocks.2.inner_skip.bias", "module.blocks.2.norm1.weight", "module.blocks.2.norm1.bias", "module.blocks.2.mlp.fwd.0.weight", "module.blocks.2.mlp.fwd.0.bias", "module.blocks.2.mlp.fwd.2.weight", "module.blocks.2.mlp.fwd.2.bias", "module.blocks.3.norm0.weight", "module.blocks.3.norm0.bias", "module.blocks.3.filter.filter.weight", "module.blocks.3.filter.filter.bias", "module.blocks.3.inner_skip.weight", "module.blocks.3.inner_skip.bias", "module.blocks.3.norm1.weight", "module.blocks.3.norm1.bias", "module.blocks.3.mlp.fwd.0.weight", "module.blocks.3.mlp.fwd.0.bias", "module.blocks.3.mlp.fwd.2.weight", "module.blocks.3.mlp.fwd.2.bias", "module.blocks.4.norm0.weight", "module.blocks.4.norm0.bias", "module.blocks.4.filter.filter.weight", "module.blocks.4.filter.filter.bias", "module.blocks.4.inner_skip.weight", "module.blocks.4.inner_skip.bias", "module.blocks.4.norm1.weight", "module.blocks.4.norm1.bias", "module.blocks.4.mlp.fwd.0.weight", "module.blocks.4.mlp.fwd.0.bias", "module.blocks.4.mlp.fwd.2.weight", "module.blocks.4.mlp.fwd.2.bias", "module.blocks.5.norm0.weight", "module.blocks.5.norm0.bias", "module.blocks.5.filter.filter.weight", "module.blocks.5.filter.filter.bias", "module.blocks.5.inner_skip.weight", "module.blocks.5.inner_skip.bias", "module.blocks.5.norm1.weight", "module.blocks.5.norm1.bias", "module.blocks.5.mlp.fwd.0.weight", "module.blocks.5.mlp.fwd.0.bias", "module.blocks.5.mlp.fwd.2.weight", "module.blocks.5.mlp.fwd.2.bias", "module.blocks.6.norm0.weight", "module.blocks.6.norm0.bias", "module.blocks.6.filter.filter.weight", "module.blocks.6.filter.filter.bias", "module.blocks.6.inner_skip.weight", "module.blocks.6.inner_skip.bias", "module.blocks.6.norm1.weight", "module.blocks.6.norm1.bias", "module.blocks.6.mlp.fwd.0.weight", "module.blocks.6.mlp.fwd.0.bias", "module.blocks.6.mlp.fwd.2.weight", "module.blocks.6.mlp.fwd.2.bias", "module.blocks.7.norm0.weight", "module.blocks.7.norm0.bias", "module.blocks.7.filter.filter.weight", "module.blocks.7.filter.filter.bias", "module.blocks.7.inner_skip.weight", "module.blocks.7.inner_skip.bias", "module.blocks.7.norm1.weight", "module.blocks.7.norm1.bias", "module.blocks.7.mlp.fwd.0.weight", "module.blocks.7.mlp.fwd.0.bias", "module.blocks.7.mlp.fwd.2.weight", "module.blocks.7.mlp.fwd.2.bias", "module.blocks.8.norm0.weight", "module.blocks.8.norm0.bias", "module.blocks.8.filter.filter.weight", "module.blocks.8.filter.filter.bias", "module.blocks.8.inner_skip.weight", "module.blocks.8.inner_skip.bias", "module.blocks.8.norm1.weight", "module.blocks.8.norm1.bias", "module.blocks.8.mlp.fwd.0.weight", "module.blocks.8.mlp.fwd.0.bias", "module.blocks.8.mlp.fwd.2.weight", "module.blocks.8.mlp.fwd.2.bias", "module.blocks.9.norm0.weight", "module.blocks.9.norm0.bias", "module.blocks.9.filter.filter.weight", "module.blocks.9.filter.filter.bias", "module.blocks.9.inner_skip.weight", "module.blocks.9.inner_skip.bias", "module.blocks.9.norm1.weight", "module.blocks.9.norm1.bias", "module.blocks.9.mlp.fwd.0.weight", "module.blocks.9.mlp.fwd.0.bias", "module.blocks.9.mlp.fwd.2.weight", "module.blocks.9.mlp.fwd.2.bias", "module.blocks.10.norm0.weight", "module.blocks.10.norm0.bias", "module.blocks.10.filter.filter.weight", "module.blocks.10.filter.filter.bias", "module.blocks.10.inner_skip.weight", "module.blocks.10.inner_skip.bias", "module.blocks.10.norm1.weight", "module.blocks.10.norm1.bias", "module.blocks.10.mlp.fwd.0.weight", "module.blocks.10.mlp.fwd.0.bias", "module.blocks.10.mlp.fwd.2.weight", "module.blocks.10.mlp.fwd.2.bias", "module.blocks.11.norm0.weight", "module.blocks.11.norm0.bias", "module.blocks.11.filter.filter.weight", "module.blocks.11.filter.filter.bias", "module.blocks.11.inner_skip.weight", "module.blocks.11.inner_skip.bias", "module.blocks.11.norm1.weight", "module.blocks.11.norm1.bias", "module.blocks.11.mlp.fwd.0.weight", "module.blocks.11.mlp.fwd.0.bias", "module.blocks.11.mlp.fwd.2.weight", "module.blocks.11.mlp.fwd.2.bias", "module.decoder.0.weight", "module.decoder.0.bias", "module.decoder.2.weight". 
	Unexpected key(s) in state_dict: "pos_embed", "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "blocks.0.norm0.weight", "blocks.0.norm0.bias", "blocks.0.filter.filter.weight", "blocks.0.filter.filter.bias", "blocks.0.inner_skip.weight", "blocks.0.inner_skip.bias", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.mlp.fwd.0.weight", "blocks.0.mlp.fwd.0.bias", "blocks.0.mlp.fwd.2.weight", "blocks.0.mlp.fwd.2.bias", "blocks.1.norm0.weight", "blocks.1.norm0.bias", "blocks.1.filter.filter.weight", "blocks.1.filter.filter.bias", "blocks.1.inner_skip.weight", "blocks.1.inner_skip.bias", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.mlp.fwd.0.weight", "blocks.1.mlp.fwd.0.bias", "blocks.1.mlp.fwd.2.weight", "blocks.1.mlp.fwd.2.bias", "blocks.2.norm0.weight", "blocks.2.norm0.bias", "blocks.2.filter.filter.weight", "blocks.2.filter.filter.bias", "blocks.2.inner_skip.weight", "blocks.2.inner_skip.bias", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.mlp.fwd.0.weight", "blocks.2.mlp.fwd.0.bias", "blocks.2.mlp.fwd.2.weight", "blocks.2.mlp.fwd.2.bias", "blocks.3.norm0.weight", "blocks.3.norm0.bias", "blocks.3.filter.filter.weight", "blocks.3.filter.filter.bias", "blocks.3.inner_skip.weight", "blocks.3.inner_skip.bias", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.mlp.fwd.0.weight", "blocks.3.mlp.fwd.0.bias", "blocks.3.mlp.fwd.2.weight", "blocks.3.mlp.fwd.2.bias", "blocks.4.norm0.weight", "blocks.4.norm0.bias", "blocks.4.filter.filter.weight", "blocks.4.filter.filter.bias", "blocks.4.inner_skip.weight", "blocks.4.inner_skip.bias", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.mlp.fwd.0.weight", "blocks.4.mlp.fwd.0.bias", "blocks.4.mlp.fwd.2.weight", "blocks.4.mlp.fwd.2.bias", "blocks.5.norm0.weight", "blocks.5.norm0.bias", "blocks.5.filter.filter.weight", "blocks.5.filter.filter.bias", "blocks.5.inner_skip.weight", "blocks.5.inner_skip.bias", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.mlp.fwd.0.weight", "blocks.5.mlp.fwd.0.bias", "blocks.5.mlp.fwd.2.weight", "blocks.5.mlp.fwd.2.bias", "blocks.6.norm0.weight", "blocks.6.norm0.bias", "blocks.6.filter.filter.weight", "blocks.6.filter.filter.bias", "blocks.6.inner_skip.weight", "blocks.6.inner_skip.bias", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.mlp.fwd.0.weight", "blocks.6.mlp.fwd.0.bias", "blocks.6.mlp.fwd.2.weight", "blocks.6.mlp.fwd.2.bias", "blocks.7.norm0.weight", "blocks.7.norm0.bias", "blocks.7.filter.filter.weight", "blocks.7.filter.filter.bias", "blocks.7.inner_skip.weight", "blocks.7.inner_skip.bias", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.mlp.fwd.0.weight", "blocks.7.mlp.fwd.0.bias", "blocks.7.mlp.fwd.2.weight", "blocks.7.mlp.fwd.2.bias", "blocks.8.norm0.weight", "blocks.8.norm0.bias", "blocks.8.filter.filter.weight", "blocks.8.filter.filter.bias", "blocks.8.inner_skip.weight", "blocks.8.inner_skip.bias", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.mlp.fwd.0.weight", "blocks.8.mlp.fwd.0.bias", "blocks.8.mlp.fwd.2.weight", "blocks.8.mlp.fwd.2.bias", "blocks.9.norm0.weight", "blocks.9.norm0.bias", "blocks.9.filter.filter.weight", "blocks.9.filter.filter.bias", "blocks.9.inner_skip.weight", "blocks.9.inner_skip.bias", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.mlp.fwd.0.weight", "blocks.9.mlp.fwd.0.bias", "blocks.9.mlp.fwd.2.weight", "blocks.9.mlp.fwd.2.bias", "blocks.10.norm0.weight", "blocks.10.norm0.bias", "blocks.10.filter.filter.weight", "blocks.10.filter.filter.bias", "blocks.10.inner_skip.weight", "blocks.10.inner_skip.bias", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.mlp.fwd.0.weight", "blocks.10.mlp.fwd.0.bias", "blocks.10.mlp.fwd.2.weight", "blocks.10.mlp.fwd.2.bias", "blocks.11.norm0.weight", "blocks.11.norm0.bias", "blocks.11.filter.filter.weight", "blocks.11.filter.filter.bias", "blocks.11.inner_skip.weight", "blocks.11.inner_skip.bias", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.mlp.fwd.0.weight", "blocks.11.mlp.fwd.0.bias", "blocks.11.mlp.fwd.2.weight", "blocks.11.mlp.fwd.2.bias", "decoder.0.weight", "decoder.0.bias", "decoder.2.weight". . Attempting to fix "module." prefix mismatch...
2026-06-16 17:09:12,989 - root - INFO - Added "module." prefix to checkpoint keys
2026-06-16 17:09:12,989 - root - WARNING - Direct load failed: Error(s) in loading state_dict for DistributedDataParallel:
	Missing key(s) in state_dict: "module.pos_embed", "module.encoder.0.weight", "module.encoder.0.bias", "module.encoder.2.weight", "module.blocks.0.norm0.weight", "module.blocks.0.norm0.bias", "module.blocks.0.filter.filter.weight", "module.blocks.0.filter.filter.bias", "module.blocks.0.inner_skip.weight", "module.blocks.0.inner_skip.bias", "module.blocks.0.norm1.weight", "module.blocks.0.norm1.bias", "module.blocks.0.mlp.fwd.0.weight", "module.blocks.0.mlp.fwd.0.bias", "module.blocks.0.mlp.fwd.2.weight", "module.blocks.0.mlp.fwd.2.bias", "module.blocks.1.norm0.weight", "module.blocks.1.norm0.bias", "module.blocks.1.filter.filter.weight", "module.blocks.1.filter.filter.bias", "module.blocks.1.inner_skip.weight", "module.blocks.1.inner_skip.bias", "module.blocks.1.norm1.weight", "module.blocks.1.norm1.bias", "module.blocks.1.mlp.fwd.0.weight", "module.blocks.1.mlp.fwd.0.bias", "module.blocks.1.mlp.fwd.2.weight", "module.blocks.1.mlp.fwd.2.bias", "module.blocks.2.norm0.weight", "module.blocks.2.norm0.bias", "module.blocks.2.filter.filter.weight", "module.blocks.2.filter.filter.bias", "module.blocks.2.inner_skip.weight", "module.blocks.2.inner_skip.bias", "module.blocks.2.norm1.weight", "module.blocks.2.norm1.bias", "module.blocks.2.mlp.fwd.0.weight", "module.blocks.2.mlp.fwd.0.bias", "module.blocks.2.mlp.fwd.2.weight", "module.blocks.2.mlp.fwd.2.bias", "module.blocks.3.norm0.weight", "module.blocks.3.norm0.bias", "module.blocks.3.filter.filter.weight", "module.blocks.3.filter.filter.bias", "module.blocks.3.inner_skip.weight", "module.blocks.3.inner_skip.bias", "module.blocks.3.norm1.weight", "module.blocks.3.norm1.bias", "module.blocks.3.mlp.fwd.0.weight", "module.blocks.3.mlp.fwd.0.bias", "module.blocks.3.mlp.fwd.2.weight", "module.blocks.3.mlp.fwd.2.bias", "module.blocks.4.norm0.weight", "module.blocks.4.norm0.bias", "module.blocks.4.filter.filter.weight", "module.blocks.4.filter.filter.bias", "module.blocks.4.inner_skip.weight", "module.blocks.4.inner_skip.bias", "module.blocks.4.norm1.weight", "module.blocks.4.norm1.bias", "module.blocks.4.mlp.fwd.0.weight", "module.blocks.4.mlp.fwd.0.bias", "module.blocks.4.mlp.fwd.2.weight", "module.blocks.4.mlp.fwd.2.bias", "module.blocks.5.norm0.weight", "module.blocks.5.norm0.bias", "module.blocks.5.filter.filter.weight", "module.blocks.5.filter.filter.bias", "module.blocks.5.inner_skip.weight", "module.blocks.5.inner_skip.bias", "module.blocks.5.norm1.weight", "module.blocks.5.norm1.bias", "module.blocks.5.mlp.fwd.0.weight", "module.blocks.5.mlp.fwd.0.bias", "module.blocks.5.mlp.fwd.2.weight", "module.blocks.5.mlp.fwd.2.bias", "module.blocks.6.norm0.weight", "module.blocks.6.norm0.bias", "module.blocks.6.filter.filter.weight", "module.blocks.6.filter.filter.bias", "module.blocks.6.inner_skip.weight", "module.blocks.6.inner_skip.bias", "module.blocks.6.norm1.weight", "module.blocks.6.norm1.bias", "module.blocks.6.mlp.fwd.0.weight", "module.blocks.6.mlp.fwd.0.bias", "module.blocks.6.mlp.fwd.2.weight", "module.blocks.6.mlp.fwd.2.bias", "module.blocks.7.norm0.weight", "module.blocks.7.norm0.bias", "module.blocks.7.filter.filter.weight", "module.blocks.7.filter.filter.bias", "module.blocks.7.inner_skip.weight", "module.blocks.7.inner_skip.bias", "module.blocks.7.norm1.weight", "module.blocks.7.norm1.bias", "module.blocks.7.mlp.fwd.0.weight", "module.blocks.7.mlp.fwd.0.bias", "module.blocks.7.mlp.fwd.2.weight", "module.blocks.7.mlp.fwd.2.bias", "module.blocks.8.norm0.weight", "module.blocks.8.norm0.bias", "module.blocks.8.filter.filter.weight", "module.blocks.8.filter.filter.bias", "module.blocks.8.inner_skip.weight", "module.blocks.8.inner_skip.bias", "module.blocks.8.norm1.weight", "module.blocks.8.norm1.bias", "module.blocks.8.mlp.fwd.0.weight", "module.blocks.8.mlp.fwd.0.bias", "module.blocks.8.mlp.fwd.2.weight", "module.blocks.8.mlp.fwd.2.bias", "module.blocks.9.norm0.weight", "module.blocks.9.norm0.bias", "module.blocks.9.filter.filter.weight", "module.blocks.9.filter.filter.bias", "module.blocks.9.inner_skip.weight", "module.blocks.9.inner_skip.bias", "module.blocks.9.norm1.weight", "module.blocks.9.norm1.bias", "module.blocks.9.mlp.fwd.0.weight", "module.blocks.9.mlp.fwd.0.bias", "module.blocks.9.mlp.fwd.2.weight", "module.blocks.9.mlp.fwd.2.bias", "module.blocks.10.norm0.weight", "module.blocks.10.norm0.bias", "module.blocks.10.filter.filter.weight", "module.blocks.10.filter.filter.bias", "module.blocks.10.inner_skip.weight", "module.blocks.10.inner_skip.bias", "module.blocks.10.norm1.weight", "module.blocks.10.norm1.bias", "module.blocks.10.mlp.fwd.0.weight", "module.blocks.10.mlp.fwd.0.bias", "module.blocks.10.mlp.fwd.2.weight", "module.blocks.10.mlp.fwd.2.bias", "module.blocks.11.norm0.weight", "module.blocks.11.norm0.bias", "module.blocks.11.filter.filter.weight", "module.blocks.11.filter.filter.bias", "module.blocks.11.inner_skip.weight", "module.blocks.11.inner_skip.bias", "module.blocks.11.norm1.weight", "module.blocks.11.norm1.bias", "module.blocks.11.mlp.fwd.0.weight", "module.blocks.11.mlp.fwd.0.bias", "module.blocks.11.mlp.fwd.2.weight", "module.blocks.11.mlp.fwd.2.bias", "module.decoder.0.weight", "module.decoder.0.bias", "module.decoder.2.weight". 
	Unexpected key(s) in state_dict: "pos_embed", "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "blocks.0.norm0.weight", "blocks.0.norm0.bias", "blocks.0.filter.filter.weight", "blocks.0.filter.filter.bias", "blocks.0.inner_skip.weight", "blocks.0.inner_skip.bias", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.mlp.fwd.0.weight", "blocks.0.mlp.fwd.0.bias", "blocks.0.mlp.fwd.2.weight", "blocks.0.mlp.fwd.2.bias", "blocks.1.norm0.weight", "blocks.1.norm0.bias", "blocks.1.filter.filter.weight", "blocks.1.filter.filter.bias", "blocks.1.inner_skip.weight", "blocks.1.inner_skip.bias", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.mlp.fwd.0.weight", "blocks.1.mlp.fwd.0.bias", "blocks.1.mlp.fwd.2.weight", "blocks.1.mlp.fwd.2.bias", "blocks.2.norm0.weight", "blocks.2.norm0.bias", "blocks.2.filter.filter.weight", "blocks.2.filter.filter.bias", "blocks.2.inner_skip.weight", "blocks.2.inner_skip.bias", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.mlp.fwd.0.weight", "blocks.2.mlp.fwd.0.bias", "blocks.2.mlp.fwd.2.weight", "blocks.2.mlp.fwd.2.bias", "blocks.3.norm0.weight", "blocks.3.norm0.bias", "blocks.3.filter.filter.weight", "blocks.3.filter.filter.bias", "blocks.3.inner_skip.weight", "blocks.3.inner_skip.bias", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.mlp.fwd.0.weight", "blocks.3.mlp.fwd.0.bias", "blocks.3.mlp.fwd.2.weight", "blocks.3.mlp.fwd.2.bias", "blocks.4.norm0.weight", "blocks.4.norm0.bias", "blocks.4.filter.filter.weight", "blocks.4.filter.filter.bias", "blocks.4.inner_skip.weight", "blocks.4.inner_skip.bias", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.mlp.fwd.0.weight", "blocks.4.mlp.fwd.0.bias", "blocks.4.mlp.fwd.2.weight", "blocks.4.mlp.fwd.2.bias", "blocks.5.norm0.weight", "blocks.5.norm0.bias", "blocks.5.filter.filter.weight", "blocks.5.filter.filter.bias", "blocks.5.inner_skip.weight", "blocks.5.inner_skip.bias", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.mlp.fwd.0.weight", "blocks.5.mlp.fwd.0.bias", "blocks.5.mlp.fwd.2.weight", "blocks.5.mlp.fwd.2.bias", "blocks.6.norm0.weight", "blocks.6.norm0.bias", "blocks.6.filter.filter.weight", "blocks.6.filter.filter.bias", "blocks.6.inner_skip.weight", "blocks.6.inner_skip.bias", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.mlp.fwd.0.weight", "blocks.6.mlp.fwd.0.bias", "blocks.6.mlp.fwd.2.weight", "blocks.6.mlp.fwd.2.bias", "blocks.7.norm0.weight", "blocks.7.norm0.bias", "blocks.7.filter.filter.weight", "blocks.7.filter.filter.bias", "blocks.7.inner_skip.weight", "blocks.7.inner_skip.bias", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.mlp.fwd.0.weight", "blocks.7.mlp.fwd.0.bias", "blocks.7.mlp.fwd.2.weight", "blocks.7.mlp.fwd.2.bias", "blocks.8.norm0.weight", "blocks.8.norm0.bias", "blocks.8.filter.filter.weight", "blocks.8.filter.filter.bias", "blocks.8.inner_skip.weight", "blocks.8.inner_skip.bias", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.mlp.fwd.0.weight", "blocks.8.mlp.fwd.0.bias", "blocks.8.mlp.fwd.2.weight", "blocks.8.mlp.fwd.2.bias", "blocks.9.norm0.weight", "blocks.9.norm0.bias", "blocks.9.filter.filter.weight", "blocks.9.filter.filter.bias", "blocks.9.inner_skip.weight", "blocks.9.inner_skip.bias", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.mlp.fwd.0.weight", "blocks.9.mlp.fwd.0.bias", "blocks.9.mlp.fwd.2.weight", "blocks.9.mlp.fwd.2.bias", "blocks.10.norm0.weight", "blocks.10.norm0.bias", "blocks.10.filter.filter.weight", "blocks.10.filter.filter.bias", "blocks.10.inner_skip.weight", "blocks.10.inner_skip.bias", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.mlp.fwd.0.weight", "blocks.10.mlp.fwd.0.bias", "blocks.10.mlp.fwd.2.weight", "blocks.10.mlp.fwd.2.bias", "blocks.11.norm0.weight", "blocks.11.norm0.bias", "blocks.11.filter.filter.weight", "blocks.11.filter.filter.bias", "blocks.11.inner_skip.weight", "blocks.11.inner_skip.bias", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.mlp.fwd.0.weight", "blocks.11.mlp.fwd.0.bias", "blocks.11.mlp.fwd.2.weight", "blocks.11.mlp.fwd.2.bias", "decoder.0.weight", "decoder.0.bias", "decoder.2.weight". . Attempting to fix "module." prefix mismatch...
2026-06-16 17:09:12,989 - root - WARNING - Direct load failed: Error(s) in loading state_dict for DistributedDataParallel:
	Missing key(s) in state_dict: "module.pos_embed", "module.encoder.0.weight", "module.encoder.0.bias", "module.encoder.2.weight", "module.blocks.0.norm0.weight", "module.blocks.0.norm0.bias", "module.blocks.0.filter.filter.weight", "module.blocks.0.filter.filter.bias", "module.blocks.0.inner_skip.weight", "module.blocks.0.inner_skip.bias", "module.blocks.0.norm1.weight", "module.blocks.0.norm1.bias", "module.blocks.0.mlp.fwd.0.weight", "module.blocks.0.mlp.fwd.0.bias", "module.blocks.0.mlp.fwd.2.weight", "module.blocks.0.mlp.fwd.2.bias", "module.blocks.1.norm0.weight", "module.blocks.1.norm0.bias", "module.blocks.1.filter.filter.weight", "module.blocks.1.filter.filter.bias", "module.blocks.1.inner_skip.weight", "module.blocks.1.inner_skip.bias", "module.blocks.1.norm1.weight", "module.blocks.1.norm1.bias", "module.blocks.1.mlp.fwd.0.weight", "module.blocks.1.mlp.fwd.0.bias", "module.blocks.1.mlp.fwd.2.weight", "module.blocks.1.mlp.fwd.2.bias", "module.blocks.2.norm0.weight", "module.blocks.2.norm0.bias", "module.blocks.2.filter.filter.weight", "module.blocks.2.filter.filter.bias", "module.blocks.2.inner_skip.weight", "module.blocks.2.inner_skip.bias", "module.blocks.2.norm1.weight", "module.blocks.2.norm1.bias", "module.blocks.2.mlp.fwd.0.weight", "module.blocks.2.mlp.fwd.0.bias", "module.blocks.2.mlp.fwd.2.weight", "module.blocks.2.mlp.fwd.2.bias", "module.blocks.3.norm0.weight", "module.blocks.3.norm0.bias", "module.blocks.3.filter.filter.weight", "module.blocks.3.filter.filter.bias", "module.blocks.3.inner_skip.weight", "module.blocks.3.inner_skip.bias", "module.blocks.3.norm1.weight", "module.blocks.3.norm1.bias", "module.blocks.3.mlp.fwd.0.weight", "module.blocks.3.mlp.fwd.0.bias", "module.blocks.3.mlp.fwd.2.weight", "module.blocks.3.mlp.fwd.2.bias", "module.blocks.4.norm0.weight", "module.blocks.4.norm0.bias", "module.blocks.4.filter.filter.weight", "module.blocks.4.filter.filter.bias", "module.blocks.4.inner_skip.weight", "module.blocks.4.inner_skip.bias", "module.blocks.4.norm1.weight", "module.blocks.4.norm1.bias", "module.blocks.4.mlp.fwd.0.weight", "module.blocks.4.mlp.fwd.0.bias", "module.blocks.4.mlp.fwd.2.weight", "module.blocks.4.mlp.fwd.2.bias", "module.blocks.5.norm0.weight", "module.blocks.5.norm0.bias", "module.blocks.5.filter.filter.weight", "module.blocks.5.filter.filter.bias", "module.blocks.5.inner_skip.weight", "module.blocks.5.inner_skip.bias", "module.blocks.5.norm1.weight", "module.blocks.5.norm1.bias", "module.blocks.5.mlp.fwd.0.weight", "module.blocks.5.mlp.fwd.0.bias", "module.blocks.5.mlp.fwd.2.weight", "module.blocks.5.mlp.fwd.2.bias", "module.blocks.6.norm0.weight", "module.blocks.6.norm0.bias", "module.blocks.6.filter.filter.weight", "module.blocks.6.filter.filter.bias", "module.blocks.6.inner_skip.weight", "module.blocks.6.inner_skip.bias", "module.blocks.6.norm1.weight", "module.blocks.6.norm1.bias", "module.blocks.6.mlp.fwd.0.weight", "module.blocks.6.mlp.fwd.0.bias", "module.blocks.6.mlp.fwd.2.weight", "module.blocks.6.mlp.fwd.2.bias", "module.blocks.7.norm0.weight", "module.blocks.7.norm0.bias", "module.blocks.7.filter.filter.weight", "module.blocks.7.filter.filter.bias", "module.blocks.7.inner_skip.weight", "module.blocks.7.inner_skip.bias", "module.blocks.7.norm1.weight", "module.blocks.7.norm1.bias", "module.blocks.7.mlp.fwd.0.weight", "module.blocks.7.mlp.fwd.0.bias", "module.blocks.7.mlp.fwd.2.weight", "module.blocks.7.mlp.fwd.2.bias", "module.blocks.8.norm0.weight", "module.blocks.8.norm0.bias", "module.blocks.8.filter.filter.weight", "module.blocks.8.filter.filter.bias", "module.blocks.8.inner_skip.weight", "module.blocks.8.inner_skip.bias", "module.blocks.8.norm1.weight", "module.blocks.8.norm1.bias", "module.blocks.8.mlp.fwd.0.weight", "module.blocks.8.mlp.fwd.0.bias", "module.blocks.8.mlp.fwd.2.weight", "module.blocks.8.mlp.fwd.2.bias", "module.blocks.9.norm0.weight", "module.blocks.9.norm0.bias", "module.blocks.9.filter.filter.weight", "module.blocks.9.filter.filter.bias", "module.blocks.9.inner_skip.weight", "module.blocks.9.inner_skip.bias", "module.blocks.9.norm1.weight", "module.blocks.9.norm1.bias", "module.blocks.9.mlp.fwd.0.weight", "module.blocks.9.mlp.fwd.0.bias", "module.blocks.9.mlp.fwd.2.weight", "module.blocks.9.mlp.fwd.2.bias", "module.blocks.10.norm0.weight", "module.blocks.10.norm0.bias", "module.blocks.10.filter.filter.weight", "module.blocks.10.filter.filter.bias", "module.blocks.10.inner_skip.weight", "module.blocks.10.inner_skip.bias", "module.blocks.10.norm1.weight", "module.blocks.10.norm1.bias", "module.blocks.10.mlp.fwd.0.weight", "module.blocks.10.mlp.fwd.0.bias", "module.blocks.10.mlp.fwd.2.weight", "module.blocks.10.mlp.fwd.2.bias", "module.blocks.11.norm0.weight", "module.blocks.11.norm0.bias", "module.blocks.11.filter.filter.weight", "module.blocks.11.filter.filter.bias", "module.blocks.11.inner_skip.weight", "module.blocks.11.inner_skip.bias", "module.blocks.11.norm1.weight", "module.blocks.11.norm1.bias", "module.blocks.11.mlp.fwd.0.weight", "module.blocks.11.mlp.fwd.0.bias", "module.blocks.11.mlp.fwd.2.weight", "module.blocks.11.mlp.fwd.2.bias", "module.decoder.0.weight", "module.decoder.0.bias", "module.decoder.2.weight". 
	Unexpected key(s) in state_dict: "pos_embed", "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "blocks.0.norm0.weight", "blocks.0.norm0.bias", "blocks.0.filter.filter.weight", "blocks.0.filter.filter.bias", "blocks.0.inner_skip.weight", "blocks.0.inner_skip.bias", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.mlp.fwd.0.weight", "blocks.0.mlp.fwd.0.bias", "blocks.0.mlp.fwd.2.weight", "blocks.0.mlp.fwd.2.bias", "blocks.1.norm0.weight", "blocks.1.norm0.bias", "blocks.1.filter.filter.weight", "blocks.1.filter.filter.bias", "blocks.1.inner_skip.weight", "blocks.1.inner_skip.bias", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.mlp.fwd.0.weight", "blocks.1.mlp.fwd.0.bias", "blocks.1.mlp.fwd.2.weight", "blocks.1.mlp.fwd.2.bias", "blocks.2.norm0.weight", "blocks.2.norm0.bias", "blocks.2.filter.filter.weight", "blocks.2.filter.filter.bias", "blocks.2.inner_skip.weight", "blocks.2.inner_skip.bias", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.mlp.fwd.0.weight", "blocks.2.mlp.fwd.0.bias", "blocks.2.mlp.fwd.2.weight", "blocks.2.mlp.fwd.2.bias", "blocks.3.norm0.weight", "blocks.3.norm0.bias", "blocks.3.filter.filter.weight", "blocks.3.filter.filter.bias", "blocks.3.inner_skip.weight", "blocks.3.inner_skip.bias", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.mlp.fwd.0.weight", "blocks.3.mlp.fwd.0.bias", "blocks.3.mlp.fwd.2.weight", "blocks.3.mlp.fwd.2.bias", "blocks.4.norm0.weight", "blocks.4.norm0.bias", "blocks.4.filter.filter.weight", "blocks.4.filter.filter.bias", "blocks.4.inner_skip.weight", "blocks.4.inner_skip.bias", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.mlp.fwd.0.weight", "blocks.4.mlp.fwd.0.bias", "blocks.4.mlp.fwd.2.weight", "blocks.4.mlp.fwd.2.bias", "blocks.5.norm0.weight", "blocks.5.norm0.bias", "blocks.5.filter.filter.weight", "blocks.5.filter.filter.bias", "blocks.5.inner_skip.weight", "blocks.5.inner_skip.bias", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.mlp.fwd.0.weight", "blocks.5.mlp.fwd.0.bias", "blocks.5.mlp.fwd.2.weight", "blocks.5.mlp.fwd.2.bias", "blocks.6.norm0.weight", "blocks.6.norm0.bias", "blocks.6.filter.filter.weight", "blocks.6.filter.filter.bias", "blocks.6.inner_skip.weight", "blocks.6.inner_skip.bias", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.mlp.fwd.0.weight", "blocks.6.mlp.fwd.0.bias", "blocks.6.mlp.fwd.2.weight", "blocks.6.mlp.fwd.2.bias", "blocks.7.norm0.weight", "blocks.7.norm0.bias", "blocks.7.filter.filter.weight", "blocks.7.filter.filter.bias", "blocks.7.inner_skip.weight", "blocks.7.inner_skip.bias", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.mlp.fwd.0.weight", "blocks.7.mlp.fwd.0.bias", "blocks.7.mlp.fwd.2.weight", "blocks.7.mlp.fwd.2.bias", "blocks.8.norm0.weight", "blocks.8.norm0.bias", "blocks.8.filter.filter.weight", "blocks.8.filter.filter.bias", "blocks.8.inner_skip.weight", "blocks.8.inner_skip.bias", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.mlp.fwd.0.weight", "blocks.8.mlp.fwd.0.bias", "blocks.8.mlp.fwd.2.weight", "blocks.8.mlp.fwd.2.bias", "blocks.9.norm0.weight", "blocks.9.norm0.bias", "blocks.9.filter.filter.weight", "blocks.9.filter.filter.bias", "blocks.9.inner_skip.weight", "blocks.9.inner_skip.bias", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.mlp.fwd.0.weight", "blocks.9.mlp.fwd.0.bias", "blocks.9.mlp.fwd.2.weight", "blocks.9.mlp.fwd.2.bias", "blocks.10.norm0.weight", "blocks.10.norm0.bias", "blocks.10.filter.filter.weight", "blocks.10.filter.filter.bias", "blocks.10.inner_skip.weight", "blocks.10.inner_skip.bias", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.mlp.fwd.0.weight", "blocks.10.mlp.fwd.0.bias", "blocks.10.mlp.fwd.2.weight", "blocks.10.mlp.fwd.2.bias", "blocks.11.norm0.weight", "blocks.11.norm0.bias", "blocks.11.filter.filter.weight", "blocks.11.filter.filter.bias", "blocks.11.inner_skip.weight", "blocks.11.inner_skip.bias", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.mlp.fwd.0.weight", "blocks.11.mlp.fwd.0.bias", "blocks.11.mlp.fwd.2.weight", "blocks.11.mlp.fwd.2.bias", "decoder.0.weight", "decoder.0.bias", "decoder.2.weight". . Attempting to fix "module." prefix mismatch...
2026-06-16 17:09:12,989 - root - INFO - Added "module." prefix to checkpoint keys
2026-06-16 17:09:12,989 - root - INFO - Added "module." prefix to checkpoint keys
2026-06-16 17:09:13,008 - root - INFO - Successfully loaded checkpoint after fixing "module." prefix2026-06-16 17:09:13,008 - root - INFO - Successfully loaded checkpoint after fixing "module." prefix
2026-06-16 17:09:13,008 - root - INFO - Successfully loaded checkpoint after fixing "module." prefix
2026-06-16 17:09:13,008 - root - INFO - Successfully loaded checkpoint after fixing "module." prefix

2026-06-16 17:09:13,008 - root - INFO - Restored from epoch 29, iteration 66149
2026-06-16 17:09:13,008 - root - INFO - Restored from epoch 29, iteration 66149
2026-06-16 17:09:13,008 - root - INFO - Restored from epoch 29, iteration 661492026-06-16 17:09:13,008 - root - INFO - Restored from epoch 29, iteration 66149

2026-06-16 17:09:13,026 - root - INFO - Loading model from checkpoint: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/best_ckpt.tar
2026-06-16 17:09:13,026 - root - INFO - Starting Model Inference Loop...
2026-06-16 17:09:13,026 - root - INFO - Loading model from checkpoint: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/best_ckpt.tar2026-06-16 17:09:13,026 - root - INFO - Loading model from checkpoint: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/best_ckpt.tar

2026-06-16 17:09:13,026 - root - INFO - Loading model from checkpoint: /work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/checkpoints/best_ckpt.tar2026-06-16 17:09:13,027 - root - INFO - Starting Model Inference Loop...

2026-06-16 17:09:13,027 - root - INFO - Starting Model Inference Loop...
2026-06-16 17:09:13,027 - root - INFO - Starting Model Inference Loop...
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py:653: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp, dtype=self.amp_dtype):
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py:653: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp, dtype=self.amp_dtype):
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py:653: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp, dtype=self.amp_dtype):
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py:653: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp, dtype=self.amp_dtype):
Ensemble forecast 0, members 0-1:   0%|          | 0/56 [00:00<?, ?it/s]2026-06-16 17:09:20,017 - root - INFO - ===== STEPPER predict_sync: FIRST FORWARD PASS DIAGNOSTIC =====
2026-06-16 17:09:20,017 - root - INFO - input_surface shape: torch.Size([1, 8, 180, 360]), dtype: torch.float32
2026-06-16 17:09:22,109 - root - INFO - input_surface[:1,:,0,0]: tensor([[-1.4045, -0.7291,  0.8465, -2.9376, -0.2226, -1.1169, -0.5323,  0.9747]],
       device='cuda:0')
2026-06-16 17:09:22,290 - root - INFO - input_surface stats: min=-4.670988, max=5.514760, mean=0.159681
2026-06-16 17:09:22,290 - root - INFO - constant_boundary_data shape: torch.Size([1, 4, 180, 360])
2026-06-16 17:09:22,291 - root - INFO - constant_boundary_data[:1,:,0,0]: tensor([[ 2.8279,  1.2951, -0.5940,  2.7703]], device='cuda:0')
2026-06-16 17:09:22,291 - root - INFO - varying_boundary_data[:,0] shape: torch.Size([1, 3, 180, 360])
2026-06-16 17:09:22,292 - root - INFO - varying_boundary_data[:1,0,:,0,0]: tensor([[ 1.2916, -0.3275,  0.5966]], device='cuda:0')
2026-06-16 17:09:22,292 - root - INFO - input_upper_air shape: torch.Size([1, 5, 18, 180, 360])
2026-06-16 17:09:22,293 - root - INFO - input_upper_air[:1,:,:,0,0]: tensor([[[ 1.4561e+00,  1.5164e+00,  1.8433e+00,  2.1384e+00,  2.5479e+00,
           2.1512e+00,  1.8544e+00,  1.8538e+00,  1.5753e+00,  6.3904e-01,
           6.7962e-03, -1.3312e+00, -1.7090e+00, -1.7628e+00, -1.7291e+00,
          -1.6114e+00, -1.5331e+00, -1.4095e+00],
         [-3.8103e-01, -4.9101e-01, -3.5717e-01, -3.6427e-01, -5.8975e-01,
          -7.9686e-01, -1.0703e+00, -1.1143e+00, -1.0598e+00, -9.9481e-01,
          -8.9002e-01, -6.2720e-01, -4.9721e-01, -4.5168e-01, -4.6846e-01,
          -4.3518e-01, -4.0748e-01, -4.5908e-01],
         [-8.7678e-02, -6.2753e-03, -5.2211e-02,  1.2940e-02, -1.3921e-01,
          -1.6821e-01, -2.3602e-02, -4.3408e-02, -2.9684e-03,  2.8508e-02,
           9.5524e-02,  1.9428e-01,  1.5985e-01,  3.0449e-02, -1.6926e-01,
          -4.3038e-01, -4.3667e-01, -5.6548e-01],
         [ 1.5354e+00,  1.4445e+00,  1.2921e+00,  1.1472e+00,  7.0084e-01,
           2.9577e-01, -1.5721e-01, -5.7308e-01, -4.1598e-01,  3.7187e-01,
           9.4942e-01,  1.8858e+00,  2.5988e+00,  2.9019e+00,  3.0133e+00,
           3.0653e+00,  3.0811e+00,  3.0596e+00],
         [-3.7266e-01, -3.4262e-01, -3.1326e-01, -3.2240e-01, -4.5475e-01,
          -6.9713e-01, -6.4943e-01, -7.2783e-01, -9.5135e-01, -1.3438e+00,
          -1.5362e+00, -2.6249e-01, -1.1230e-01, -2.5543e-01, -2.1448e-01,
          -5.3378e-01, -3.9534e-01,  8.6021e-01]]], device='cuda:0')
2026-06-16 17:09:22,294 - root - INFO - Model training mode: False
2026-06-16 17:09:22,294 - root - INFO - Raw model training mode: False
2026-06-16 17:09:43,439 - root - INFO - out_surface shape: torch.Size([1, 8, 180, 360]), dtype: torch.float16
2026-06-16 17:09:43,498 - root - INFO - out_surface[:1,:,0,0]: tensor([[-1.3896, -0.8022,  0.8604, -2.9531, -0.2009, -1.1230, -0.5337,  0.9707]],
       device='cuda:0', dtype=torch.float16)
2026-06-16 17:09:43,498 - root - INFO - out_surface stats: min=-4.761719, max=5.484375, mean=0.155151
2026-06-16 17:09:43,500 - root - INFO - out_upper_air[:1,:,:,0,0]: tensor([[[ 1.4424e+00,  1.5371e+00,  1.8096e+00,  2.1719e+00,  2.5703e+00,
           2.1602e+00,  1.8994e+00,  1.8428e+00,  1.5010e+00,  5.8545e-01,
          -6.0089e-02, -1.3154e+00, -1.6719e+00, -1.7471e+00, -1.7207e+00,
          -1.6133e+00, -1.5332e+00, -1.3877e+00],
         [-4.1162e-01, -4.2456e-01, -3.8525e-01, -4.0869e-01, -5.9912e-01,
          -8.0664e-01, -1.1318e+00, -1.1318e+00, -1.0596e+00, -9.8682e-01,
          -8.4521e-01, -6.4062e-01, -4.8877e-01, -4.5142e-01, -3.8574e-01,
          -3.2861e-01, -2.7368e-01, -3.0981e-01],
         [-1.4600e-01, -1.3000e-01, -7.2876e-02, -1.3965e-01, -1.4502e-01,
          -1.8994e-01, -7.8857e-02,  1.6527e-03, -1.3771e-02,  4.3213e-02,
           6.7139e-02,  1.8689e-01,  1.4343e-01,  1.0048e-02, -1.2573e-01,
          -3.3496e-01, -4.3091e-01, -5.4297e-01],
         [ 1.5508e+00,  1.4482e+00,  1.3027e+00,  1.1328e+00,  7.1240e-01,
           2.9810e-01, -1.5137e-01, -5.5664e-01, -4.1260e-01,  3.7817e-01,
           9.4971e-01,  1.8770e+00,  2.5977e+00,  2.9141e+00,  3.0137e+00,
           3.0605e+00,  3.0625e+00,  3.0645e+00],
         [-3.6450e-01, -3.4277e-01, -3.2300e-01, -3.0347e-01, -4.5850e-01,
          -6.9434e-01, -6.5967e-01, -7.3291e-01, -9.6777e-01, -1.3594e+00,
          -1.5322e+00, -2.7490e-01,  6.1249e-02, -1.3660e-01, -1.3049e-01,
          -3.0249e-01, -1.1902e-01,  9.2041e-01]]], device='cuda:0',
       dtype=torch.float16)
2026-06-16 17:09:43,500 - root - INFO - ===== END STEPPER FIRST FORWARD PASS DIAGNOSTIC =====
Ensemble forecast 0, members 0-1:   2%|1         | 1/56 [00:23<21:32, 23.50s/it]Ensemble forecast 0, members 0-1:   4%|3         | 2/56 [00:23<08:45,  9.74s/it]Ensemble forecast 0, members 0-1:   5%|5         | 3/56 [00:23<04:42,  5.34s/it]Ensemble forecast 0, members 0-1:   7%|7         | 4/56 [00:23<02:50,  3.27s/it]Ensemble forecast 0, members 0-1:   9%|8         | 5/56 [00:23<01:48,  2.13s/it]Ensemble forecast 0, members 0-1:  11%|#         | 6/56 [00:24<01:12,  1.44s/it]Ensemble forecast 0, members 0-1:  12%|#2        | 7/56 [00:24<00:49,  1.00s/it]Ensemble forecast 0, members 0-1:  14%|#4        | 8/56 [00:24<00:34,  1.40it/s]Ensemble forecast 0, members 0-1:  16%|#6        | 9/56 [00:24<00:24,  1.91it/s]Ensemble forecast 0, members 0-1:  18%|#7        | 10/56 [00:24<00:18,  2.53it/s]Ensemble forecast 0, members 0-1:  20%|#9        | 11/56 [00:24<00:13,  3.27it/s]Ensemble forecast 0, members 0-1:  21%|##1       | 12/56 [00:24<00:10,  4.10it/s]Ensemble forecast 0, members 0-1:  23%|##3       | 13/56 [00:24<00:08,  4.97it/s]Ensemble forecast 0, members 0-1:  25%|##5       | 14/56 [00:24<00:07,  5.83it/s]Ensemble forecast 0, members 0-1:  27%|##6       | 15/56 [00:24<00:06,  6.62it/s]Ensemble forecast 0, members 0-1:  29%|##8       | 16/56 [00:25<00:05,  7.31it/s]Ensemble forecast 0, members 0-1:  30%|###       | 17/56 [00:25<00:04,  7.88it/s]Ensemble forecast 0, members 0-1:  32%|###2      | 18/56 [00:25<00:04,  8.34it/s]Ensemble forecast 0, members 0-1:  34%|###3      | 19/56 [00:25<00:04,  8.70it/s]Ensemble forecast 0, members 0-1:  36%|###5      | 20/56 [00:25<00:04,  8.97it/s]Ensemble forecast 0, members 0-1:  38%|###7      | 21/56 [00:25<00:03,  9.17it/s]Ensemble forecast 0, members 0-1:  39%|###9      | 22/56 [00:25<00:03,  9.32it/s]Ensemble forecast 0, members 0-1:  41%|####1     | 23/56 [00:25<00:03,  9.44it/s]Ensemble forecast 0, members 0-1:  43%|####2     | 24/56 [00:25<00:03,  9.51it/s]Ensemble forecast 0, members 0-1:  45%|####4     | 25/56 [00:25<00:03,  9.57it/s]Ensemble forecast 0, members 0-1:  46%|####6     | 26/56 [00:26<00:03,  9.60it/s]Ensemble forecast 0, members 0-1:  48%|####8     | 27/56 [00:26<00:03,  9.65it/s]Ensemble forecast 0, members 0-1:  50%|#####     | 28/56 [00:26<00:02,  9.67it/s]Ensemble forecast 0, members 0-1:  52%|#####1    | 29/56 [00:26<00:02,  9.69it/s]Ensemble forecast 0, members 0-1:  54%|#####3    | 30/56 [00:26<00:02,  9.69it/s]Ensemble forecast 0, members 0-1:  55%|#####5    | 31/56 [00:26<00:02,  9.69it/s]Ensemble forecast 0, members 0-1:  57%|#####7    | 32/56 [00:26<00:02,  9.69it/s]Ensemble forecast 0, members 0-1:  59%|#####8    | 33/56 [00:26<00:02,  9.70it/s]Ensemble forecast 0, members 0-1:  61%|######    | 34/56 [00:26<00:02,  9.70it/s]Ensemble forecast 0, members 0-1:  62%|######2   | 35/56 [00:27<00:02,  9.72it/s]Ensemble forecast 0, members 0-1:  64%|######4   | 36/56 [00:27<00:02,  9.72it/s]Ensemble forecast 0, members 0-1:  66%|######6   | 37/56 [00:27<00:01,  9.68it/s]Ensemble forecast 0, members 0-1:  70%|######9   | 39/56 [00:27<00:01,  9.86it/s]Ensemble forecast 0, members 0-1:  71%|#######1  | 40/56 [00:27<00:01,  9.81it/s]Ensemble forecast 0, members 0-1:  73%|#######3  | 41/56 [00:27<00:01,  9.78it/s]Ensemble forecast 0, members 0-1:  75%|#######5  | 42/56 [00:27<00:01,  9.76it/s]Ensemble forecast 0, members 0-1:  77%|#######6  | 43/56 [00:27<00:01,  9.74it/s]Ensemble forecast 0, members 0-1:  79%|#######8  | 44/56 [00:27<00:01,  9.72it/s]Ensemble forecast 0, members 0-1:  80%|########  | 45/56 [00:28<00:01,  9.70it/s]Ensemble forecast 0, members 0-1:  82%|########2 | 46/56 [00:28<00:01,  9.68it/s]Ensemble forecast 0, members 0-1:  84%|########3 | 47/56 [00:28<00:00,  9.69it/s]Ensemble forecast 0, members 0-1:  86%|########5 | 48/56 [00:28<00:00,  9.68it/s]Ensemble forecast 0, members 0-1:  88%|########7 | 49/56 [00:28<00:00,  9.67it/s]Ensemble forecast 0, members 0-1:  89%|########9 | 50/56 [00:28<00:00,  9.66it/s]Ensemble forecast 0, members 0-1:  91%|#########1| 51/56 [00:28<00:00,  9.67it/s]Ensemble forecast 0, members 0-1:  93%|#########2| 52/56 [00:28<00:00,  9.67it/s]Ensemble forecast 0, members 0-1:  95%|#########4| 53/56 [00:28<00:00,  9.68it/s]Ensemble forecast 0, members 0-1:  96%|#########6| 54/56 [00:28<00:00,  9.67it/s]Ensemble forecast 0, members 0-1:  98%|#########8| 55/56 [00:29<00:00,  9.69it/s]Ensemble forecast 0, members 0-1: 100%|##########| 56/56 [00:29<00:00,  9.71it/s]Ensemble forecast 0, members 0-1: 100%|##########| 56/56 [00:29<00:00,  1.92it/s]
[rank2]: Traceback (most recent call last):
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 211, in _acquire_with_cache_info
[rank2]:     file = self._cache[self._key]
[rank2]:            ~~~~~~~~~~~^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/lru_cache.py", line 56, in __getitem__
[rank2]:     value = self._cache[key]
[rank2]:             ~~~~~~~~~~~^^^^^
[rank2]: KeyError: [<class 'netCDF4._netCDF4.Dataset'>, ('/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001_run.0000-0001_output.nc',), 'a', (('clobber', True), ('diskless', False), ('format', 'NETCDF4'), ('persist', False)), 'bde18f96-1c3d-4191-b644-09dff6f345d0']

[rank2]: During handling of the above exception, another exception occurred:

[rank2]: Traceback (most recent call last):
[rank2]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1470, in <module>
[rank2]:     stepper.predict()
[rank2]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 463, in predict
[rank2]:     valid_time, valid_logs = self.predict_sync()
[rank2]:                              ^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 844, in predict_sync
[rank2]:     self.save_prediction(ensemble_datasets, particle_idxs, ensemble_start, ensemble_end)
[rank2]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1128, in save_prediction
[rank2]:     dataset.to_netcdf(filepath, mode='w', compute=True, format="NETCDF4")
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/core/dataset.py", line 2380, in to_netcdf
[rank2]:     return to_netcdf(  # type: ignore[return-value]  # mypy cannot resolve the overloads:(
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/api.py", line 1911, in to_netcdf
[rank2]:     store = store_open(target, mode, format, group, **kwargs)
[rank2]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 452, in open
[rank2]:     return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 393, in __init__
[rank2]:     self.format = self.ds.data_model
[rank2]:                   ^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 461, in ds
[rank2]:     return self._acquire()
[rank2]:            ^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 455, in _acquire
[rank2]:     with self._manager.acquire_context(needs_lock) as root:
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/contextlib.py", line 137, in __enter__
[rank2]:     return next(self.gen)
[rank2]:            ^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 199, in acquire_context
[rank2]:     file, cached = self._acquire_with_cache_info(needs_lock)
[rank2]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 217, in _acquire_with_cache_info
[rank2]:     file = self._opener(*self._args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "src/netCDF4/_netCDF4.pyx", line 2521, in netCDF4._netCDF4.Dataset.__init__
[rank2]:   File "src/netCDF4/_netCDF4.pyx", line 2158, in netCDF4._netCDF4._ensure_nc_success
[rank2]: PermissionError: [Errno 13] Permission denied: '/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001_run.0000-0001_output.nc'
[rank0]: Traceback (most recent call last):
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 211, in _acquire_with_cache_info
[rank0]:     file = self._cache[self._key]
[rank0]:            ~~~~~~~~~~~^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/lru_cache.py", line 56, in __getitem__
[rank0]:     value = self._cache[key]
[rank0]:             ~~~~~~~~~~~^^^^^
[rank0]: KeyError: [<class 'netCDF4._netCDF4.Dataset'>, ('/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001_run.0000-0001_output.nc',), 'a', (('clobber', True), ('diskless', False), ('format', 'NETCDF4'), ('persist', False)), '0ad52ed2-bcf3-4d22-84ed-dad99dc524f6']

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1470, in <module>
[rank0]:     stepper.predict()
[rank0]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 463, in predict
[rank0]:     valid_time, valid_logs = self.predict_sync()
[rank0]:                              ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 844, in predict_sync
[rank0]:     self.save_prediction(ensemble_datasets, particle_idxs, ensemble_start, ensemble_end)
[rank0]:   File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1128, in save_prediction
[rank0]:     dataset.to_netcdf(filepath, mode='w', compute=True, format="NETCDF4")
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/core/dataset.py", line 2380, in to_netcdf
[rank0]:     return to_netcdf(  # type: ignore[return-value]  # mypy cannot resolve the overloads:(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/api.py", line 1911, in to_netcdf
[rank0]:     store = store_open(target, mode, format, group, **kwargs)
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 452, in open
[rank0]:     return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 393, in __init__
[rank0]:     self.format = self.ds.data_model
[rank0]:                   ^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 461, in ds
[rank0]:     return self._acquire()
[rank0]:            ^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/netCDF4_.py", line 455, in _acquire
[rank0]:     with self._manager.acquire_context(needs_lock) as root:
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/contextlib.py", line 137, in __enter__
[rank0]:     return next(self.gen)
[rank0]:            ^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 199, in acquire_context
[rank0]:     file, cached = self._acquire_with_cache_info(needs_lock)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/xarray/backends/file_manager.py", line 217, in _acquire_with_cache_info
[rank0]:     file = self._opener(*self._args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "src/netCDF4/_netCDF4.pyx", line 2521, in netCDF4._netCDF4.Dataset.__init__
[rank0]:   File "src/netCDF4/_netCDF4.pyx", line 2158, in netCDF4._netCDF4._ensure_nc_success
[rank0]: PermissionError: [Errno 13] Permission denied: '/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001_run.0000-0001_output.nc'
2026-06-16 17:09:50,807 - root - INFO - Validation logs: {'total_time': 37.779635429382324, 'data_time': 0, 'inference_time': 28.973570585250854, 'conversion_time': 0.39941954612731934, 'obs_time': 0, 'save_time': 1.4213078022003174}
2026-06-16 17:09:50,813 - root - INFO - DONE ---- rank 1
2026-06-16 17:09:51,109 - root - INFO - Validation logs: {'total_time': 38.081669330596924, 'data_time': 0, 'inference_time': 28.946374654769897, 'conversion_time': 0.4263145923614502, 'obs_time': 0, 'save_time': 1.7255420684814453}
2026-06-16 17:09:51,115 - root - INFO - DONE ---- rank 3
W0616 17:09:58.888000 3226795 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3226798 closing signal SIGTERM
W0616 17:09:58.888000 3226795 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3226799 closing signal SIGTERM
W0616 17:09:58.888000 3226795 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3226800 closing signal SIGTERM
E0616 17:09:59.503000 3226795 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 3226797) of binary: /work/11095/jwan4/conda-envs/sfno_pangu/bin/python
Traceback (most recent call last):
  File "/work/11095/jwan4/conda-envs/sfno_pangu/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-06-16_17:09:58
  host      : c561-008.stampede3.tacc.utexas.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 3226797)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
