from prefect import Parameter

# Define parameters
param_storm_name = Parameter('name')
param_storm_year = Parameter('year')
param_use_rdhpcs = Parameter('rdhpcs', default=False)
param_use_rdhpcs_post = Parameter('rdhpcs_post', default=False)
param_use_parametric_wind = Parameter('parametric_wind', default=False)
param_run_id = Parameter('run_id')
param_schism_dir = Parameter('schism_dir')
param_subset_mesh = Parameter('subset_mesh', default=False)
param_past_forecast = Parameter('past_forecast', default=False)
param_hr_prelandfall = Parameter('hr_before_landfall', default=-1)
param_ensemble = Parameter('ensemble', default=False)
param_ensemble_n_perturb = Parameter('ensemble_num_perturbations', default=40)
param_ensemble_sample_rule = Parameter('ensemble_sample_rule', default='korobov')

param_mesh_hmax = Parameter('mesh_hmax', default=20000)
param_mesh_hmin_low = Parameter('mesh_hmin_low', default=1500)
param_mesh_rate_low = Parameter('mesh_rate_low', default=2e-3)
param_mesh_trans_elev = Parameter('mesh_cutoff', default=-200)
param_mesh_hmin_high = Parameter('mesh_hmin_high', default=300)
param_mesh_rate_high = Parameter('mesh_rate_high', default=1e-3)
