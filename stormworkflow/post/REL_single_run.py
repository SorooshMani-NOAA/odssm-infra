import argparse
import logging
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

pd.options.mode.copy_on_write = True


def stack_station_coordinates(x, y):
    """
    Create numpy.column_stack based on
    coordinates of observation points
    """
    coord_combined = np.column_stack([x, y])
    return coord_combined


def create_search_tree(longitude, latitude):
    """
    Create scipy.spatial.CKDTree based on Lat. and Long.
    """
    long_lat = np.column_stack((longitude.T.ravel(), latitude.T.ravel()))
    tree = sp.spatial.cKDTree(long_lat)
    return tree


def find_nearby_prediction(ds, variable, indices):
    """
    Reads netcdf file, target variable, and indices
    Returns max value among corresponding indices for each point 
    """
    obs_count = indices.shape[0]  # total number of search/observation points
    max_prediction_index = len(ds.node.values)  # total number of nodes

    prediction_prob = np.zeros(obs_count)  # assuming all are dry (probability of zero)

    for obs_point in range(obs_count):
        idx_arr = np.delete(
            indices[obs_point], np.where(indices[obs_point] == max_prediction_index)[0]
        )  # len is length of surrogate model array
        val_arr = ds[variable].values[idx_arr]
        val_arr = np.nan_to_num(val_arr)  # replace nan with zero (dry node)

        # # Pick the nearest non-zero probability (option #1)
        # for val in val_arr:
        #     if val > 0.0:
        #         prediction_prob[obs_point] = round(val,4) #round to 0.1 mm
        #         break

        # pick the largest value (option #2)
        if val_arr.size > 0:
            prediction_prob[obs_point] = val_arr.max()
    return prediction_prob


def main(args):
    storm_name = args.storm_name.capitalize()
    storm_year = args.storm_year
    leadtime = args.leadtime
    prob_nc_path = Path(args.prob_nc_path)
    obs_df_path = Path(args.obs_df_path)
    save_dir = args.save_dir

    # *.nc file coordinates
    thresholds_ft = [3, 4, 5, 6, 9]  # in ft [3, 6, 9]  # in ft
    thresholds_m = [round(i * 0.3048, 4) for i in thresholds_ft]  # convert to meter
    sources = ['model', 'surrogate']

    # attributes of input files
    prediction_variable = 'probabilities'
    obs_attribute = 'Elev_m_xGEOID20b'

    # search criteria
    max_distance = 1500  # [in meters] to set distance_upper_bound
    max_neighbors = 10  # to set k

    # Load obs file, extract storm obs points and coordinates
    df_obs = pd.read_csv(obs_df_path)
    Event_name = f'{storm_name}_{storm_year}'
    df_obs_storm = df_obs[df_obs.Event == Event_name]
    obs_coordinates = stack_station_coordinates(
        df_obs_storm.Longitude.values, df_obs_storm.Latitude.values
    )

    blank_arr = np.empty((len(thresholds_ft), 1, 1, len(sources), len(df_obs_storm)))
    blank_arr[:] = np.nan

    obs_true_arr = blank_arr.copy()
    pred_prob_arr = blank_arr.copy()

    # Load probabilities.nc file
    ds_prob = xr.open_dataset(prob_nc_path)

    # Loop through thresholds and sources and find corresponding values from probabilities.nc
    threshold_count = -1
    for threshold in thresholds_m:
        threshold_count += 1
        source_count = -1
        for source in sources:
            source_count += 1
            ds_temp = ds_prob.sel(level=threshold, source=source)
            tree = create_search_tree(ds_temp.x.values, ds_temp.y.values)
            dist, indices = tree.query(
                obs_coordinates, k=max_neighbors, distance_upper_bound=max_distance * 1e-5
            )  # 0.01 is equivalent to 1000 m
            prediction_prob = find_nearby_prediction(
                ds=ds_temp, variable=prediction_variable, indices=indices
            )
            obs_true = df_obs_storm[obs_attribute] > threshold

            # Enter observed above threshold and prediction probability into array
            obs_true_arr[threshold_count, 0, 0, source_count, :] = obs_true
            pred_prob_arr[threshold_count, 0, 0, source_count, :] = prediction_prob

    ds_REL = xr.Dataset(
        coords=dict(
            threshold=thresholds_ft,
            storm=[storm_name],
            leadtime=[leadtime],
            source=sources,
            points=range(len(df_obs_storm)),
        ),
        data_vars=dict(
            obs_true=(['threshold', 'storm', 'leadtime', 'source', 'points'], obs_true_arr),
            pred_prob=(['threshold', 'storm', 'leadtime', 'source', 'points'], pred_prob_arr),
        ),
    )
    ds_REL.to_netcdf(
        os.path.join(
            save_dir, f'{storm_name}_{storm_year}_{leadtime}hr_leadtime_reliability.nc'
        )
    )

    # plot reliability curves
    marker_list = ['s', '.']
    linestyle_list = ['dotted', '-']
    threshold_count = -1
    for threshold in thresholds_ft:
        threshold_count += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axline(
            (0.0, 0.0), (1.0, 1.0), linestyle='--', color='grey', label='perfect reliability'
        )
        source_count = -1
        for source in sources:
            source_count += 1
            true_prob, pred_prob = calibration_curve(
                obs_true_arr[threshold_count, 0, 0, source_count, :],
                pred_prob_arr[threshold_count, 0, 0, source_count, :],
                n_bins=5,
                strategy='uniform',
            )
            plt.plot(
                pred_prob,
                true_prob,
                label=f'{source}',
                marker=marker_list[source_count],
                linestyle=linestyle_list[source_count],
                markersize=5,
            )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.legend(loc='lower right')
        plt.xlabel(f'Predicted probability of exceedance')
        plt.ylabel(f'Observed fraction of exceedances')
        plt.grid(True)

        plt.title(
            f'{storm_name}_{storm_year}, {leadtime}-hr leadtime, {threshold} ft threshold: N={len(df_obs_storm)}'
        )
        plt.savefig(
            os.path.join(
                save_dir, f'REL_{storm_name}_{storm_year}_{leadtime}hr_leadtime_{threshold}_ft.png'
            )
        )
        plt.close()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--storm_name', help='name of the storm', type=str)

    parser.add_argument('--storm_year', help='year of the storm', type=int)

    parser.add_argument('--leadtime', help='OFCL track leadtime hr', type=int)

    parser.add_argument('--prob_nc_path', help='path to probabilities.nc', type=str)

    parser.add_argument('--obs_df_path', help='Path to observations dataframe', type=str)

    # optional
    parser.add_argument(
        '--save_dir', help='directory for saving analysis', default=os.getcwd(), type=str
    )

    main(parser.parse_args())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    cli()
