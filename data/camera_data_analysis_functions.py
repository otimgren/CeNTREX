"""
Contains functions used for analyzing rotational cooling data taken using an EMCCD camera
"""

# Standard data analysis libraries
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from scipy.stats import median_abs_deviation

# matplotlib for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

# lmfit for fitting
from lmfit import Model
from lmfit.models import Gaussian2dModel

# fancy typing
from typing import Union, Tuple

# Custom functions
from hdf_utils import load_measurement_data_devices_attrs, load_camera_data

# Parallelization
from joblib import Parallel, delayed


def get_data(data_path: Union[str, Path], run_name: str, scan_parameter = None) -> pd.DataFrame:
    """
    Fetches camera and DAQ data from hdf file
    """
    devices = ()

    #Get data for all channels of PXIe
    pxie, pxie_time, pxie_attrs, data_devices = load_measurement_data_devices_attrs(data_path,
                                                                                    run_name, devices,
                                                                                   pxie_path = 'readout')
    #Separate data for each channel to correct array
    # Define which channel on PXIe corresponds to which data:
    yag_channel = 0
    pmt_channel = 1
    pd_channel = 2
    pdn_channel = 3
    shutter_channel = 4
    pdrc_channel = 5

    data_dict_CAM = {}
    data_dict_DAQ = {}

    # Get the data for the camera
    camera_data, camera_time = load_camera_data(data_path, run_name)

    n_data = np.min((len(camera_data)+1, len(pxie)+1))

    data_dict_CAM["Camera data"] = [camera_data[idx] for idx in range(1, len(camera_data)+1)]
    data_dict_CAM["Camera time"] = [camera_time[idx] for idx in range(1, len(camera_data)+1)]
    
    
    # Get data for NI PXIe-5171
    data_dict_DAQ["YAG PD"] =[pxie[idx][:,yag_channel] for idx in range(1,len(pxie)+1)]
    data_dict_DAQ["Absorption PD"] =[pxie[idx][:,pd_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict_DAQ["Abs Norm PD"] =[pxie[idx][:,pdn_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict_DAQ["Shutter"] =[pxie[idx][:,shutter_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict_DAQ["RC Norm PD"] =[pxie[idx][:,pdrc_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict_DAQ["NI5171Time"] =[pxie_time[idx].astype(float) for idx in range(1,len(pxie)+1)]

    if scan_parameter:
        data_dict_DAQ[scan_parameter] =[float(pxie_attrs[idx][scan_parameter]) for idx in range(1,len(pxie)+1)]
    
    dataframe_CAM = pd.DataFrame(data_dict_CAM)
    dataframe_CAM["rounded_time"] = np.round(dataframe_CAM["Camera time"])
    dataframe_DAQ = pd.DataFrame(data_dict_DAQ)
    dataframe_DAQ["rounded_time"] = np.round(dataframe_DAQ["NI5171Time"])

    dataframe = dataframe_CAM.merge(dataframe_DAQ, how = 'outer', left_index=True, right_index=True)

    return dataframe


def transform_data(df: pd.DataFrame, plots = False)-> pd.DataFrame:
    """
    Applies transformations to data (e.g. integrate absorption)
    """
    #Define slices used for integrals
    slice_absm = np.s_[-3000:] #Slice for determining absorption background
    slice_absi = np.s_[10:2000] #Slice for calculating absorption integral

    #Generate a column that labels datapoints by ablation spot
    df["Ablation spot"] = np.array(df.index/10, dtype = int)

    #Generate a column for normalized absorption
    df["Normalized Absorption"] = df["Absorption PD"]/df["Abs Norm PD"]
    df["Normalized Absorption"] = (df["Normalized Absorption"]
                                        /(df["Normalized Absorption"].apply(lambda x: x[slice_absm].mean())))

    #Calculate integrated absorption signal from the normalized absorption
    def calculate_integrated_absorption(trace):
        return -np.trapz(trace[slice_absi] - np.mean(trace[slice_absm]))

    df["Integrated absorption"] = df["Normalized Absorption"].apply(calculate_integrated_absorption)

    # Add additional columns for cuts
    # Define limits for cuts
    yag_cutoff = 250 #Limit for considering YAG to have fired
    ptn_cutoff = 10000 #Limit for considering absorption laser to be on
    rc_cutoff = 1500
    abs_cutoff = 5 #Limit for absorption integral (if below, there might be no molecules during the pulse so it is removed)
    timing_cutoff = 0.1 # How much camera and DAQ are allowed to be out of sync until cut off data
    
    # Cut for absorption
    df["Abs cut"] = df["Integrated absorption"] > abs_cutoff
    # Check if YAG fired
    df["YAG fired"] = df["YAG PD"].apply(np.max) > yag_cutoff

    # Check if absorption laser was on
    df["Absorption ON"] = df["Abs Norm PD"].apply(np.min) > ptn_cutoff

    #Check if rotational cooling was on
    df["RC ON"] = df["RC Norm PD"].apply(np.min) > rc_cutoff

    #Check if shutter was open
    df["Shutter OPEN"] = df["Shutter"].apply(np.max) > 10000

    # Check timing between NI DAQ and Camera
    df["time_difference"] = df["Camera time"] - df["NI5171Time"]
    df["TimingCut"] = np.abs(df["time_difference"]) < timing_cutoff

    # If desired, add plots
    if plots:
        fig, ax = plt.subplots(2,2, figsize = (16,9))
        
        # Plot for absorption cutoff
        df["Integrated absorption"][df["YAG fired"] & df["Absorption ON"]].hist(bins = 20, ax = ax[0,0])
        df["Integrated absorption"][~df["YAG fired"] & df["Absorption ON"]].hist(bins = 20, ax = ax[0,0])
        ax[0,0].axvline(abs_cutoff, c = 'k', ls = '--')
        ax[0,0].set_xlabel('Integrated absorption')

        # Plot for YAG firing
        df["YAG PD"].apply(np.max).plot.line(ax = ax[0,1])
        ax[0,1].axhline(yag_cutoff, c = 'k', ls = '--')
        ax[0,1].set_xlabel('Data row')
        ax[0,1].set_ylabel('YAG PD signal')

        # Plot for absorption laser being on
        df["Abs Norm PD"].apply(np.min).plot.line(ax = ax[1,0])
        ax[1,0].axhline(ptn_cutoff, c = 'k', ls = '--')
        ax[1,0].set_xlabel('Data row')
        ax[1,0].set_ylabel('Absorption norm PD signal')

        # Plot for RC laser being on
        df["RC Norm PD"].apply(np.min).plot.line(ax = ax[1,1])
        ax[1,1].axhline(rc_cutoff, c = 'k', ls = '--')
        ax[1,1].set_xlabel('Data row')
        ax[1,1].set_ylabel('RC PD signal')

    return df

def cut_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cuts to data
    """

    mask = (df["YAG fired"] & df["Absorption ON"] & df["Abs cut"] & df["TimingCut"])
    #How much of acquired data being used?
    print("{:.1f}% data discarded".format(100-(mask.sum())/len(df)*100))
    df = df[mask]

    return df

def gaussian2D(x, y, A, center_x, center_y, sigmax, sigmay, C, phi):
    """
    Returns a Gaussian with center at (x0, y0), standard deviation σx/y, amplitude A, and constant offset C, and possibly rotated by
    angle ϕ
    """
    xp = (x - center_x)*np.cos(phi) - (y - center_y)*np.sin(phi)
    yp = (x - center_x)*np.sin(phi) + (y - center_y)*np.cos(phi)
    R = (xp/sigmax)**2 + (yp/sigmay)**2
    
    return A * np.exp(-R/2) + C

def fit_2D_gaussian(data: pd.DataFrame, mean_background: np.array, x_max = None, y_max = None, plot = False, 
                    fig = None, ax = None, ROI = [0,0,512,512], use_guess = False):
    """
    Fits a 2D gaussian to fluorescence data obtained using CCD camera
    """

    # Calculate the mean fluorescence for this set of data
    mean_fluorescence = np.nanmean(np.array(list(data['Camera data'])), axis = 0).T[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    

    # Calculate background subtracted fluorescence
    mean_fluorescence_bs = mean_fluorescence - mean_background[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    # print(mean_fluorescence_bs.max())
    index = mean_fluorescence_bs > 300
    mean_fluorescence_bs[index] = mean_fluorescence_bs.mean()
    # print(mean_fluorescence_bs.max())


    ### Fit a 2D gaussian to the data
    # Define lmfit Model

    # Get approximations for center of gaussian if not provided
    if not x_max or x_max >= np.max(mean_fluorescence.shape[0]) or x_max <= 0:
        x_max = np.unravel_index(np.argmax(mean_fluorescence_bs),mean_fluorescence_bs.shape)[1]
    if not y_max or y_max >= np.max(mean_fluorescence.shape[0]) or y_max <= 0:
        y_max = np.unravel_index(np.argmax(mean_fluorescence_bs),mean_fluorescence_bs.shape)[0]
    
    # Approximate amplitude
    A = mean_fluorescence_bs[x_max, y_max]

    # Define Gaussian model and setup parameters
    model = Model(gaussian2D, independent_vars=['x', 'y'])

    # Setup arrays that contain the x- and y-coordinates
    x = np.arange(mean_fluorescence.shape[0])
    y = np.arange(mean_fluorescence.shape[1])
    x0, x1 = 106, 406
    y0, y1 = 106, 406
    x_fit_slice = np.s_[x0:x1]
    y_fit_slice = np.s_[y0:y1]
    x_fit = x[x_fit_slice]
    y_fit = y[y_fit_slice]
    X, Y = np.meshgrid(y_fit,x_fit)
    fit_data = mean_fluorescence_bs[x_fit_slice, y_fit_slice]

    # Convert data into vectors
    # x_coords = []
    # y_coords = []
    # counts = []
    # for i in range(mean_fluorescence_bs.shape[0]):
    #     for j in range(mean_fluorescence_bs.shape[1]):
    #         x_coords.append(j)
    #         y_coords.append(i)
    #         counts.append(mean_fluorescence_bs[i,j])

    # x_coords = np.array(x_coords)
    # y_coords = np.array(y_coords)
    # counts = np.array(counts)

    # Guess the parameters using a built in lmfit model
    guess_params = Gaussian2dModel().guess(fit_data.flatten(), x = X.flatten(), y = Y.flatten())
    A = guess_params['height'].value
    x_max = guess_params['centerx'].value
    y_max = guess_params['centery'].value
    sigmax = guess_params['sigmax'].value
    sigmay = guess_params['sigmay'].value
    if A < 0 or 0 > x_max > 512 or 0 > y_max > 512 or not use_guess:
        A = 1
        x_max = 277
        y_max = 225
        sigmax = 30
        sigmay = 15

    # Set parameters for the custom model
    params = model.make_params(C = 0)
    params['A'].set(value = A, min = 0)
    params['center_x'].set(value = x_max, max = x1, min = x0)
    params['center_y'].set(value = y_max, max = y1, min = y0)
    params['phi'].set(value = 0, min = 0, max = np.pi/4)
    params['sigmax'].set(value = sigmax, min = 10, max = 50)
    params['sigmay'].set(value = sigmay, min = 10, max = 50)
    
    # Fit model
    result = model.fit(fit_data.flatten(), x = X.flatten(), y = Y.flatten(), params = params,
                         max_nfev= 1000, method = 'least_squares')
    # result = model.fit(counts, x = x_coords, y = y_coords, params = params, max_nfev= 1000)

    # Make plot if desired
    if plot:
        # If fig and ax were not provided, make new ones
        if not ax and not fig:
            # Set up axses for plot
            fig, ax = plt.subplots(figsize = (32,18))

        divider  = make_axes_locatable(ax)
        top_ax = divider.append_axes("top", 1.05, pad=0.1, sharex=ax)
        right_ax = divider.append_axes("right", 1.05, pad=0.1, sharey=ax)

        # Get rid of some ticklabels
        top_ax.xaxis.set_tick_params(labelbottom=False)
        right_ax.yaxis.set_tick_params(labelleft=False)

        # Plot the mean fluorescence
        imag = ax.imshow(fit_data, vmax = 70, vmin = -22, extent=[x0,x1,y1,y0])
        ax.autoscale(enable=False)

        # Plot fit
        fit_values = result.eval(x = X.flatten(), y = Y.flatten()).reshape(fit_data.shape)
        ax.contour(fit_values, extent=[x0,x1,y0,y1])

        # Colorbar
        fig.colorbar(imag, ax = ax, shrink = 0.9)

        # Find the maximum
        center_x, center_y = result.params['center_x'].value, result.params['center_y'].value
        ax.plot(center_x, center_y, color = 'red', marker = 'x', markersize = 10)

        # Plot profile along lines that intersect at maximum
        v_prof, = right_ax.plot(fit_data[:,int(center_x-x0)],np.arange(x0,fit_data.shape[0]+x0))
        h_prof, = top_ax.plot(np.arange(x0, fit_data.shape[1]+x0), fit_data[int(center_y-y0),:])

        # Lines to indicate where the profile is being cut
        v_line = ax.axvline(center_x, color='k', ls = '--')
        h_line = ax.axhline(center_y, color='k', ls = '--')

    return result

def sum_fluorescence_counts(data: pd.DataFrame, mean_background: np.array, sum_ROI: np.array, ROI = [0,0,512,512], plot = False):
    """
    Sums up camera intensity counts in a specified area of the plot
    """
    # Calculate the mean fluorescence for this set of data
    mean_fluorescence = np.mean(np.array(list(data['Camera data'])), axis = 0).T[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    
    # Calculate background subtracted fluorescence
    mean_fluorescence_bs = mean_fluorescence - mean_background[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    
    # Sometimes there are large spikes in the counts; remove them (are they cosmic rays?)
    index = mean_fluorescence_bs > 150
    mean_fluorescence_bs[index] = mean_fluorescence_bs.mean()

    # Sum up the counts in the ROI:
    x0 = sum_ROI[0]
    y0 = sum_ROI[1]
    x1 = sum_ROI[2]
    y1 = sum_ROI[3]
    summed_counts = mean_fluorescence_bs[x0:x1,y0:y1].sum()

    # Plot the background subtracted fluorescence and the region of interest
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(mean_fluorescence_bs)
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill = False, ec = 'r'))


    return summed_counts

def analyze_frequency_scan(df : pd.DataFrame, mean_background : np.ndarray, scan_param: str, plots = False, center_x = None, center_y = None) -> pd.DataFrame:
    """
    Analyzes data for a frequency scan
    """
    # Find the frequencies that were used in the scan
    frequencies = np.sort(df[scan_param].unique())

    # Initialize containers for results
    data_dict = {
        "frequency": [],
        "results": [],
        "mean_integrated_absorption": [],
        "fluorescence": [],
        "normalized_fluorescence": [],
        "amplitude": [],
        "center_x": [],
        "center_y": [],
        "sigma_x": [],
        "sigma_y": [],
        "rotation": [],
        "redchi": []
    }

    # If plots are required, set them up
    if plots:
        n_axes = len(frequencies)
        n_cols = 4
        n_rows = int(n_axes/n_cols) + 1*int(n_axes%n_cols>0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize = (45,9*n_rows), tight_layout = True)
    
    # Loop over the frequncies and analyze data
    for i, freq  in enumerate(tqdm(frequencies)):
        
        # Extract data at this frequency
        data = df[(df[scan_param] == freq)]
        
        # Fit 2D gaussian to data
        if plots:
            result = fit_2D_gaussian(data, mean_background, plot = plots, ax = axes.ravel()[i], fig = fig, x_max = center_x, y_max = center_x)
        else: 
            result = fit_2D_gaussian(data, mean_background, plot = plots, x_max = center_x, y_max = center_x)

        # Help the fitting by storing center coordinates of fit
        center_x = int(result.params['center_x'].value)
        center_y = int(result.params['center_y'].value)
        
        # Calculate mean integrated absorption for this frequency
        mean_integrated_absorption = np.mean(data['Integrated absorption'])

        # Calculate integral of gaussian fit
        A = result.params['A'].value
        sigmax = result.params['sigmax'].value
        sigmay = result.params['sigmay'].value
        integrated_gaussian = A*np.pi*sigmax*sigmay
        
        # Append data to data dictionary
        data_dict["frequency"].append(freq)
        data_dict["results"].append(result)
        data_dict["mean_integrated_absorption"].append(mean_integrated_absorption)
        data_dict["fluorescence"].append(integrated_gaussian)
        data_dict["normalized_fluorescence"].append(integrated_gaussian/mean_integrated_absorption)
        data_dict["amplitude"].append(result.params["A"].value)
        data_dict["center_x"].append(result.params["center_x"].value)
        data_dict["center_y"].append(result.params["center_y"].value)
        data_dict["sigma_x"].append(sigmax)
        data_dict["sigma_y"].append(sigmay)
        data_dict["rotation"].append(result.params["phi"].value)
        data_dict["redchi"].append(result.redchi)

    # Make dataframe out of dictionary
    df_fl = pd.DataFrame(data_dict)

    # Also get summed counts for each frequency
    # Determine ROI that should be used:
    center_x = df_fl["center_x"].median()
    center_y = df_fl["center_y"].median()
    sigma_x = df_fl["sigma_x"].median()
    sigma_y = df_fl["sigma_y"].median()

    n_sigma = 2
    x0, y0 = int(center_x - n_sigma*sigma_x), int(center_y - n_sigma*sigma_y)
    x1, y1 = int(center_x + n_sigma*sigma_x), int(center_y + n_sigma*sigma_y)
    sum_ROI = np.array([x0,y0,x1,y1])

    data_dict["summed_counts"] = []
    # Loop over the frequncies and analyze data
    for i, freq  in enumerate(tqdm(frequencies)):
        
        # Extract data at this frequency
        data = df[(df[scan_param] == freq)]

        # Calculate mean integrated absorption for this frequency
        mean_integrated_absorption = np.mean(data['Integrated absorption'])

        # Sum up fluorescence counts and add to dictionary
        # print(sum_fluorescence_counts(data, mean_background, sum_ROI))
        data_dict["summed_counts"].append(sum_fluorescence_counts(data, mean_background, sum_ROI)/mean_integrated_absorption)

    df_fl = pd.DataFrame(data_dict)
    return df_fl

def analyze_frequency_scan_bootstrap(df : pd.DataFrame, mean_background : np.ndarray, scan_param: str, x0 = None, y0 = None, n = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes data for a frequency scan and gets errorbars using bootstrap
    """

    # Number of rounds to use for bootstrap
    n_bootstrap = n

    # Container for results
    df_results = pd.DataFrame()

    # The code below doesn't actually run in parallel, not sure why
    df_results = Parallel(n_jobs = -1, verbose = 1)(delayed(analyze_frequency_scan)(df.sample(replace = True, frac = 0.5), mean_background, scan_param, plots = False, center_x=x0, center_y = y0)
                             for i in range(n_bootstrap))
    df_results = pd.concat(df_results, ignore_index = True)

    # Aggregate results to get mean and errorbar
    # Define function that calculates errorbar
    def err(series):
        # Return MAD that is scaled to correspond to standard deviation
        return 1.4826*median_abs_deviation(series)
    
    df_agg = df_results.groupby(by = 'frequency').agg([np.median, err])
    new_columns = [col[0] +'_err' if col[1] == 'err' else col[0] for col in df_agg.columns ]
    df_agg.columns = new_columns

    return df_results, df_agg.reset_index()
