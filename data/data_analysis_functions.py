# Import packages:
import h5py
import copy
import lmfit
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from plot_utils import set_fontsize
from hdf_utils import load_measurement_data_devices_attrs
import lmfit
import pandas as pd


def gain_analysis(data_path, run_names, scan_parameter, ptn_cutoff = 5000,
                  yag_cutoff = 250, abs_cutoff = 5, fl_cutoff = -1e6):
    """
    Function that calculates the gain ratios for a given data run.

    inputs:
    data_path            : Path to file that contains data in hdf5
    run_name             : name of run to analyze
    scan_parameter       : parameter that was scanned during the analysis

    ouputs:
    results_df           : dataframe with collected results 

    """
    ### 1. Importing data

    # Get data for all channels of PXIe
    run_name = run_names[0]
    devices = ()
    pxie, pxie_time, pxie_attrs, data_devices = load_measurement_data_devices_attrs(data_path, run_name, devices)

    # Define which channel on PXIe corresponds to which data:
    yag_channel = 0
    pmt_channel = 1
    pd_channel = 2
    pdn_channel = 3
    shutter_channel = 4
    pdrc_channel = 5

    # Separate data for each channel to correct array
    data_dict = {}
    data_dict["YAG PD"] = [pxie[idx][:,yag_channel] for idx in range(1,len(pxie)+1)]
    data_dict["PMT output"] = [-pxie[idx][:,pmt_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Absorption PD"] = [pxie[idx][:,pd_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Abs Norm PD"] =[pxie[idx][:,pdn_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Shutter"] =[pxie[idx][:,shutter_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["RC Norm PD"] =[pxie[idx][:,pdrc_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict[scan_parameter] =[float(pxie_attrs[idx][scan_parameter]) for idx in range(1,len(pxie)+1)]

    # If we have more than one run, get the data for those as well
    for run_name in run_names[1:]:
        pxie, pxie_time, pxie_attrs, data_devices = load_measurement_data_devices_attrs(data_path, run_name, devices)
        # Separate data for each channel to correct array
        data_dict["YAG PD"] += [pxie[idx][:,yag_channel] for idx in range(1,len(pxie)+1)]
        data_dict["PMT output"] += [-pxie[idx][:,pmt_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Absorption PD"] += [pxie[idx][:,pd_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Abs Norm PD"] +=[pxie[idx][:,pdn_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Shutter"] +=[pxie[idx][:,shutter_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["RC Norm PD"] +=[pxie[idx][:,pdrc_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict[scan_parameter] +=[float(pxie_attrs[idx][scan_parameter]) for idx in range(1,len(pxie)+1)]

    ### 2. Generate dataframe
    #Generating a pandas dataframe to contain the data and calculating things based on the data
    dataframe = pd.DataFrame(data_dict)

    #Define slices used for integrals
    slice_flm  = np.s_[-3000:] #Slice for determining fluoresence background
    slice_absm = np.s_[-3000:] #Slice for determining absorption background
    slice_fli  = np.s_[150:] # Slice for calculating fluorescence integral
    slice_absi = np.s_[10:2000] #Slice for calculating absorption integral

    #Generate a column for normalized absorption (laser power after molecules divided by before)
    dataframe["Normalized Absorption"] = dataframe["Absorption PD"]/dataframe["Abs Norm PD"]

    #Calculate integrated absorption signal from the normalized absorption
    def calculate_integrated_absorption(trace):
        return -np.trapz(trace[slice_absi] - np.mean(trace[slice_absm]))

    dataframe["Integrated absorption"] = dataframe["Normalized Absorption"].apply(calculate_integrated_absorption)

    #Calculate integrated fluorescence signals from PMT output
    def calculate_integrated_fluorescence(trace):
        return np.trapz(trace[slice_fli] - np.mean(trace[slice_flm]))

    dataframe["Integrated fluorescence"] = dataframe["PMT output"].apply(calculate_integrated_fluorescence)

    #Calculate fluorescence normalized by absorption
    dataframe["Normalized fluorescence"] = dataframe["Integrated fluorescence"]/dataframe["Integrated absorption"]

    ### 3. Cuts
    # Applying various cuts to data based on if lasers were on and if signal size is large enough

    # Add additional columns for cuts
    # Check if YAG fired
    dataframe["YAG fired"] = dataframe["YAG PD"].apply(np.max) > yag_cutoff

    # Check if absorption laser was on
    dataframe["Absorption ON"] = dataframe["Abs Norm PD"].apply(np.min) > ptn_cutoff

    # Check if rotational cooling was open
    dataframe["RC ON"] = dataframe["RC Norm PD"].apply(np.min) > 500

    # Check if shutter was on
    dataframe["Shutter OPEN"] = dataframe["Shutter"].apply(np.max) > 10000

    # Check if absorption was large enough
    dataframe["Abs cut"] = dataframe["Integrated absorption"] > abs_cutoff

    # Check if fluorescence was large enough
    dataframe["Fl cut"] = dataframe["Integrated fluorescence"] > fl_cutoff

    # Generate masks for use with data analysis
    mask_RC_on = (dataframe["YAG fired"] & dataframe["RC ON"] & dataframe["Absorption ON"] 
              & dataframe["Shutter OPEN"] & dataframe["Abs cut"] & dataframe["Fl cut"])

    mask_RC_off = (dataframe["YAG fired"] & dataframe["Absorption ON"]  & ~dataframe["RC ON"] 
                & ~dataframe["Shutter OPEN"] & dataframe["Abs cut"] & dataframe["Fl cut"])

    # Calculate how much of the original data is being used
    percentage_data_used = (mask_RC_on.sum()+mask_RC_off.sum())/dataframe.index.size
    print(f"Percentage of traces used in final results: {percentage_data_used*100:.1f} %")

    
    ### 4. Calculate gain
    # Determine the unique frequencies
    frequencies = np.unique(dataframe[scan_parameter])

    # Containers for final results
    means_on = []
    errs_on = []
    means_off = []
    errs_off = []

    # Loop over frequencies and plot histogram for each
    for f in frequencies:
        series_RC_off = dataframe["Normalized fluorescence"][mask_RC_off & (dataframe[scan_parameter] == f)]

        series_RC_on = dataframe["Normalized fluorescence"][mask_RC_on & (dataframe[scan_parameter] == f)]
        
        #Store results
        means_on.append(series_RC_on.mean())
        errs_on.append(series_RC_on.std()/np.sqrt(len(series_RC_on)-1))
        
        means_off.append(series_RC_off.mean())
        errs_off.append(series_RC_off.std()/np.sqrt(len(series_RC_off)-1))


    # Define a new dataframe with the results
    results_dict = {"Frequency": frequencies,
                    "Mean RC ON": means_on, 
                    "Error RC ON": errs_on, 
                    "Mean RC OFF": means_off, 
                    "Error RC OFF": errs_off}
    results_df = pd.DataFrame(results_dict)

    # Calculate ratios and errors in ratios for each frequency
    results_df["Ratio"] = results_df["Mean RC ON"]/results_df["Mean RC OFF"]
    results_df["Ratio error"] = np.sqrt((results_df["Error RC ON"]/results_df["Mean RC ON"])**2 
                                        + (results_df["Error RC OFF"]/results_df["Mean RC OFF"])**2)*results_df["Ratio"]


    # Return the results dataframe
    return results_df

from scipy.interpolate import interp1d
from scipy.optimize import fsolve
def gain_width_accumulation(frequencies, gains, baseline = 1, threshold = 0.5, x0 = [-1,1]):
    """
    Function that calculates the width of a peak in gain by determining where the gain falls
    below a threshold amount
    """
    # Calculate gain compared to baseline
    gains = gains - baseline

    # Generate an interpolation function based on the given data
    interp_func = interp1d(frequencies, gains, bounds_error=False, fill_value = 0)

    # Find the maximum of the gain function
    index_max = np.argmax(gains)
    max_gain = gains[index_max]
    freq_max = frequencies[index_max]

    # Find the frequencies where the gain is at threshold
    roots = fsolve((lambda x: interp_func(x) - max_gain*threshold), [freq_max + x0[0],freq_max + x0[1]])

    # Return the difference between the roots
    if len(roots) == 2:
        return np.abs(roots[0] - roots[1])

    #If something went wrong, raise an error
    elif len(roots) != 2:
        raise ValueError(f"found {len(roots)} roots")


def gain_width_depletion(frequencies, gains, baseline = 1, threshold = 0.5, x0 = [-1,1]):
    """
    Function that calculates the width of a peak in gain by determining where the gain falls
    below a threshold amount for depletion
    """
    # Calculate gain compared to baseline
    gains = baseline - gains

    # Generate an interpolation function based on the given data
    interp_func = interp1d(frequencies, gains, bounds_error=False, fill_value = 0)

    # Find the maximum of the gain function
    index_max = np.argmax(gains)
    max_gain = gains[index_max]
    freq_max = frequencies[index_max]

    # Find the frequencies where the gain is at threshold
    roots = fsolve((lambda x: interp_func(x) - max_gain*threshold), [freq_max + x0[0],freq_max + x0[1]])

    # Return the difference between the roots
    if len(roots) == 2:
        return np.abs(roots[0] - roots[1])

    #If something went wrong, raise an error
    elif len(roots) != 2:
        raise ValueError(f"found {len(roots)} roots")


def gain_from_datadict(data_dict, scan_parameter, ptn_cutoff = 5000,
                        yag_cutoff = 250, abs_cutoff = 5, fl_cutoff = 0):
    """
    Function that calculates the gain ratios for a given data dictionary.

    inputs:
    data_path            : Path to file that contains data in hdf5
    run_name             : name of run to analyze
    scan_parameter       : parameter that was scanned during the analysis

    ouputs:
    results_df           : dataframe with collected results 

    """

    ### 1. Generate dataframe
    #Generating a pandas dataframe to contain the data and calculating things based on the data
    dataframe = pd.DataFrame(data_dict)

    #Define slices used for integrals
    slice_flm  = np.s_[-3000:] #Slice for determining fluoresence background
    slice_absm = np.s_[-3000:] #Slice for determining absorption background
    slice_fli  = np.s_[150:] # Slice for calculating fluorescence integral
    slice_absi = np.s_[10:2000] #Slice for calculating absorption integral

    #Generate a column for normalized absorption (laser power after molecules divided by before)
    dataframe["Normalized Absorption"] = dataframe["Absorption PD"]/dataframe["Abs Norm PD"]

    #Calculate integrated absorption signal from the normalized absorption
    def calculate_integrated_absorption(trace):
        return -np.trapz(trace[slice_absi] - np.mean(trace[slice_absm]))

    dataframe["Integrated absorption"] = dataframe["Normalized Absorption"].apply(calculate_integrated_absorption)

    #Calculate integrated fluorescence signals from PMT output
    def calculate_integrated_fluorescence(trace):
        return np.trapz(trace[slice_fli] - np.mean(trace[slice_flm]))

    dataframe["Integrated fluorescence"] = dataframe["PMT output"].apply(calculate_integrated_fluorescence)

    #Calculate fluorescence normalized by absorption
    dataframe["Normalized fluorescence"] = dataframe["Integrated fluorescence"]/dataframe["Integrated absorption"]

    ### 2. Cuts
    # Applying various cuts to data based on if lasers were on and if signal size is large enough

    # Add additional columns for cuts
    # Check if YAG fired
    dataframe["YAG fired"] = dataframe["YAG PD"].apply(np.max) > yag_cutoff

    # Check if absorption laser was on
    dataframe["Absorption ON"] = dataframe["Abs Norm PD"].apply(np.min) > ptn_cutoff

    # Check if rotational cooling was open
    dataframe["RC ON"] = dataframe["RC Norm PD"].apply(np.min) > 1500

    # Check if shutter was on
    dataframe["Shutter OPEN"] = dataframe["Shutter"].apply(np.max) > 10000

    # Check if absorption was large enough
    dataframe["Abs cut"] = dataframe["Integrated absorption"] > abs_cutoff

    # Check if fluorescence was large enough
    dataframe["Fl cut"] = dataframe["Integrated fluorescence"] > fl_cutoff

    # Generate masks for use with data analysis
    mask_RC_on = (dataframe["YAG fired"] & dataframe["RC ON"] & dataframe["Absorption ON"] 
              & dataframe["Shutter OPEN"] & dataframe["Abs cut"] & dataframe["Fl cut"])

    mask_RC_off = (dataframe["YAG fired"] & dataframe["Absorption ON"]  & ~dataframe["RC ON"] 
                & ~dataframe["Shutter OPEN"] & dataframe["Abs cut"] & dataframe["Fl cut"])

    # Calculate how much of the original data is being used
    percentage_data_used = (mask_RC_on.sum()+mask_RC_off.sum())/dataframe.index.size
    print(f"Percentage of traces used in final results: {percentage_data_used*100:.1f} %")

    
    ### 3. Calculate gain
    # Determine the unique frequencies
    frequencies = np.unique(dataframe[scan_parameter])

    # Containers for final results
    means_on = []
    errs_on = []
    means_off = []
    errs_off = []
    means_fl_on = []
    errs_fl_on = []
    means_fl_off = []
    errs_fl_off = []


    # Loop over frequencies and plot histogram for each
    for f in frequencies:
        series_RC_off = dataframe["Normalized fluorescence"][mask_RC_off & (dataframe[scan_parameter] == f)]
        series_RC_on = dataframe["Normalized fluorescence"][mask_RC_on & (dataframe[scan_parameter] == f)]

        series_fl_RC_off = dataframe["Integrated fluorescence"][mask_RC_off & (dataframe[scan_parameter] == f)]
        series_fl_RC_on = dataframe["Integrated fluorescence"][mask_RC_on & (dataframe[scan_parameter] == f)]
        
        #Store results
        means_on.append(series_RC_on.mean())
        errs_on.append(series_RC_on.std()/np.sqrt(len(series_RC_on)-1))
        
        means_off.append(series_RC_off.mean())
        errs_off.append(series_RC_off.std()/np.sqrt(len(series_RC_off)-1))

        means_fl_on.append(series_fl_RC_on.mean())
        errs_fl_on.append(series_fl_RC_on.std()/np.sqrt(len(series_fl_RC_on)-1))
        
        means_fl_off.append(series_fl_RC_off.mean())
        errs_fl_off.append(series_fl_RC_off.std()/np.sqrt(len(series_fl_RC_off)-1))


    # Define a new dataframe with the results
    results_dict = {"Frequency": frequencies,
                    "Mean RC ON": means_on, 
                    "Error RC ON": errs_on, 
                    "Mean RC OFF": means_off, 
                    "Error RC OFF": errs_off,
                    "Mean Fl RC ON": means_fl_on, 
                    "Error Fl RC ON": errs_fl_on, 
                    "Mean Fl RC OFF": means_fl_off, 
                    "Error Fl RC OFF": errs_fl_off}
    results_df = pd.DataFrame(results_dict)

    # Calculate ratios and errors in ratios for each frequency
    results_df["Ratio"] = results_df["Mean RC ON"]/results_df["Mean RC OFF"]
    results_df["Ratio error"] = np.sqrt((results_df["Error RC ON"]/results_df["Mean RC ON"])**2 
                                        + (results_df["Error RC OFF"]/results_df["Mean RC OFF"])**2)*results_df["Ratio"]

    # Return the results dataframe
    return results_df


def RC_analysis(data_path, run_index, scan_parameter, ptn_cutoff = 5000,
                  yag_cutoff = 250, abs_cutoff = 6, fl_cutoff = -1e6):
    """
    Function that calculates the gain ratios for a given data run.

    inputs:
    data_path            : Path to file that contains data in hdf5
    run_name             : name of run to analyze
    scan_parameter       : parameter that was scanned during the analysis

    ouputs:
    results_df           : dataframe with collected results 

    """
    ### 1. Importing data

    # Get data for all channels of PXIe
    with h5py.File(data_path, 'r') as f:
        dset_names = list(f.keys())
    run_names = [dset_names[run_index]]
    run_name = run_names[0]
    devices = ()
    pxie, pxie_time, pxie_attrs, data_devices = load_measurement_data_devices_attrs(data_path, run_name, devices)

    # Define which channel on PXIe corresponds to which data:
    yag_channel = 0
    pmt_channel = 1
    pd_channel = 2
    pdn_channel = 3
    shutter_channel = 4
    pdrc_channel = 5

    # Separate data for each channel to correct array
    data_dict = {}
    data_dict["YAG PD"] = [pxie[idx][:,yag_channel] for idx in range(1,len(pxie)+1)]
    data_dict["PMT output"] = [pxie[idx][:,pmt_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Absorption PD"] = [pxie[idx][:,pd_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Abs Norm PD"] =[pxie[idx][:,pdn_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["Shutter"] =[pxie[idx][:,shutter_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict["RC Norm PD"] =[pxie[idx][:,pdrc_channel].astype(float) for idx in range(1,len(pxie)+1)]
    data_dict[scan_parameter] =[float(pxie_attrs[idx][scan_parameter]) for idx in range(1,len(pxie)+1)]

    # If we have more than one run, get the data for those as well
    for run_name in run_names[1:]:
        pxie, pxie_time, pxie_attrs, data_devices = load_measurement_data_devices_attrs(data_path, run_name, devices)
        # Separate data for each channel to correct array
        data_dict["YAG PD"] += [pxie[idx][:,yag_channel] for idx in range(1,len(pxie)+1)]
        data_dict["PMT output"] += [-pxie[idx][:,pmt_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Absorption PD"] += [pxie[idx][:,pd_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Abs Norm PD"] +=[pxie[idx][:,pdn_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["Shutter"] +=[pxie[idx][:,shutter_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict["RC Norm PD"] +=[pxie[idx][:,pdrc_channel].astype(float) for idx in range(1,len(pxie)+1)]
        data_dict[scan_parameter] +=[float(pxie_attrs[idx][scan_parameter]) for idx in range(1,len(pxie)+1)]

    ### 2. Generate dataframe
    #Generating a pandas dataframe to contain the data and calculating things based on the data
    dataframe = pd.DataFrame(data_dict)

    #Define slices used for integrals
    slice_flm  = np.s_[-3000:] #Slice for determining fluoresence background
    slice_absm = np.s_[-3000:] #Slice for determining absorption background
    slice_fli  = np.s_[150:] # Slice for calculating fluorescence integral
    slice_absi = np.s_[10:2000] #Slice for calculating absorption integral

    #Generate a column that labels datapoints by ablation spot
    dataframe["Ablation spot"] = np.array(dataframe.index/10, dtype = int)

    #Generate a column for normalized absorption
    dataframe["Normalized Absorption"] = dataframe["Absorption PD"]/dataframe["Abs Norm PD"]
    dataframe["Normalized Absorption"] = (dataframe["Normalized Absorption"]
                                        /(dataframe["Normalized Absorption"].apply(lambda x: x[slice_absm].mean())))

    #Calculate integrated absorption signal from the normalized absorption
    def calculate_integrated_absorption(trace):
        return -np.trapz(trace[slice_absi] - np.mean(trace[slice_absm]))

    dataframe["Integrated absorption"] = dataframe["Normalized Absorption"].apply(calculate_integrated_absorption)

    #Calculate integrated fluorescence signals from PMT output
    def calculate_integrated_fluorescence(trace):
        return -np.trapz(trace[slice_fli] - np.mean(trace[slice_flm]))

    dataframe["Integrated fluorescence"] = dataframe["PMT output"].apply(calculate_integrated_absorption)

    #Calculate fluorescence normalized by absorption
    dataframe["Normalized fluorescence"] = dataframe["Integrated fluorescence"]/dataframe["Integrated absorption"]
    ### 3. Cuts
    # Applying various cuts to data based on if lasers were on and if signal size is large enough

    # Add additional columns for cuts
    # Check if YAG fired
    dataframe["YAG fired"] = dataframe["YAG PD"].apply(np.max) > yag_cutoff

    # Check if absorption laser was on
    dataframe["Absorption ON"] = dataframe["Abs Norm PD"].apply(np.min) > ptn_cutoff

    #Check if rotational cooling was on
    dataframe["RC ON"] = dataframe["RC Norm PD"].apply(np.min) > 1500

    #Check if shutter was on
    dataframe["Shutter ON"] = dataframe["Shutter"].apply(np.max) > 10000

    #Based on the plot, generate a cut for cases where there may not be any actual molecules
    dataframe["Abs cut"] = dataframe["Integrated absorption"] > abs_cutoff

    #Add a cut for normalized integrated fluorescence 
    dataframe["Fl cut"] = dataframe["Integrated fluorescence"] > fl_cutoff

    # Generate masks for use with data analysis
    mask = (dataframe["YAG fired"] & dataframe["Absorption ON"] & dataframe["Abs cut"] & dataframe["Fl cut"])

    # Calculate how much of the original data is being used
    percentage_data_used = (mask.sum())/len(dataframe)
    print(f"Percentage of traces used in final results: {percentage_data_used*100:.1f} %")

    ### 4. Calculate gain
    #Determine the unique frequencies
    frequencies = np.unique(dataframe[scan_parameter])

    #Containers for final results
    means = []
    errs = []

    #Loop over frequencies and plot histogram for each
    for f in frequencies:
        # Get the normalized fluorescence for this frequency
        dataseries = dataframe["Normalized fluorescence"][mask & (dataframe[scan_parameter] == f)]
        
        #Store results
        means.append(dataseries.mean())
        errs.append(dataseries.std()/np.sqrt(len(dataseries)-1))


    #Define a new dataframe with the results
    results_dict = {"Frequency": frequencies,
                    "Mean": means, 
                    "Error": errs
                }
    results_df = pd.DataFrame(results_dict)

    # Return the results dataframe
    return results_df