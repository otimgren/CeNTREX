import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))

def set_fontsize(ax, fs):
    ax.title.set_fontsize(fs*1.3)
    ax.xaxis.label.set_fontsize(fs)
    ax.yaxis.label.set_fontsize(fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs/1.25)

def plot_binned_datasets_laserscan(binned_data_dsets, descriptions):
    # getting the zorders from the max_heights
    max_heights = np.array([np.nanmax(list(values.values())) for values in binned_data_dsets.values()])
    zorders = {p:zorder for p, zorder in zip(descriptions, [np.where(idx == np.argsort(-max_heights))[0][0] for idx in range(len(max_heights))])}
    color_cycle = {p:plt.get_cmap('tab10')(len(descriptions)-1 - i) for p,i in zorders.items()}


    fig_dot, ax_dot = plt.subplots(figsize = (10,8))
    fig_bar, ax_bar = plt.subplots(figsize = (10,8))

    for desc, binned_integrals_averaged in binned_data_dsets.items():
        bins = binned_integrals_averaged.keys()

        y = [binned_integrals_averaged[b][0] for b in bins]
        yerr = [binned_integrals_averaged[b][1] for b in bins]
        ax_dot.errorbar(bins, y, yerr, fmt = '.', ms = 10, label = f'{desc}')

        ax_bar.bar(bins, y, yerr = yerr, alpha = 1, width = 1, label = f'{desc}',
                   edgecolor = 'k', zorder = zorders[desc], color = color_cycle[desc])

    ax_dot.legend(fontsize = 15)
    ax_dot.set_xlabel('[MHz]')
    ax_dot.set_ylabel('integral [adc]')
    set_fontsize(ax_dot, 15)

    ax_bar.legend(fontsize = 15)
    ax_bar.set_xlabel('[MHz]')
    ax_bar.set_ylabel('integral [adc]')
    set_fontsize(ax_bar, 15)

    return {'scan errorbar': (fig_dot, ax_dot), 'scan histogram': (fig_bar, ax_bar)}

def plot_binned_dataset_switching(bin_state_data, ratio, title, switch_labels):
    t = np.arange(0,20,1/(1e2))

    figures = {}

    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(t, np.mean(bin_state_data[1], axis = 0), label = f'{switch_labels[1]}')
    ax.plot(t, np.mean(bin_state_data[0], axis = 0), label = f'{switch_labels[0]}')

    ax.set_title(title)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('ADC [arb.]')
    ax.legend(fontsize = 15)
    set_fontsize(ax, 15)

    figures['trace mean'] = (fig, ax)

    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(t[500:1500], (np.mean(bin_state_data[1], axis = 0)/np.mean(bin_state_data[0], axis = 0))[500:1500])

    ax.set_title(title)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('ratio')
    set_fontsize(ax, 15)

    figures['trace ratio'] = (fig, ax)

    # excluding ratios larger than 10, clearly some noisy data there
    ratio = np.array(ratio)
    ratio = ratio[(ratio < 10) & (ratio > 0)]

    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(ratio, lw = 2)

    textstr = f"ratio = {np.mean(ratio):.2f} $\pm$ {np.std(ratio)/np.sqrt(ratio.size):.2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=props)

    ax.set_title(title)
    ax.set_xlabel('shot')
    ax.set_ylabel('signal ratio')
    set_fontsize(ax, 15)

    figures['ratios'] = (fig, ax)

    bins = np.arange(ratio.min(), ratio.max()+1/8, 1/8)
    fig, ax = plt.subplots(figsize = (8,6))
    n, bins, _ = ax.hist(ratio, bins = bins, density = True, histtype='bar')
    popt, pcov = curve_fit(gaussian, bins[:-1]+(bins[1]-bins[0])/2, n, p0 = (np.mean(ratio), np.std(ratio)))

    ax.plot(np.linspace(bins.min(), bins.max(), 101), gaussian(np.linspace(bins.min(), bins.max(), 101), *popt),
            lw = 3)
    ax.set_xlabel('ratio')
    set_fontsize(ax, 15)
    textstr = f"$\mu$ = {popt[0]:.2f},  $\sigma$ = {popt[1]:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=props);

    figures['histogram'] = (fig, ax)

    return figures
