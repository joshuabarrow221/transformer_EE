"""Module for binning data and plotting histograms."""
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle  # Import the Circle class
from matplotlib.patches import Ellipse  # Import the Ellipse class

def plot_xstat(x_list, y_list, stat1="mean", stat2="rms", name="xbinstat", **kwargs):
    _stat_var = {
        "mean": np.mean,
        "std": np.std,
        "rms": lambda x: np.sqrt(np.mean(x**2)),
        "stderr": lambda x: np.std(x) / np.sqrt(len(x)),
    }

    _range = kwargs.get("range", (min([min(x) for x in x_list]), max([max(x) for x in x_list])))
    colors = kwargs.get("colors", ["orange"] * len(x_list))
    labels = kwargs.get("labels", [""] * len(x_list))
    legend_loc = kwargs.get("legend_loc", "topright")

    # Map legend_loc values to Matplotlib's loc values
    legend_positions = {
        "topright": "upper right",
        "topleft": "upper left",
        "bottomright": "lower right",
        "bottomleft": "lower left",
    }
    loc = legend_positions.get(legend_loc, "upper right")  # Default to top right if invalid loc is provided

    _fig, _ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)), dpi=kwargs.get("dpi", 80))

    for x, y, color, label in zip(x_list, y_list, colors, labels):
        bin_y, _bin_edges, _ = stats.binned_statistic(x, y, statistic=_stat_var[stat1], bins=kwargs.get("bins", 50), range=_range)
        bin_yerr, _bin_edges, _ = stats.binned_statistic(x, y, statistic=_stat_var[stat2], bins=kwargs.get("bins", 50), range=_range)
        bin_x = (_bin_edges[:-1] + _bin_edges[1:]) / 2

        _ax.errorbar(
            bin_x,
            bin_y,
            yerr=bin_yerr,
            xerr=(_bin_edges[1:] - _bin_edges[:-1]) / 2,
            fmt=kwargs.get("fmt", "none"),
            ecolor=color,
            label=f"{label}: {stat1.upper()} & {stat2.upper()}",
        )

    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", f"{stat1.upper()} with {stat2.upper()} Error Bars"), fontsize=16)
    _ax.hlines(0, _range[0], _range[1], linestyles="dashed", color="green")
    _ax.set_xlim(_range[0], _range[1])

    if kwargs.get('scale') == 'log':
        _ax.set_xscale('log')
        _ax.set_yscale('log')

    if kwargs.get("plotgrid", True):
        _ax.grid(visible=True, which="major", color="#666666", linestyle="--", alpha=0.8)
        _ax.minorticks_on()
        _ax.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

    _ax.legend(loc=loc)
    plt.savefig(
        os.path.join(
            kwargs.get("outdir", "."),
            name + "_" + stat1 + "_" + stat2 + "." + kwargs.get("ext", "png"),
        )
    )

    return {"bin_x": bin_x, "bin_y": bin_y, "bin_yerr": bin_yerr}



def plot_y_hist(*args, name="yhist", **kwargs):
    """
    Make a histogram of the given data sets with consistent binning.

    optional:
    parameter       default val
    ----------      -----------
    bins            500
    range           (min(x), max(x))
    ext             "png"
    xlabel          ""
    ylabel          ""
    title           ""
    figsize         (8, 6)
    dpi             80
    outdir          "."
    labels          []
    colors          []
    xlim            None
    ylim            None
    log             False
    vline           False
    normalize       False

    --------------------------
    """
    
    _range = kwargs.get("range", None)
    bins = kwargs.get("bins", 500)
    normalize = kwargs.get("normalize", False)

    # Determine global min and max from all datasets to ensure consistent binning
    global_min = min([np.min(x) for x in args])
    global_max = max([np.max(x) for x in args])

    # If a range is provided, use that, otherwise use the global min/max
    if _range is None:
        _range = (global_min, global_max)

    # Create bin edges using the global range and number of bins
    bin_edges = np.linspace(_range[0], _range[1], bins + 1)

    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 6)), dpi=kwargs.get("dpi", 80)
    )

    labels = kwargs.get("labels", [f"Hist {i+1}" for i in range(len(args))])
    colors = kwargs.get("colors", [None] * len(args))

    for i, x in enumerate(args):
        # If normalize is True, compute the area and use density=True for automatic normalization
        if normalize:
            _ax.hist(
                x,
                bins=bin_edges,  # Use the precomputed bin edges
                range=_range,
                weights=kwargs.get("weights", None),
                histtype="step",
                label=labels[i],
                color=colors[i],
                density=True  # This ensures the area under the histogram is 1
            )
        else:
            _ax.hist(
                x,
                bins=bin_edges,  # Use the precomputed bin edges
                range=_range,
                weights=kwargs.get("weights", None),
                histtype="step",
                label=labels[i],
                color=colors[i]
            )

    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", "Number of Events" if not normalize else "Arbitrary Units"), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=18)

    if labels:
        _ax.legend()

    if kwargs.get("log", False):
        _ax.set_yscale('log')

    if kwargs.get("vline", False):
        _ax.axvline(x=0.0, color='black', linestyle='--')

    if kwargs.get("xrange"):
        _ax.set_xlim(kwargs["xrange"])

    if kwargs.get("yrange"):
        _ax.set_ylim(kwargs["yrange"])

    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "png"))
    )

    plt.close(_fig)  # Close the figure after saving to free memory

    return None


def plot_2d_hist_count(x, y, name="hist2D", **kwargs):
    """
    Make a 2D histogram of x and y

    optional:
    parameter       default val
    ----------      -----------
    bins            50
    xbins           bins
    ybins           bins
    xrange          (0, 6)
    yrange          (0, 6)
    ext             "pdf"
    xlabel          ""
    ylabel          ""
    title           ""
    figsize         (8, 8)
    dpi             80
    outdir          "."

    --------------------------

    return: mu, sigma, rms, a
    """

    _bins = kwargs.get("bins", 500)
    _xbins = kwargs.get("xbins", _bins)
    _ybins = kwargs.get("ybins", _bins)
    _xrange = kwargs.get("xrange", (0, 6))
    _yrange = kwargs.get("yrange", (0, 6))

    H, xedges, yedges = np.histogram2d(
        x, y, bins=(_xbins, _ybins), range=(_xrange, _yrange)
    )

    H = H.T
    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 8)), dpi=kwargs.get("dpi", 80)
    )
    X, Y = np.meshgrid(xedges, yedges)

    # Apply logarithmic normalization if requested
    norm = LogNorm() if kwargs.get('zscale') == 'log' else None
    
    im = _ax.pcolormesh(X, Y, H, edgecolors="face", norm=norm)
    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=16)

    if(kwargs.get('scale')=='log'):
      _ax.set_xscale('log')
      _ax.set_yscale('log')

    cax = _fig.add_axes(
        [
            _ax.get_position().x1 + 0.01,
            _ax.get_position().y0,
            0.02,
            _ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)

    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "png"))
    )


##Experimental
def plot_2d_hist_contour(x, y, name="hist2D", **kwargs):
    """
    Make a 2D histogram of x and y with optional contours, diagonal line, circle, and labels.

    Parameters:
    - bins: Number of bins for the histogram (default: 500)
    - xbins, ybins: Number of bins in x and y directions for histogram (default: bins)
    - bins_contour: Number of bins for contours (default: 50)
    - xbins_contour, ybins_contour: Number of bins in x and y directions for contours (default: bins_contour)
    - xrange, yrange: Range for x and y axes (default: (0, 6))
    - zscale: Scale for the z-axis ('log' for logarithmic, default: None)
    - scale: Scale for x and y axes ('log' for logarithmic, default: None)
    - contours: Whether to draw contours (True/False, default: False)
    - histogram: Whether to draw the histogram (True/False, default: True)
    - diag_line: Whether to draw a diagonal line (True/False, default: False)
    - circle: Whether to draw a circle centered at (0,0) (True/False, default: False)
    - contour_labels: Whether to add labels to the contours (True/False, default: True)
    - ext: File extension for saving the plot (default: "pdf")
    - xlabel, ylabel, title: Labels and title for the plot (default: empty)
    - figsize, dpi: Figure size and resolution (default: (8, 8), 80)
    - outdir: Output directory for saving the plot (default: ".")
    """

    # Histogram binning
    _bins = kwargs.get("bins", 500)
    _xbins = kwargs.get("xbins", _bins)
    _ybins = kwargs.get("ybins", _bins)

    # Contour binning
    _bins_contour = kwargs.get("bins_contour", 50)
    _xbins_contour = kwargs.get("xbins_contour", _bins_contour)
    _ybins_contour = kwargs.get("ybins_contour", _bins_contour)

    _xrange = kwargs.get("xrange", (0, 6))
    _yrange = kwargs.get("yrange", (0, 6))

    # Create the 2D histogram for the contours
    H_contour, xedges_contour, yedges_contour = np.histogram2d(
        x, y, bins=(_xbins_contour, _ybins_contour), range=(_xrange, _yrange)
    )
    H_contour = H_contour.T
    X_contour, Y_contour = np.meshgrid(xedges_contour, yedges_contour)

    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 8)), dpi=kwargs.get("dpi", 80)
    )

    # Plot the histogram if requested
    if kwargs.get('histogram', True):
        H, xedges, yedges = np.histogram2d(
            x, y, bins=(_xbins, _ybins), range=(_xrange, _yrange)
        )
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)
        
        # Apply logarithmic normalization if requested
        norm = LogNorm() if kwargs.get('zscale') == 'log' else None

        im = _ax.pcolormesh(X, Y, H, edgecolors="face", norm=norm)
        cax = _fig.add_axes(
            [
                _ax.get_position().x1 + 0.01,
                _ax.get_position().y0,
                0.02,
                _ax.get_position().height,
            ]
        )
        plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax=cax)

    # Plot contours if requested
    if kwargs.get('contours', False):
        # Flatten and sort the histogram values in descending order
        H_sorted = np.sort(H_contour.ravel())[::-1]
        H_cumsum = np.cumsum(H_sorted)
        H_cumsum /= H_cumsum[-1]  # Normalize cumulative sum to get percentages

        # Define contour levels for 68%, 90%, and 95%
        levels = [0.95, 0.90, 0.68]
        contour_levels = [H_sorted[np.searchsorted(H_cumsum, level)] for level in levels]

        # Ensure contour levels are strictly increasing
        contour_levels = np.unique(contour_levels)
        contour_levels = np.sort(contour_levels)

        # Plot contours
        CS = _ax.contour(X_contour[:-1, :-1], Y_contour[:-1, :-1], H_contour, levels=contour_levels,
                    colors=['red', 'red', 'red'],
                    linestyles=['solid', 'dashed', 'dotted'])

        # Add labels to contours if requested
        if kwargs.get('contour_labels', True):
            fmt = {contour_levels[0]: '95%', contour_levels[1]: '90%', contour_levels[2]: '68%'}
            _ax.clabel(CS, CS.levels, inline=True, fontsize=15, fmt=fmt)

    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=16)

    # Plot a diagonal line if requested
    if kwargs.get('diag_line', False):
        _ax.plot([_xrange[0], _xrange[1]], [_yrange[0], _yrange[1]], color='darkorange', lw=2)

    # Plot a circle centered at (0,0) with a radius of 10 if requested
    if kwargs.get('circle', False):
        radius = kwargs.get('radius', 10)
        circle = Circle((0, 0), radius, color='darkorange', fill=False, lw=2)
        _ax.add_patch(circle)

    # Plot an ellipse centered at (0,0) with specified semimajor and semiminor axes if requested
    if kwargs.get('ellipse', False):
        x_semimajor = kwargs.get('x_semimajor', 10)
        y_semiminor = kwargs.get('y_semiminor', 5)
        ellipse = Ellipse((0, 0), width=2*x_semimajor, height=2*y_semiminor, color='darkorange', fill=False, lw=2)
        _ax.add_patch(ellipse)

    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "png"))
    )
    plt.show()



def plot_2d_hist_percentile_contour(x, y, name="hist2D_percentile", **kwargs):
    """
    Make a 2D histogram of x and y with contours that outline the region where 68% and 90% of the events
    lie along the y-axis for each x-bin.

    Parameters:
    - bins: Number of bins for the histogram (default: 500)
    - xbins, ybins: Number of bins in x and y directions for histogram (default: bins)
    - xbins_contour, ybins_contour: Number of bins in x and y directions for contours (default: bins_contour)
    - xrange, yrange: Range for x and y axes (default: (0, 6))
    - zscale: Scale for the z-axis ('log' for logarithmic, default: None)
    - scale: Scale for x and y axes ('log' for logarithmic, default: None)
    - contours: Whether to draw the percentile-based contours (True/False, default: True)
    - histogram: Whether to draw the 2D histogram (True/False, default: True)
    - contour_labels: Whether to add labels to the contours (True/False, default: True)
    - diag_line: Whether to draw a diagonal line (True/False, default: False)
    - circle: Whether to draw a circle centered at (0,0) (True/False, default: False)
    - ellipse: Whether to draw an ellipse centered at (0,0) (True/False, default: False)
    - ext: File extension for saving the plot (default: "pdf")
    - xlabel, ylabel, title: Labels and title for the plot (default: empty)
    - figsize, dpi: Figure size and resolution (default: (8, 8), 80)
    - outdir: Output directory for saving the plot (default: ".")
    """

    _bins = kwargs.get("bins", 500)
    _xbins = kwargs.get("xbins", _bins)
    _ybins = kwargs.get("ybins", _bins)

    # Contour binning
    _xbins_contour = kwargs.get("xbins_contour", _xbins)
    _ybins_contour = kwargs.get("ybins_contour", _ybins)

    _xrange = kwargs.get("xrange", (0, 6))
    _yrange = kwargs.get("yrange", (0, 6))

    # Bin the data along the x-axis for contours
    x_bin_edges = np.linspace(_xrange[0], _xrange[1], _xbins_contour + 1)
    x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2

    # Initialize lists to store the percentile-based ranges for contours
    y_68_low, y_68_high = [], []
    y_90_low, y_90_high = [], []

    # Loop over each x-bin and compute percentiles along the y-axis for contours
    for i in range(_xbins_contour):
        indices = (x >= x_bin_edges[i]) & (x < x_bin_edges[i + 1])
        y_in_bin = y[indices]

        if len(y_in_bin) > 0:
            # Calculate the percentiles for the y-values in this x-bin
            y_5th, y_32nd, y_68th, y_95th = np.percentile(y_in_bin, [5, 32, 68, 95])

            # Store the percentiles for contouring
            y_68_low.append(y_32nd)
            y_68_high.append(y_68th)
            y_90_low.append(y_5th)
            y_90_high.append(y_95th)
        else:
            # Handle empty bins
            y_68_low.append(np.nan)
            y_68_high.append(np.nan)
            y_90_low.append(np.nan)
            y_90_high.append(np.nan)

    # Create the plot
    _fig, _ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 8)), dpi=kwargs.get("dpi", 80))

    # Plot the histogram if requested
    if kwargs.get('histogram', True):
        H, xedges, yedges = np.histogram2d(x, y, bins=(_xbins, _ybins), range=(_xrange, _yrange))
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)

        # Apply logarithmic normalization if requested
        norm = LogNorm() if kwargs.get('zscale') == 'log' else None

        im = _ax.pcolormesh(X, Y, H, edgecolors="face", norm=norm)
        cax = _fig.add_axes(
            [
                _ax.get_position().x1 + 0.01,
                _ax.get_position().y0,
                0.02,
                _ax.get_position().height,
            ]
        )
        plt.colorbar(im, cax=cax)

    # Plot the percentile-based contours
    if kwargs.get('contours', True):
        # Plot 68% contour (solid for lower, dashed for upper)
        _ax.plot(x_bin_centers, y_68_low, 'r-', label='68% Contour (Lower)', lw=2)
        _ax.plot(x_bin_centers, y_68_high, 'r-', label='68% Contour (Upper)', lw=2)

        # Plot 90% contour (solid for lower, dashed for upper)
        _ax.plot(x_bin_centers, y_90_low, 'b-', label='90% Contour (Lower)', lw=2)
        _ax.plot(x_bin_centers, y_90_high, 'b-', label='90% Contour (Upper)', lw=2)

        # Add contour labels if requested
        if kwargs.get('contour_labels', True):
            # Manually place the labels at the center of the contour lines
            mid_bin = len(x_bin_centers) // 2
            _ax.text(x_bin_centers[mid_bin], (y_68_low[mid_bin] + y_68_high[mid_bin]) / 2.1, '68%', fontsize=10, color='r')
            _ax.text(x_bin_centers[mid_bin], (y_90_low[mid_bin] + y_90_high[mid_bin]) / 2.5, '90%', fontsize=10, color='b')

    # Additional options for diagonal line, circles, and ellipses
    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=16)

    if kwargs.get('diag_line', False):
        _ax.plot([_xrange[0], _xrange[1]], [_yrange[0], _yrange[1]], color='darkorange', lw=2)

    if kwargs.get('circle', False):
        radius = kwargs.get('radius', 10)
        circle = Circle((0, 0), radius, color='darkorange', fill=False, lw=2)
        _ax.add_patch(circle)

    if kwargs.get('ellipse', False):
        x_semimajor = kwargs.get('x_semimajor', 10)
        y_semiminor = kwargs.get('y_semiminor', 5)
        ellipse = Ellipse((0, 0), width=2*x_semimajor, height=2*y_semiminor, color='darkorange', fill=False, lw=2)
        _ax.add_patch(ellipse)

    # Save the figure
    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "png"))
    )
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import LogNorm

def plot_2d_hist_percentile_multi_contour(x_list, y_list, names, name="hist2D_percentile", **kwargs):
    """
    Make a 2D histogram of x and y with contours that outline the region where 90% of the events
    lie along the y-axis for each x-bin for multiple datasets.

    Parameters:
    - x_list, y_list: Lists of x and y datasets to plot contours for.
    - names: List of dataset names for the legend.
    - bins: Number of bins for the histogram (default: 500)
    - xbins_contour, ybins_contour: Number of bins in x and y directions for contours (default: bins_contour)
    - xrange, yrange: Range for x and y axes (default: (0, 6))
    - contour_colors: List of colors for each dataset's contours (default: auto-assigned)
    - line_widths: List of line widths for each dataset's contours (default: 2)
    - contour_labels: Whether to add labels to the contours (True/False, default: True)
    - diag_line: Whether to draw a diagonal line (True/False, default: False)
    - circle: Whether to draw a circle centered at (0,0) (True/False, default: False)
    - ellipse: Whether to draw an ellipse centered at (0,0) (True/False, default: False)
    - xlabel, ylabel, title: Labels and title for the plot (default: empty)
    - figsize, dpi: Figure size and resolution (default: (8, 8), 80)
    - outdir: Output directory for saving the plot (default: ".")
    """
    _xbins_contour = kwargs.get("xbins_contour", 100)
    _xrange = kwargs.get("xrange", (0, 6))
    _yrange = kwargs.get("yrange", (0, 6))

    # Colors and line widths for contours
    contour_colors = kwargs.get("contour_colors", plt.cm.viridis(np.linspace(0, 1, len(x_list))))
    line_widths = kwargs.get("line_widths", [2] * len(x_list))

    # Create the plot
    _fig, _ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 8)), dpi=kwargs.get("dpi", 80))

    # Bin the x-axis for contours
    x_bin_edges = np.linspace(_xrange[0], _xrange[1], _xbins_contour + 1)
    x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2

    # Loop over datasets in x_list and y_list
    for idx, (x, y) in enumerate(zip(x_list, y_list)):
        # Initialize lists to store percentile-based ranges for contours
        y_90_low, y_90_high = [], []

        # Loop over each x-bin and compute percentiles along the y-axis for contours
        for i in range(_xbins_contour):
            indices = (x >= x_bin_edges[i]) & (x < x_bin_edges[i + 1])
            y_in_bin = y[indices]

            if len(y_in_bin) > 0:
                # Calculate the percentiles for the y-values in this x-bin
                y_5th, y_95th = np.percentile(y_in_bin, [5, 95])

                # Store the percentiles for contouring
                y_90_low.append(y_5th)
                y_90_high.append(y_95th)
            else:
                # Handle empty bins
                y_90_low.append(np.nan)
                y_90_high.append(np.nan)

        # Plot 90% contours (solid for both lower and upper)
        _ax.plot(x_bin_centers, y_90_low, color=contour_colors[idx], lw=line_widths[idx], label=f'{names[idx]} 90% Contour (Lower)')
        _ax.plot(x_bin_centers, y_90_high, color=contour_colors[idx], lw=line_widths[idx], label=f'{names[idx]} 90% Contour (Upper)')

        # Add contour labels if requested
        if kwargs.get('contour_labels', True):
            mid_bin = len(x_bin_centers) // 2
            _ax.text(x_bin_centers[mid_bin], (y_90_low[mid_bin] + y_90_high[mid_bin]) / 2.5, f'{names[idx]} 90%', fontsize=10, color=contour_colors[idx])

    # Additional options for diagonal line, circle, and ellipse
    if kwargs.get('diag_line', False):
        _ax.plot([_xrange[0], _xrange[1]], [_yrange[0], _yrange[1]], color='darkorange', lw=2)

    if kwargs.get('circle', False):
        radius = kwargs.get('radius', 10)
        circle = Circle((0, 0), radius, color='darkorange', fill=False, lw=2)
        _ax.add_patch(circle)

    if kwargs.get('ellipse', False):
        x_semimajor = kwargs.get('x_semimajor', 10)
        y_semiminor = kwargs.get('y_semiminor', 5)
        ellipse = Ellipse((0, 0), width=2*x_semimajor, height=2*y_semiminor, color='darkorange', fill=False, lw=2)
        _ax.add_patch(ellipse)

    # Set axis labels and title
    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=16)

    # Add legend
    _ax.legend()

    # Save the figure
    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "png"))
    )
    plt.show()

