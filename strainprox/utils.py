import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
from pathlib import Path
import cupy as cp

def callbackx(x: np.ndarray, ui: np.ndarray, xtrue: np.ndarray, 
              xhist: List[np.ndarray], xsnr: List[float], xerr: List[float]) -> None:
    """
    Callback function for tracking a single variable in optimization.
    
    Parameters
    ----------
    x : np.ndarray
        Current variable update
    ui : np.ndarray
        Current estimate of the model
    xtrue : np.ndarray
        True/reference model for comparison
    xhist : List[np.ndarray]
        History of model updates
    xsnr : List[float]
        History of SNR values
    xerr : List[float]
        History of RRE values
    """
    x += ui 
    xhist.append(x)
    xsnr.append(SNR(xtrue, x))
    xerr.append(RRE(xtrue, x))
    

def callbackxy(x: np.ndarray, y: np.ndarray, xtrue: np.ndarray, 
               xhist: List[np.ndarray], yhist: List[np.ndarray], 
               xsnr: List[float], ysnr: List[float], 
               xerr: List[float], yerr: List[float]) -> None:
    """
    Callback function for tracking two variables in optimization.
    
    Parameters
    ----------
    x : np.ndarray
        First variable update
    y : np.ndarray
        Second variable update
    xtrue : np.ndarray
        True/reference model for comparison
    xhist : List[np.ndarray]
        History of first variable updates
    yhist : List[np.ndarray]
        History of second variable updates
    xsnr : List[float]
        History of SNR values for first variable
    ysnr : List[float]
        History of SNR values for second variable
    xerr : List[float]
        History of RRE values for first variable
    yerr : List[float]
        History of RRE values for second variable
    """
    xhist.append(x)
    yhist.append(y)
    xsnr.append(SNR(xtrue, x))
    ysnr.append(SNR(xtrue, y))
    xerr.append(RRE(xtrue, x))
    yerr.append(RRE(xtrue, y))    


def RRE(x: np.ndarray, xinv: np.ndarray) -> float:
    """
    Calculate the Relative Reconstruction Error.
    
    Parameters
    ----------
    x : np.ndarray
        True/reference model
    xinv : np.ndarray
        Reconstructed/estimated model
        
    Returns
    -------
    float
        Relative Reconstruction Error
    """
    return np.linalg.norm(x - xinv) / np.linalg.norm(x)


def SNR(xref: np.ndarray, xest: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio in decibels.
    
    Parameters
    ----------
    xref : np.ndarray
        Reference signal
    xest : np.ndarray
        Estimated signal
        
    Returns
    -------
    float
        Signal-to-Noise Ratio in dB
    """
    xrefv = np.mean(np.abs(xref) ** 2)
    return 10. * np.log10(xrefv / np.mean(np.abs(xref - xest)**2))


def apply_time_shift(d: np.ndarray,
                     time_shift: np.ndarray,
                     t: np.ndarray,
                     dims: Tuple[int, ...]) -> np.ndarray:
    """Apply time shift to a given 2D or 3D seismic dataset.

    Parameters
    ----------
    d : np.ndarray
        2D or 3D data to be shifted (e.g., monitor data).
    time_shift : np.ndarray
        2D or 3D time shift field.
    t : np.ndarray
        1D array of time values, corresponding to the first axis of `d`.
    dims : Tuple[int, ...]
        The dimensions of the data, e.g., (nt, nx) or (nt, nx, ny).

    Returns
    -------
    np.ndarray
        Shifted data array.
    """
    # Reshape arrays if model is 3D
    if len(dims) == 3:
        d = d.reshape(dims[0], -1)
        dims = (dims[0], dims[1]*dims[2])

    d_shifted = np.zeros_like(d)

    # Create time-shift interpolators for each trace
    shifted_time_grid = t.copy()[:, np.newaxis] - time_shift.reshape(dims)

    # Apply interpolation for each trace
    for col in range(d.shape[1]):
        interpolator = interp1d(
            shifted_time_grid[:, col],
            d[:, col],
            kind='cubic',
            fill_value="extrapolate"
        )
        d_shifted[:, col] = interpolator(t)

    return d_shifted

def apply_time_shift_cupy(d: cp.ndarray,
                          time_shift: cp.ndarray,
                          t: cp.ndarray,
                          dims: Tuple[int, ...]) -> cp.ndarray:
    """Apply time shift to a given 2D or 3D seismic dataset.

    Parameters
    ----------
    d : cp.ndarray
        2D or 3D data to be shifted (e.g., monitor data).
    time_shift : cp.ndarray
        2D or 3D time shift field.
    t : cp.ndarray
        1D array of time values, corresponding to the first axis of `d`.
    dims : Tuple[int, ...]
        The dimensions of the data, e.g., (nt, nx) or (nt, nx, ny).

    Returns
    -------
    cp.ndarray
        Shifted data array.
    """
    # Reshape arrays if model is 3D
    if len(dims) == 3:
        d = d.reshape(dims[0], -1)
        dims = (dims[0], dims[1] * dims[2])

    
    d_shifted = cp.zeros_like(d)

    # Create the grid of shifted time points for each trace
    shifted_time_grid = t.copy()[:, cp.newaxis] - time_shift.reshape(dims)

    for col in range(d.shape[1]):
        d_shifted[:, col] = cp.interp(t, shifted_time_grid[:, col], d[:, col])

    return d_shifted

def plotter_4D(b: np.ndarray, m: np.ndarray, dt: float = 1.0, 
               type: str = 'impedance', perc: float = 1.0, 
               dif_scale: float = 0.01, ref: Optional[np.ndarray] = None,
               height: float = 4.0, width: float = 15.0, 
               mtrue: Optional[np.ndarray] = None, 
               cmap: str = 'seismic_r', vline: Optional[int] = None, 
               ztitle: str = '') -> None:
    """
    Create a 4D visualization comparing baseline and monitor data.
    
    Parameters
    ----------
    b : np.ndarray
        Baseline data
    m : np.ndarray
        Monitor data
    dt : float, optional
        Time sampling interval, by default 1.0
    type : str, optional
        Type of data ('seismic' or 'impedance'), by default 'impedance'
    perc : float, optional
        Percentile for colormap clipping, by default 1.0
    dif_scale : float, optional
        Scale for difference plot, by default 0.01
    ref : np.ndarray, optional
        Reference data for colormap scaling, by default None
    height : float, optional
        Figure height in inches, by default 4.0
    width : float, optional
        Figure width in inches, by default 15.0
    mtrue : np.ndarray, optional
        True model for computing metrics, by default None
    cmap : str, optional
        Colormap for seismic data, by default 'seismic_r'
    vline : int, optional
        Position for vertical line, by default None
    ztitle : str, optional
        Z-axis title, by default ''
    """
    if ref is not None:
        vmin, vmax = np.percentile(ref, [perc, 100 - perc])
    else:
        vmin, vmax = np.percentile(b, [perc, 100 - perc])

    fig = plt.figure(figsize=(width, height))
    
    if type == 'seismic':
        gs = gridspec.GridSpec(5, 4, width_ratios=(1, 1, 1, .05), height_ratios=(.1, .5, .5, .5, .1),
                               left=0.1, right=0.9, bottom=0.1, top=0.9,
                               wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[:, 0])
        base = ax0.imshow(b, vmin=-vmax, vmax=vmax, cmap=cmap, extent=[0, m.shape[1], m.shape[0] * dt, 0])
        ax0.set_ylabel(ztitle)
        ax0.set_title('a) Baseline Seismic')
        ax0.axis('tight')
        
        ax1 = fig.add_subplot(gs[:, 1])
        ax1.imshow(m, vmin=-vmax, vmax=vmax, cmap=cmap)
        ax1.set_yticklabels([])
        ax1.set_title('b) Monitor Seismic')
        ax1.axis('tight')
        
        ax2 = fig.add_subplot(gs[:, 2])
        ax2.imshow(m - b, vmin=-vmax, vmax=vmax, cmap=cmap)
        ax2.set_yticklabels([])
        ax2.set_title('c) Monitor - Baseline')
        ax2.axis('tight')
        
        ax3 = fig.add_subplot(gs[2, 3])
        ax3.set_title('Amplitude', loc='left')
        Colorbar(ax=ax3, mappable=base)

    elif type == 'impedance':
        gs = gridspec.GridSpec(5, 4, width_ratios=(1, 1, 1, .05), height_ratios=(.1, .5, .3, .5, .1),
                               left=0.1, right=0.9, bottom=0.1, top=0.9,
                               wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[:, 0])
        base = ax0.imshow(b, vmin=vmin, vmax=vmax, cmap='terrain', extent=[0, m.shape[1], m.shape[0] * dt, 0])
        ax0.set_ylabel(ztitle)
        ax0.set_title('a) Baseline')
        ax0.axis('tight')
        
        ax1 = fig.add_subplot(gs[:, 1])
        mon = ax1.imshow(m, vmin=vmin, vmax=vmax, cmap='terrain')
        ax1.set_yticklabels([])
        ax1.set_title('b) Monitor')
        ax1.axis('tight')
        
        ax2 = fig.add_subplot(gs[:, 2])
        dif = ax2.imshow((m - b) / m, vmin=-dif_scale, vmax=dif_scale, cmap='seismic_r')
        ax2.set_yticklabels([])
        ax2.set_title('c) Monitor - Baseline')
        ax2.axis('tight')
        
        if vline is not None:
            plt.vlines(vline, 0, m.shape[0], 'k')
            
        ax3 = fig.add_subplot(gs[1, 3])
        ax3.set_title('Impedance \n $[m/s*g/cm^3]$', loc='left')
        Colorbar(ax=ax3, mappable=base)
        
        ax3 = fig.add_subplot(gs[3, 3])
        ax3.set_title('Difference \n [%]', loc='left')
        Colorbar(ax=ax3, mappable=dif)

        if mtrue is not None:
            rre1 = RRE(mtrue[0], b)
            snr1 = SNR(mtrue[0], b)
            rre2 = RRE(mtrue[1], m)
            snr2 = SNR(mtrue[1], m)
            rre3 = RRE(mtrue[1]-mtrue[0], m-b)
            # snr3 = SNR(mtrue[1]-mtrue[0], m-b)
            ax0.set_title(f'a) Baseline \n RRE = {rre1:.2f} SNR = {snr1:.2f}')
            ax1.set_title(f'b) Monitor \n RRE = {rre2:.2f} SNR = {snr2:.2f}')
            ax2.set_title(f'c) Monitor - Baseline \n RRE = {rre3:.2f}')

def plotter_timeshift(d1: np.ndarray, d2: np.ndarray, d2s: np.ndarray, shift: np.ndarray, 
                      dt: float = 1.0, perc: float = 1.0, height: float = 5.0, 
                      width: float = 20.0, dif_scale: float = 0.03, 
                      alpha: float = 0.7, cmap: str = 'RdGy') -> None:
    """
    Create a visualization of time-shifted seismic data.
    
    Parameters
    ----------
    d1 : np.ndarray
        Baseline data
    d2 : np.ndarray
        Monitor data
    d2s : np.ndarray
        Shifted monitor data
    shift : np.ndarray
        Time shift values
    dt : float, optional
        Time sampling interval, by default 1.0
    perc : float, optional
        Percentile for colormap clipping, by default 1.0
    height : float, optional
        Figure height in inches, by default 5.0
    width : float, optional
        Figure width in inches, by default 20.0
    dif_scale : float, optional
        Scale for timeshift display, by default 0.03
    alpha : float, optional
        Transparency for time shift overlay, by default 0.7
    cmap : str, optional
        Colormap for seismic data, by default 'RdGy'
    """
    vmin, vmax = np.percentile(d1, [perc, 100 - perc])

    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(5, 5, width_ratios=(1, 1, 1, 1., .05), height_ratios=(.1, .5, .5, .5, .1),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(d1, vmin=-vmax, vmax=vmax, cmap=cmap, extent=[0, d1.shape[1], d1.shape[0] * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
    ax0.set_title('Baseline')
    ax0.axis('tight')
    
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(d2, vmin=-vmax, vmax=vmax, cmap=cmap)
    ts = ax1.imshow(shift, vmin=-dif_scale, vmax=dif_scale, cmap='seismic', alpha=alpha)
    ax1.set_yticklabels([])
    ax1.set_title('Monitor')
    ax1.axis('tight')
    
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(d2 - d1, vmin=-vmax, vmax=vmax, cmap=cmap)
    ax2.set_yticklabels([])
    ax2.set_title('Monitor - Baseline')
    ax2.axis('tight')
    
    ax3 = fig.add_subplot(gs[:, 3])
    ax3.imshow(d2s - d1, vmin=-vmax, vmax=vmax, cmap=cmap)
    ax3.set_yticklabels([])
    ax3.set_title('Monitor(s) - Baseline')
    ax3.axis('tight')
    
    ax4 = fig.add_subplot(gs[1, 4])
    ax4.set_title('Amplitude', loc='left')
    Colorbar(ax=ax4, mappable=base)
    
    ax4 = fig.add_subplot(gs[3, 4])
    ax4.set_title('Time-shift \n [ms]', loc='left')
    Colorbar(ax=ax4, mappable=ts)

def plot_results(ui: np.ndarray, d1: np.ndarray, d2: np.ndarray, d2i: np.ndarray, 
                 C: Callable, dims: Tuple[int, ...], dt: float, 
                 xerr: Optional[List[float]] = None, xsnr: Optional[List[float]] = None, 
                 l2niter: Optional[int] = None) -> None:
    """
    Plot the results of strain inversion.
    
    Parameters
    ----------
    ui : np.ndarray
        Inverted strain
    d1 : np.ndarray
        Baseline data
    d2 : np.ndarray
        Original monitor data
    d2i : np.ndarray
        Shifted monitor data
    C : Callable
        Causal integration operator
    dims : Tuple[int, ...]
        Dimensions of the model
    dt : float
        Time sampling interval
    xerr : List[float], optional
        Relative reconstruction error at each iteration
    xsnr : List[float], optional
        Signal-to-noise ratio at each iteration
    l2niter : int, optional
        Number of L2 norm iterations for x-axis scaling
    """
    # Plot the inverted strain
    plt.figure()
    plt.imshow(ui.reshape(dims), cmap='PiYG', vmin=-0.1, vmax=0.1, extent=(0, dims[1], 0, dims[0]*dt))
    plt.title('Inverted Strain')
    plt.ylabel('Time [s]')
    plt.xlabel('Trace number')
    plt.axis('tight')
    plt.colorbar(shrink=0.5)
    plt.show()
    
    # Plot the time-shift
    plt.figure()
    plt.imshow((C*ui).reshape(dims), cmap='seismic', vmin=-0.03, vmax=0.03, 
              extent=(0, dims[1], 0, dims[0]*dt))
    plt.title('Time-shift')
    plt.ylabel('Time [s]')
    plt.xlabel('Trace number')
    plt.axis('tight')
    plt.colorbar(shrink=0.5)
    plt.show()
    
    # Plot the shifted data comparison
    plotter_timeshift(d1, d2, d2i, shift=(C*ui).reshape(dims), dt=dt)
    plt.show()
    
    # Plot error metrics if provided
    if xerr is not None and xsnr is not None and l2niter is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        x_iters = np.arange(len(xerr))*l2niter
        
        # Plot RRE
        ax[0].plot(x_iters, xerr, color='blue', linestyle='-', linewidth=1.5)
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('RRE value')
        ax[0].set_title('Relative Reconstruction Error')
        ax[0].grid(True, linestyle='--')
        
        # Plot SNR
        ax[1].plot(x_iters, xsnr, color='blue', linestyle='-', linewidth=1.5)
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('SNR (dB)')
        ax[1].set_title('Signal-to-Noise Ratio')
        ax[1].grid(True, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2) 
        plt.show()


def clim(in_content: np.ndarray, ratio: float = 95) -> Tuple[float, float]:
    """Calculate symmetric color limits for a plot.

    Parameters
    ----------
    in_content : np.ndarray
        Input data to calculate limits from.
    ratio : float, optional
        Percentile to use for clipping. Default is 95.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the negative and positive color limits.
    """
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(
    volume: np.ndarray, 
    t: Optional[int] = None, 
    x: Optional[int] = None, 
    y: Optional[int] = None, 
    vmin: Optional[float] = None, 
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8), 
    cmap: str = 'bone', 
    clipval: Optional[Tuple[float, float]] = None, 
    p: int = 98,
    tlim: Optional[Tuple[float, float]] = None, 
    xlim: Optional[Tuple[float, float]] = None, 
    ylim: Optional[Tuple[float, float]] = None,
    labels: Tuple[str, str, str] = ('[ms]', '', ''),
    tlabel: str = 't',
    ratio: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    linespec: Optional[Dict[str, Any]] = None, 
    title: str = '',
    filename: Optional[Union[str, Path]] = None,
    save_opts: Optional[Dict[str, Any]] = None,
    whspace: Optional[Tuple[float, float]] = None,
    colorbar_title: str = ''
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Create an exploded view of a 3D volume.

    Displays three orthogonal slices of a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        The 3D data volume.
    t, x, y : int, optional
        Indices of the slices to display. If None, defaults to the center.
    vmin, vmax : float, optional
        Color limits.
    figsize : tuple of (int, int), optional
        Figure size. Default is (8, 8).
    cmap : str, optional
        Colormap. Default is 'bone'.
    clipval : tuple of (float, float), optional
        Absolute values for clipping.
    p : int, optional
        Percentile for clipping if `clipval` is not provided. Default is 98.
    tlim, xlim, ylim : tuple of (float, float), optional
        Axis limits.
    labels : tuple of str, optional
        Axis labels units.
    tlabel : str, optional
        Label for the time/depth axis.
    ratio : tuple, optional
        Aspect ratios for the plots.
    linespec : dict, optional
        Specifications for the reference lines.
    title : str, optional
        Figure title.
    filename : str or Path, optional
        Path to save the figure.
    save_opts : dict, optional
        Options for saving the figure.
    whspace : tuple of (float, float), optional
        Width and height space between subplots.
    colorbar_title : str, optional
        Title for the colorbar.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    tuple of plt.Axes
        A tuple containing the three axes objects for the slices.
    """
    if linespec is None:
        linespec = dict(ls='-', lw=1, color='orange')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels
    
    t = t if t is not None else nt//2
    x = x if x is not None else nx//2
    y = y if y is not None else ny//2

    if tlim is None:
        t_label = "samples"
        tlim = (0, volume.shape[0])
    if xlim is None:
        x_label = "samples"
        xlim = (0, volume.shape[1])
    if ylim is None:
        y_label = "samples"
        ylim = (0, volume.shape[2])
    
    # vertical lines for coordinates reference
    tline = (tlim[1] - tlim[0]) / nt * t + tlim[0]
    xline = (xlim[1] - xlim[0]) / nx * x + xlim[0]
    yline = (ylim[1] - ylim[0]) / ny * y + ylim[0]
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    if ratio is None:
        wr = (nx, ny)
        hr = (nx, nt)
    else:
        wr = ratio[0]
        hr = ratio[1]

    if whspace is None:
        whspace = (0., 0.)

    opts = dict(cmap=cmap, vmin=vmin, vmax=vmax, 
                clim=clipval if clipval is not None else clim(volume, p))
    opts2 = dict(aspect=1.)
    
    gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                          left=0.1, right=0.9, bottom=0.1, top=1.0,
                          wspace=whspace[0], hspace=whspace[1])
    
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_ = fig.add_subplot(gs[0, 1], sharex=ax_right, sharey=ax_top)
    ax_.axis('off')
    ax.axis('tight')

    # central plot
    ax.imshow(volume[:, :, y], extent=[xlim[0], xlim[1], tlim[1], tlim[0]], **opts, **opts2)
    ax.axvline(x=xline, **linespec)
    ax.axhline(y=tline, **linespec)
    ax.axis('tight')

    # top plot
    ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    ax_top.sharex(ax)
    ax_top.invert_yaxis()
    
    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts, **opts2)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    ax_right.axis('tight')
    
    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("crossline " + x_label)
    ax.set_ylabel(tlabel + " " + t_label)
    ax_right.set_xlabel("inline " + y_label)
    ax_top.set_ylabel("inline " + y_label)

    cbar_ax = fig.add_axes([0.6, 0.7, 0.02, 0.15])
    cbar_ax.set_title(colorbar_title, fontsize=10, loc='left')
    plt.colorbar(ax.images[0], cax=cbar_ax)
    fig.tight_layout()
    
    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)
    return fig, (ax, ax_top, ax_right)


def plot_loss(
    epoch_losses: List[float], 
    snrs: List[float], 
    snrs_std: List[float], 
    filename: Optional[Union[str, Path]] = None
) -> None:
    """Plot loss and SNR curves over epochs.

    Parameters
    ----------
    epoch_losses : list of float
        A list of loss values for each epoch.
    snrs : list of float
        A list of SNR values for the mean estimate at each epoch.
    snrs_std : list of float
        A list of SNR values for the standard deviation at each epoch.
    filename : str or Path, optional
        Path to save the figure. If None, the plot is not saved.
        Default is None.
    """
    fig, ax = plt.subplots(1,3, figsize=(12,3))
    ax[0].plot(epoch_losses, 'k')
    ax[0].set_xlabel('# Iterations')
    ax[0].set_title('Total loss')
    ax[1].plot(snrs, 'k')
    ax[1].set_xlabel('# Iterations')
    ax[1].set_ylim(0,25)
    ax[1].set_title('SNR mean')
    ax[2].plot(snrs_std, 'k')
    ax[2].set_xlabel('# Iterations')
    ax[2].set_title('SNR std')
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()