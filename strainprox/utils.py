import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Union, Callable

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


def apply_time_shift(t: np.ndarray, ui: np.ndarray, d2: np.ndarray, 
                     C: Callable, dims: Tuple[int, ...]) -> np.ndarray:
    """
    Apply time shift to the monitor data based on current strain estimate.
    
    Parameters
    ----------
    t : np.ndarray
        Time axis
    ui : np.ndarray
        Current strain estimate
    d2 : np.ndarray
        Original monitor data
    C : Callable
        Causal integration operator
    dims : Tuple[int, ...]
        Dimensions of the model
    
    Returns
    -------
    np.ndarray
        Shifted monitor data
    """
    d2i = np.zeros_like(d2)
    
    # Create time-shift interpolators for each trace
    time_shifts = (C*ui).reshape(dims)
    shifted_time_grid = t.copy()[:, np.newaxis] - time_shifts
    
    # Apply interpolation for each trace
    for col in range(d2.shape[1]):
        interpolator = interp1d(
            shifted_time_grid[:, col], 
            d2[:, col], 
            kind='cubic', 
            fill_value="extrapolate"
        )
        d2i[:, col] = interpolator(t)
        
    return d2i


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