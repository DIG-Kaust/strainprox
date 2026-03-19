import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
from pathlib import Path
import cupy as cp


#########################
# Metrics
#########################

def _to_float_arrays(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt, dtype=np.float64)
    # Allow broadcasting (e.g., gt (H,W) vs pred (N,H,W))
    pred, gt = np.broadcast_arrays(pred, gt)
    return pred, gt

def roi(gt, alpha: float = 0.05) -> np.ndarray:
    """
    ROI mask from ground truth magnitude:
        M = {|gt| >= alpha * max(|gt|)}
    """
    gt = np.asarray(gt, dtype=np.float64)
    tau = alpha * np.max(np.abs(gt))
    return np.abs(gt) >= tau

def rmse(pred, gt) -> float:
    pred, gt = _to_float_arrays(pred, gt)
    err = pred - gt
    return float(np.sqrt(np.mean(err * err)))

def mae(pred, gt) -> float:
    pred, gt = _to_float_arrays(pred, gt)
    return float(np.mean(np.abs(pred - gt)))

def roi_rmse(pred, gt, roi_mask: np.ndarray) -> float:
    pred, gt = _to_float_arrays(pred, gt)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    roi_mask = np.broadcast_to(roi_mask, pred.shape)  # allow broadcasting masks
    err = pred - gt
    n = int(np.sum(roi_mask))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.sum((err[roi_mask]) ** 2) / n))

def roi_mae(pred, gt, roi_mask: np.ndarray) -> float:
    pred, gt = _to_float_arrays(pred, gt)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    roi_mask = np.broadcast_to(roi_mask, pred.shape)
    err = np.abs(pred - gt)
    n = int(np.sum(roi_mask))
    if n == 0:
        return float("nan")
    return float(np.sum(err[roi_mask]) / n)

def ble(pred, roi_mask: np.ndarray) -> float:
    """
    Background Leakage Energy (assuming GT ~ 0 in background):
        Leak_out = sqrt( mean_{i in ~M} pred_i^2 )

    If you want leakage of the *error* instead, call with pred=(pred-gt).
    """
    pred = np.asarray(pred, dtype=np.float64)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    pred = np.asarray(pred, dtype=np.float64)
    pred, roi_mask = np.broadcast_arrays(pred, roi_mask)

    bg = ~roi_mask
    n = int(np.sum(bg))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean(pred[bg] ** 2)))

def dice(
    pred,
    gt,
    alpha: float = 0.05,
    use_lcc: bool = False,
) -> float:
    """
    Dice overlap between GT anomaly mask and predicted anomaly mask.
    Threshold is set from GT: tau = alpha * max(|gt|).
      M      = {|gt|   >= tau}
      M_hat  = {|pred| >= tau}

    If use_lcc=True, you can optionally keep only the largest connected component
    (requires scipy; see note below).
    """
    pred, gt = _to_float_arrays(pred, gt)
    tau = alpha * np.max(np.abs(gt))
    M = np.abs(gt) >= tau
    Mhat = np.abs(pred) >= tau

    if use_lcc:
        # Optional: keep largest connected component (2D/3D only).
        # Requires: from scipy.ndimage import label
        from scipy.ndimage import label

        def lcc(mask: np.ndarray) -> np.ndarray:
            lab, nlab = label(mask)
            if nlab == 0:
                return mask
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0  # background
            keep = sizes.argmax()
            return lab == keep

        M = lcc(M)
        Mhat = lcc(Mhat)

    inter = np.logical_and(M, Mhat).sum(dtype=np.float64)
    denom = M.sum(dtype=np.float64) + Mhat.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)

def compute_all_metrics(
    pred,
    gt,
    alpha: float = 0.05,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper. If roi_mask is None, it is built from GT using alpha.
    """
    if isinstance(pred, cp.ndarray):
        pred = cp.asnumpy(pred)
    if isinstance(gt, cp.ndarray):
        gt = cp.asnumpy(gt)
    if isinstance(roi_mask, cp.ndarray):
        roi_mask = cp.asnumpy(roi_mask)

    if roi_mask is None:
        roi_mask = roi(gt, alpha=alpha)

    return {
        "global_rmse": rmse(pred, gt),
        "global_mae": mae(pred, gt),
        "roi_rmse": roi_rmse(pred, gt, roi_mask),
        "roi_mae": roi_mae(pred, gt, roi_mask),
        "bg_leakage": ble(pred, roi_mask),
        "dice": dice(pred, gt, alpha=alpha),
    }


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
               cmap: str = 'RdGy', vline: Optional[int] = None, 
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
    if isinstance(b, cp.ndarray):
        b = cp.asnumpy(b)
    if isinstance(m, cp.ndarray):
        m = cp.asnumpy(m)
    if isinstance(ref, cp.ndarray):
        ref = cp.asnumpy(ref)
    if isinstance(mtrue, cp.ndarray):
        mtrue = cp.asnumpy(mtrue)

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
    if isinstance(d1, cp.ndarray):
        d1 = cp.asnumpy(d1)
    if isinstance(d2, cp.ndarray):
        d2 = cp.asnumpy(d2)
    if isinstance(d2s, cp.ndarray):
        d2s = cp.asnumpy(d2s)
    if isinstance(shift, cp.ndarray):
        shift = cp.asnumpy(shift)

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

def strain_shift(strain, shift, 
                vmin_strain: float = -0.1, vmax_strain: float = 0.1, 
                vmin_shift: float = -0.03, vmax_shift: float = 0.03,
                title_strain: str = 'time strain', title_shift: str = 'time shift',
                filename: Optional[str] = None):
    """
    Plot time strain and time shift side by side.

    Args:
        strain: 2D array for time strain.
        shift: 2D array for time shift.
        vmin_strain: float, optional
        vmax_strain: float, optional
        vmin_shift: float, optional
        vmax_shift: float, optional
        title_strain: str, optional
        title_shift: str, optional
        filename: str, optional. If provided, save figure to this path instead of showing.
    """
    if isinstance(strain, cp.ndarray):
        strain = cp.asnumpy(strain)
    if isinstance(shift, cp.ndarray):
        shift = cp.asnumpy(shift)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Left: time strain
    im0 = axes[0].imshow(strain, cmap='PiYG', vmin=vmin_strain, vmax=vmax_strain)
    axes[0].set_title(title_strain)
    axes[0].axis('tight')
    fig.colorbar(im0, ax=axes[0], shrink=0.5)

    # Right: time shift
    im1 = axes[1].imshow(shift, cmap='seismic', vmin=vmin_shift, vmax=vmax_shift)
    axes[1].set_title(title_shift)
    axes[1].axis('tight')
    fig.colorbar(im1, ax=axes[1], shrink=0.5)

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def results_grid(
    utrue: np.ndarray,
    tautrue: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    methods: Dict[str, Dict[str, np.ndarray]],
    dims: Tuple[int, ...],
    dt: float,
    *,
    figsize: Tuple[float, float] = (15, 20),
    strain_limits: Tuple[float, float] = (-0.1, 0.1),
    shift_limits: Tuple[float, float] = (-0.03, 0.03),
    amp_limits: Tuple[float, float] = (-0.1, 0.1),
    cmap_seis: str = 'RdGy'
) -> plt.Figure:
    """
    Plot a 5x3 grid: rows = [Ground truth, TK, TKST, TV, JIS], cols = [time strain, time shift, difference].

    Parameters
    ----------
    utrue : np.ndarray
        Ground-truth time strain field (flattened or shaped to `dims`).
    d1 : np.ndarray
        Baseline seismic (shaped to `dims`).
    d2 : np.ndarray
        Original monitor seismic (shaped to `dims`), used to derive corrected monitors if not provided.
    methods : Dict[str, Dict[str, np.ndarray]]
        Dictionary keyed by method identifier ('tk', 'tkst', 'tv', 'jis').
        Each value is a dict with:
            - 'strain': np.ndarray, time strain estimate
            - 'shift': np.ndarray, time shift estimate
            - 'd2s': np.ndarray, corrected monitor seismic
            - 'strain_roi_mae': float, ROI MAE for time strain
            - 'shift_roi_mae': float, ROI MAE for time shift
    dims : tuple
        Target 2D or 3D dimensions (nt, nx[, ny]) for reshaping.
    dt : float
        Time sampling interval in seconds.
    figsize : tuple, optional
        Figure size in inches. Default: (15, 12).
    strain_limits : tuple, optional
        vmin, vmax for time strain plots. Default: (-0.1, 0.1).
    shift_limits : tuple, optional
        vmin, vmax for time shift plots. Default: (-0.03, 0.03).
    amp_limits : tuple, optional
        vmin, vmax for seismic amplitude differences. Default: (-0.1, 0.1).
    cmap_seis : str, optional
        Colormap for seismic images. Default: 'RdGy'.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    method_labels = {'tk': 'TK', 'tkst': 'TKST', 'tv': 'TV', 'jis': 'JIS'}
    method_order = ['tk', 'tkst', 'tv', 'jis']

    def as_img(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(dims)

    t = np.arange(dims[0]) * dt

    rows: List[Dict[str, Any]] = []

    # Ground truth row
    shift_true = as_img(tautrue)
    d2_true = apply_time_shift(d2, shift_true, t, dims)
    rows.append(dict(name='Ground truth', u=utrue, shift=shift_true, d2s=d2_true))

    # Method rows in fixed order
    for key in method_order:
        if key not in methods:
            continue
        m = methods[key]
        rows.append(dict(
            name=method_labels.get(key, key.upper()),
            u=m['strain'],
            shift=as_img(m['shift']),
            d2s=m['d2s'],
            strain_roi_mae=m['strain_roi_mae'][-1],
            shift_roi_mae=m['shift_roi_mae'][-1],
        ))

    # Prepare figure with 5 rows x 3 columns
    nrows = 5
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])  # ensure 2D indexing

    # Column headers on first row
    col_titles = ['Time strain', 'Time shift', 'Monitor(s) - Baseline']

    # Determine how many rows we will actually draw (cap at 5)
    used_rows = min(len(rows), nrows)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    for i in range(nrows):
        if i < used_rows:
            row = rows[i]
            u_img = as_img(row['u'])
            shift_img = row['shift']
            diff_img = d1 - row['d2s']

            # Time strain
            ax = axes[i, 0]
            im0 = ax.imshow(u_img, cmap='PiYG', vmin=strain_limits[0], vmax=strain_limits[1],
                            extent=(0, dims[1], dims[0] * dt, 0))
            ax.axis('tight')
            ax.set_ylabel('TWT $[s]$')
            if row['name'] == 'Ground truth':
                ax.set_title(f"{row['name']}")
            else:
                ax.set_title(f"{row['name']} | MAE: {row['strain_roi_mae']:.4f}")

            # Time shift
            ax = axes[i, 1]
            im1 = ax.imshow(shift_img, cmap='seismic', vmin=shift_limits[0], vmax=shift_limits[1],
                            extent=(0, dims[1], dims[0] * dt, 0))
            ax.axis('tight')
            ax.set_yticklabels([])
            if row['name'] != 'Ground truth':
                ax.set_title(f"MAE: {row['shift_roi_mae']:.4f}")

            # Difference
            ax = axes[i, 2]
            im2 = ax.imshow(diff_img, cmap=cmap_seis, vmin=amp_limits[0], vmax=amp_limits[1],
                            extent=(0, dims[1], dims[0] * dt, 0))
            ax.axis('tight')
            ax.set_yticklabels([])
        else:
            # Hide unused rows
            for j in range(ncols):
                axes[i, j].axis('off')

    plt.close(fig)
    return fig


def metrics_comparison(
    methods: Dict[str, Dict[str, np.ndarray]],
    *,
    figsize: Tuple[float, float] = (8, 16),
    log_interval: int = 1,
) -> plt.Figure:
    """
    Plot 12 metric convergence curves (4x3 grid) comparing methods.

    Parameters
    ----------
    methods : Dict[str, Dict[str, np.ndarray]]
        Dictionary keyed by method identifier ('tk', 'tkst', 'tv', 'jis').
        Each value must contain 1-D arrays for:
            'shift_global_rmse', 'shift_roi_rmse', 'shift_global_mae',
            'shift_roi_mae', 'shift_bg_leakage', 'shift_dice',
            'strain_global_rmse', 'strain_roi_rmse', 'strain_global_mae',
            'strain_roi_mae', 'strain_bg_leakage', 'strain_dice'
    figsize : tuple, optional
        Figure size in inches. Default: (18, 16).
    log_interval : int, optional
        Iteration spacing between logged values. Default: 10.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    metric_keys = [
        'shift_global_rmse', 'shift_roi_rmse', 'shift_global_mae',
        'shift_roi_mae', 'shift_bg_leakage', 'shift_dice',
        'strain_global_rmse', 'strain_roi_rmse', 'strain_global_mae',
        'strain_roi_mae', 'strain_bg_leakage', 'strain_dice',
    ]
    metric_labels = {
        'shift_global_rmse': 'Shift Global RMSE',
        'shift_roi_rmse': 'Shift ROI RMSE',
        'shift_global_mae': 'Shift Global MAE',
        'shift_roi_mae': 'Shift ROI MAE',
        'shift_bg_leakage': 'Shift BG Leakage',
        'shift_dice': 'Shift Dice',
        'strain_global_rmse': 'Strain Global RMSE',
        'strain_roi_rmse': 'Strain ROI RMSE',
        'strain_global_mae': 'Strain Global MAE',
        'strain_roi_mae': 'Strain ROI MAE',
        'strain_bg_leakage': 'Strain BG Leakage',
        'strain_dice': 'Strain Dice',
    }
    method_labels = {'tk': 'TK', 'tkst': 'TKST', 'tv': 'TV', 'jis': 'JIS'}
    method_order = ['tk', 'tkst', 'tv', 'jis']

    

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=figsize, constrained_layout=True)

    for idx, key in enumerate(metric_keys):
        ax = axes[idx // 2, idx % 2]
        xmin = 0
        xmax = 100
        ax.set_xlim(xmin, xmax)

        vals_all = np.concatenate([np.asarray(methods[m][key][xmin:xmax]) for m in method_order if m in methods and key in methods[m]])
        ax.set_ylim(vals_all.min() * 0.9, vals_all.max() * 1.1)

        for mkey in method_order:
            if mkey not in methods or key not in methods[mkey]:
                continue
            vals = np.asarray(methods[mkey][key])
            iters = np.arange(1, len(vals) + 1) * log_interval
            ax.plot(iters, vals, label=method_labels.get(mkey, mkey.upper()))
        ax.set_title(metric_labels[key])
        ax.set_xlabel('Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
            

    plt.close(fig)
    return fig


def metrics_comparison2(
    methods: Dict[str, Dict[str, np.ndarray]],
    *,
    figsize: Tuple[float, float] = (16, 4),
    log_interval: int = 1,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot shift and strain metric convergence curves comparing methods.

    Returns two separate figures (1x4 each): one for shift metrics,
    one for strain metrics.

    Parameters
    ----------
    methods : Dict[str, Dict[str, np.ndarray]]
        Dictionary keyed by method identifier ('tk', 'tkst', 'tv', 'jis').
        Each value must contain 1-D arrays for the metric keys listed below.
    figsize : tuple, optional
        Figure size in inches for each figure. Default: (8, 16).
    log_interval : int, optional
        Iteration spacing between logged values. Default: 1.

    Returns
    -------
    (fig_shift, fig_strain) : tuple of matplotlib.figure.Figure
    """
    shift_keys = ['shift_global_mae', 'shift_roi_mae', 'shift_bg_leakage', 'shift_dice']
    strain_keys = ['strain_global_mae', 'strain_roi_mae', 'strain_bg_leakage', 'strain_dice']
    metric_labels = {
        'shift_global_mae': 'Global MAE',
        'shift_roi_mae': 'ROI MAE',
        'shift_bg_leakage': 'BG Leakage',
        'shift_dice': 'Dice',
        'strain_global_mae': 'Global MAE',
        'strain_roi_mae': 'ROI MAE',
        'strain_bg_leakage': 'BG Leakage',
        'strain_dice': 'Dice',
    }
    method_labels = {'tk': 'TK', 'tkst': 'TKST', 'tv': 'TV', 'jis': 'JIS'}
    method_order = ['tk', 'tkst', 'tv', 'jis']

    def _plot_group(keys, title):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize, constrained_layout=True)
        fig.suptitle(title)
        for idx, key in enumerate(keys):
            ax = axes[idx]
            xmin, xmax = 0, 100
            ax.set_xlim(xmin, xmax)

            vals_all = np.concatenate([
                np.asarray(methods[m][key][xmin:xmax])
                for m in method_order if m in methods and key in methods[m]
            ])
            ax.set_ylim(vals_all.min() * 0.9, vals_all.max() * 1.1)

            for mkey in method_order:
                if mkey not in methods or key not in methods[mkey]:
                    continue
                vals = np.asarray(methods[mkey][key])
                iters = np.arange(1, len(vals) + 1) * log_interval
                ax.plot(iters, vals, label=method_labels.get(mkey, mkey.upper()))
            ax.set_title(metric_labels[key])
            ax.set_xlabel('Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        return fig

    fig_shift = _plot_group(shift_keys, 'Shift Metrics')
    fig_strain = _plot_group(strain_keys, 'Strain Metrics')
    return fig_shift, fig_strain



def metrics_comparison3(
    results_dir: Optional[str] = '/ibex/user/romerojd/strainprox/notebooks/results/hess/grid_search',
    *,
    files: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 4),
    log_interval: int = 1,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Load .npz result files and plot shift / strain metric convergence
    curves comparing all configurations.

    Provide *either* ``results_dir`` (loads every .npz in the directory)
    *or* ``files`` (an explicit list of .npz paths).  If both are given
    ``files`` takes precedence.

    Returns two separate figures (1x4 each): one for shift metrics,
    one for strain metrics.

    Parameters
    ----------
    results_dir : str or None
        Directory containing the .npz result files.
    files : list of str or None
        Explicit list of .npz file paths to load.
    figsize : tuple, optional
        Figure size in inches for each figure.
    log_interval : int, optional
        Iteration spacing between logged values. Default: 1.

    Returns
    -------
    (fig_shift, fig_strain) : tuple of matplotlib.figure.Figure
    """
    if files is not None:
        npz_files = [Path(f) for f in files]
    elif results_dir is not None:
        npz_files = sorted(Path(results_dir).glob('*.npz'))
    else:
        raise ValueError('Provide either results_dir or files')
    if not npz_files:
        raise FileNotFoundError('No .npz files found')

    methods: Dict[str, Dict[str, np.ndarray]] = {}
    for f in npz_files:
        label = f.stem.replace('time_strain_jis_', '')
        data = np.load(f)
        methods[label] = {k: data[k] for k in data.files}

    method_names = sorted(methods.keys())

    shift_keys = ['shift_global_mae', 'shift_roi_mae', 'shift_bg_leakage', 'shift_dice']
    strain_keys = ['strain_global_mae', 'strain_roi_mae', 'strain_bg_leakage', 'strain_dice']
    metric_labels = {
        'shift_global_mae': 'Global MAE',
        'shift_roi_mae': 'ROI MAE',
        'shift_bg_leakage': 'BG Leakage',
        'shift_dice': 'Dice',
        'strain_global_mae': 'Global MAE',
        'strain_roi_mae': 'ROI MAE',
        'strain_bg_leakage': 'BG Leakage',
        'strain_dice': 'Dice',
    }

    def _plot_group(keys, title):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize, constrained_layout=True)
        fig.suptitle(title)
        for idx, key in enumerate(keys):
            ax = axes[idx]
            xmin, xmax = 0, 100
            ax.set_xlim(xmin, xmax)

            vals_all = np.concatenate([
                np.asarray(methods[m][key][xmin:xmax])
                for m in method_names if key in methods[m]
            ])
            ax.set_ylim(vals_all.min() * 0.9, vals_all.max() * 1.1)

            for mkey in method_names:
                if key not in methods[mkey]:
                    continue
                vals = np.asarray(methods[mkey][key])
                iters = np.arange(1, len(vals) + 1) * log_interval
                ax.plot(iters, vals, alpha=0.4, linewidth=0.8, label=mkey)
            ax.set_title(metric_labels[key])
            ax.set_xlabel('Iteration')
            ax.grid(True, alpha=0.3)
        return fig

    fig_shift = _plot_group(shift_keys, 'Shift Metrics')
    fig_strain = _plot_group(strain_keys, 'Strain Metrics')
    return fig_shift, fig_strain


def top_k_configs(
    methods: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    *,
    iter_range: Tuple[int, int] = (0, 100),
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the *k* best configurations for a given metric over an
    iteration window.

    "Best" = lowest value for MAE / RMSE / leakage metrics,
    highest value for Dice.

    Parameters
    ----------
    methods : Dict[str, Dict[str, np.ndarray]]
        Dictionary keyed by configuration label.  Each value is a dict
        mapping metric names to 1-D arrays (one entry per logged iteration).
    metric : str
        Metric key, e.g. ``'strain_roi_mae'``, ``'shift_dice'``.
    iter_range : (int, int), optional
        (start, stop) iteration indices (0-based, exclusive stop).
        Default: ``(0, 100)``.
    k : int, optional
        Number of top configurations to return. Default: 5.

    Returns
    -------
    list of (label, score)
        Sorted from best to worst.
    """
    higher_is_better = 'dice' in metric.lower()
    start, stop = iter_range

    scores: List[Tuple[str, float]] = []
    for label, data in methods.items():
        if metric not in data:
            continue
        vals = np.asarray(data[metric])[start:stop]
        best_val = float(vals.max() if higher_is_better else vals.min())
        scores.append((label, best_val))

    scores.sort(key=lambda x: x[1], reverse=higher_is_better)
    return scores[:k]