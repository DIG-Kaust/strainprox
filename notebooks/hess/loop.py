import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from pylops.avo.poststack import PoststackLinearModelling
from strainprox.utils import *
from strainprox.functions import strain_jis_cp
from pylops import CausalIntegration
from pylops.basicoperators import Gradient, Diagonal, FirstDerivative
from pyproximal import L21, L2
from pyproximal.optimization.primaldual import PrimalDual


def main():
    parser = argparse.ArgumentParser(description="Strain inversion grid search")
    parser.add_argument("--mu", type=float, required=True, help="Step size parameter mu")
    parser.add_argument("--beta", type=float, required=True, help="TV regularization weight for segmentation")
    parser.add_argument("--delta", type=float, required=True, help="Segmentation misfit weight")
    parser.add_argument("--alpha", type=float, default=10., help="TV regularization weight for inversion")
    parser.add_argument("--niter", type=int, default=2, help="Number of inner iterations")
    parser.add_argument("--pdniter", type=int, default=10, help="Primal-Dual iterations")
    parser.add_argument("--segmentniter", type=int, default=10, help="Segmentation iterations")
    parser.add_argument("--bisectniter", type=int, default=10, help="Bisection iterations for simplex")
    parser.add_argument("--outdir", type=str, default="../results/hess/grid_search",
                        help="Output directory for images")
    args = parser.parse_args()

    mu = args.mu
    beta = args.beta
    delta = args.delta
    alpha = args.alpha
    niter = args.niter
    pdniter = args.pdniter
    segmentniter = args.segmentniter
    bisectniter = args.bisectniter
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    dtype = 'float64'
    hess = np.load('../../data/Hess/Hess4d_time.npz')
    wav = cp.asarray(hess['wav'].astype(np.float32))
    d1 = cp.asarray(hess['dn1'].astype(np.float32))
    d2 = cp.asarray(hess['dn2'].astype(np.float32))
    utrue = cp.asarray(hess['straint'].astype(np.float32))
    dims = utrue.shape
    dt = 0.004
    t = cp.arange(dims[0], dtype='float64') * dt

    C = CausalIntegration(dims=dims, axis=0, sampling=dt, dtype=dtype)
    D = FirstDerivative(dims=dims, axis=0, sampling=dt, dtype=dtype)
    G = PoststackLinearModelling(wav / 2, nt0=dims[0], spatdims=dims[1:])
    Dop = Gradient(dims=dims, edge=True, dtype=dtype, kind='forward')
    l1 = L21(ndim=len(dims), sigma=1.)

    outeriter = 100
    l2niter = 25

    ui = cp.zeros(np.prod(dims))
    d2i = d2.copy()
    gt_shift = C * utrue.reshape(dims)
    gt_strain = utrue.reshape(dims)

    cl = cp.array([-0.15, 0, 0.15])

    L_approx = 8.
    tau = 0.99 / (mu * L_approx)

    metric_keys = ['global_mae', 'global_rmse', 'roi_mae', 'roi_rmse', 'bg_leakage', 'dice']
    history = {k: [] for k in [
        'loss_total', 'loss_data', 'loss_reg',
        *[f'shift_{k}' for k in metric_keys],
        *[f'strain_{k}' for k in metric_keys],
    ]}

    start_time = time.perf_counter()
    for i in range(outeriter):
        d = d1.ravel() - d2i.ravel() - (G * ui)

        J = Diagonal(D * d2i, dtype='float32')
        Op = G + J * C

        du, ui, seg_v, vcl = strain_jis_cp(
            d=d,
            Op=Op,
            x0=cp.zeros(np.prod(dims)),
            ui=ui,
            dims=dims,
            cl=cl,
            alpha=alpha,
            beta=beta,
            delta=delta,
            tau=tau,
            mu=mu,
            niter=niter,
            l2niter=l2niter,
            pdniter=pdniter,
            segmentniter=segmentniter,
            bisectniter=bisectniter,
            bregman=False,
            plotflag=False,
            show=False
        )

        d2i = apply_time_shift_cupy(d=d2, time_shift=C * ui, t=t, dims=dims)

        data_term = float(cp.linalg.norm(d - Op * du) ** 2)
        reg_term = float(l1(Dop * ui))
        history['loss_data'].append(data_term)
        history['loss_reg'].append(reg_term)
        history['loss_total'].append(data_term + reg_term)

        pred_shift = C * ui.reshape(dims)
        pred_strain = ui.reshape(dims)
        shift_metrics = compute_all_metrics(pred=pred_shift, gt=gt_shift, alpha=0.05)
        strain_metrics = compute_all_metrics(pred=pred_strain, gt=gt_strain, alpha=0.05)
        for k, val in shift_metrics.items():
            history[f'shift_{k}'].append(val)
        for k, val in strain_metrics.items():
            history[f'strain_{k}'].append(val)

        if i % 10 == 0:
            print(
                f"iter {i + 1:03d} | "
                f"total = {data_term + reg_term:.3f} | "
                f"data = {data_term:.3f} | "
                f"reg = {reg_term:.3f} | "
                f"step = {float(cp.linalg.norm(du)):.3f} | "
                + " | ".join(f"shift_{k} = {val:.3f}" for k, val in shift_metrics.items())
                + " | "
                + " | ".join(f"strain_{k} = {val:.3f}" for k, val in strain_metrics.items())
            )

    total_time = time.perf_counter() - start_time
    print(f'Total time {total_time / 60:.2f} minutes')

    pred_shift_np = cp.asnumpy(pred_shift)
    pred_strain_np = cp.asnumpy(pred_strain)
    d2i_np = cp.asnumpy(d2i)

    tag = f"mu{mu}_alpha{alpha}_beta{beta}_delta{delta}_niter{niter}_pd{pdniter}_seg{segmentniter}_bis{bisectniter}"
    fname = os.path.join(outdir, f"strain_shift_{tag}.png")
    strain_shift(strain=pred_strain_np, shift=pred_shift_np, filename=fname)
    print(f"Saved: {fname}")

    np.savez(
        os.path.join(outdir, f"time_strain_jis_{tag}.npz"),
        # shift=pred_shift_np,
        # strain=pred_strain_np,
        # d2s=d2i_np,
        **{k: np.array(v) for k, v in history.items()},
    )


if __name__ == "__main__":
    main()
