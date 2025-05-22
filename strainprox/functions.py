import numpy as np
from pylops import Diagonal, Gradient, VStack as VStacklop
from pyproximal import L21, L2
import matplotlib.pyplot as plt
from strainprox.utils import *
from pyproximal.optimization.segmentation import Segment
from pyproximal.optimization.primaldual import PrimalDual

def strain_jis(d, Op, x0, ui,  dims, cl, 
               alpha, beta, delta, tau, mu,
               niter=4, l2niter=20, pdniter=100, segmentniter=10, bisectniter=30, 
               tolstop=0., bregman=False, utrue=None, plotflag=True, show=False):
    
    r"""
    Joint Inversion and Segmentation of seismic time-strains (JISS) using a Primal-Dual solver.

    This function performs joint inversion and segmentation of input data
    using a primal-dual optimization algorithm. It incorporates total 
    variation (TV) regularization for the inversion and segmentation, 
    as well as optional Bregman iterations for enhanced convergence.

    Parameters
    ----------
    d : np.ndarray
        Input data array, typically representing measurements (e.g., seismic data).
        Must be 2D or 3D, with depth/time along the first axis.
    Op : pylops.avo.poststack.PoststackLinearModelling
        Forward modeling operator used in the inversion.
    x0 : np.ndarray
        Initial vector for the Primal-Dual solver. Must have the same shape as ui.
    ui : np.ndarray
        Initial estimate of the inverted model.
    dims : tuple
        Dimensions of the model grid (e.g., `(nz, nx)` for 2D or `(nz, nx, ny)` for 3D).
    cl : np.ndarray
        Array of class values for segmentation.
    alpha : float
        Scaling factor for the TV regularization term applied to the model inversion.
    beta : float
        Scaling factor for the TV regularization term applied to segmentation.
    delta : float
        Weight for the segmentation misfit term.
    tau : float
        Step size for the primal variable updates.
    mu : float
        Step size for the dual variable updates.
    niter : int, optional
        Number of outer iterations of the joint inversion-segmentation scheme. Default is 4.
    l2niter : int, optional
        Number of iterations for the L2 proximal solver. Default is 20.
    pdniter : int, optional
        Number of iterations for the Primal-Dual solver. Default is 100.
    segmentniter : int, optional
        Number of iterations for the segmentation solver. Default is 10.
    bisectniter : int, optional
        Number of iterations for bisection in the simplex proximal method. Default is 30.
    tolstop : float, optional
        Stopping tolerance based on the change in segmentation between iterations. Default is 0.
    bregman : bool, optional
        If True, enables Bregman iterations to enhance convergence. Default is False.
    utrue : np.ndarray, optional
        True model for comparison. If provided, metrics like RRE and PSNR are computed.
        Must have the same dimensions as the model.
    plotflag : bool, optional
        If True, visualizes intermediate results for each iteration. Default is True.
    show : bool, optional
        If True, prints iteration logs for solvers. Default is False.

    Returns
    -------
    ui : np.ndarray
        Final inverted time-strains.
    v : np.ndarray
        Final segmentation probabilities.
    vcl : np.ndarray
        Estimated segmentation (assigned class for each model point).
    xerr : list or None
        Relative Reconstruction Error (RRE) for each iteration, if `utrue` is provided.
    xsnr : list or None
        Peak Signal-to-Noise Ratio (PSNR) for each iteration, if `utrue` is provided.

    Notes
    -----
    - The algorithm alternates between inversion (updating `ui`) and segmentation 
      (updating `v` and `vcl`) for `niter` iterations.
    - The use of Bregman iterations (`bregman=True`) might in some cases improve 
      performance by recovering the contrast that is usually lost in standard TV
      regularization.
    """

    print('Working with alpha=%f,  beta=%f,  delta=%f' % (alpha, beta, delta))
    
    msize = x0.size
    ncl = len(cl)

    # TV regularization term
    Dop = Gradient(dims=dims, edge=True, dtype='float32', kind='forward')
    l1 = L21(ndim=2, sigma=alpha)
    v = np.zeros(ncl * msize)

    if bregman:
        p = np.zeros(msize)
        q = np.zeros(ncl * msize)
    
    # u_hist = []
    # v_hist = []
    if utrue is not None:
        xhist, xsnr, xerr = [], [], []

    if plotflag:
        fig, axs = plt.subplots(2, niter, figsize=(4 * niter, 10))
    
    for iiter in range(niter):
        print('Iteration %d...' % iiter)
        ui_old, v_old = ui.copy(), v.copy()

    
        # Gradient on the previous estimate
        gu = Dop * ui

        #############
        # Inversion #
        #############
        if iiter == 0:
            l2 = L2(Op=Op, b=d, niter=l2niter, warm=True, x0=np.zeros(np.prod(dims)))
        
        else:
            # define misfit term
            v = v.reshape((msize, ncl))         
            L1op = VStacklop([Op] + [Diagonal(np.sqrt(2.*delta)*np.sqrt(v[:, icl])) for icl in range(ncl)])
            d1 = np.hstack([d.ravel(), np.sqrt(2.*delta)*(np.sqrt(v).T).ravel() * ((ui[:, None] - cl).T).ravel()])
            l2 = L2(Op=L1op, b=d1, niter=l2niter, warm=True, q=p if bregman else None, alpha=-alpha if bregman else None)

        # solve
        if utrue is not None:
            du = PrimalDual(proxf=l2, proxg=l1.precomposition(a=1., b=gu), A=Dop, 
                    tau=tau, mu=mu, theta=1., x0=x0, niter=pdniter,
                callback=lambda xx:callbackx(xx.copy(), ui.copy(), utrue.ravel(), xhist, xsnr, xerr), 
                show=False)
        else:
            du = PrimalDual(proxf=l2, proxg=l1.precomposition(a=1., b=gu), A=Dop, 
                tau=tau, mu=mu, theta=1., x0=x0, niter=pdniter,
                show=show)
        ui += du

        if bregman:
            l2_grad = L2(Op=(Op if iiter == 0 else L1op), b=(d.ravel() if iiter == 0 else d1))
            p -= np.real((1. / alpha) * l2_grad.grad(du))

        # u_hist.append(ui.copy())

        if plotflag:
            if niter==1:
                axs[0].imshow(np.real(ui).reshape(dims), 'gray')
                axs[0].axis('tight')
            else:
                axs[0, iiter].imshow(np.real(ui).reshape(dims), 'gray')
                axs[0, iiter].axis('tight')

        ################
        # Segmentation #
        ################
        v, vcl = Segment(ui, cl, 2 * delta, 2 * beta, z=(-beta * q if bregman else None),
                        niter=segmentniter, callback=None, show=show,
                        kwargs_simplex=dict(engine='numba',
                                            maxiter=bisectniter, call=False))
        # v_hist.append(v)

        # Update q
        if bregman:
            q -= (delta / beta) * ((ui.ravel() - cl[:, np.newaxis]) ** 2).ravel()

        if plotflag:
            if niter==1:
                axs[1].imshow(vcl.reshape(dims), 'gray')
                axs[1].axis('tight')
            else:    
                axs[1, iiter].imshow(vcl.reshape(dims), 'gray')
                axs[1, iiter].axis('tight')

        # Monitor cost functions
        print('f=', L2(Op=Op, b=d.ravel())(ui))
        print('||v-v_old||_2=', np.linalg.norm(v.ravel() - v_old.ravel()))
        print('||m-m_old||_2=', np.linalg.norm(ui.ravel() - ui_old.ravel()))

        # Check stopping criterion
        if np.linalg.norm(v.ravel()-v_old.ravel()) < tolstop:
            break

    return ui, v, vcl, (xerr if utrue is not None else None), (xsnr if utrue is not None else None)


