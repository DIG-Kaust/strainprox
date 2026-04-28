import numpy as np
from pylops import Diagonal, Gradient, VStack as VStacklop
from pyproximal import L21, L2
import matplotlib.pyplot as plt
from strainprox.utils import *
from pyproximal.optimization.segmentation import Segment
from pyproximal.optimization.primaldual import PrimalDual

def strain_jis(d, Op, x0, ui,  dims, cl, 
               alpha, beta, delta, tau, mu, tau_seg=0.7,
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
    tau_seg : float, optional
        Step size for the primal variable in the segmentation solver. Default is 0.7.
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
    l1 = L21(ndim=len(dims), sigma=alpha)
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
            d1 = np.hstack([d.ravel(), -np.sqrt(2.*delta)*(np.sqrt(v).T).ravel() * ((ui[:, None] - cl).T).ravel()])
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
        v, vcl = Segment_(ui, cl, 2 * delta, 2 * beta, tau_seg=tau_seg, z=(-beta * q if bregman else None),
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

    if utrue is None:
        return du, ui, v, vcl
    else:
        return du, ui, v, vcl, xerr, xsnr



def strain_jis_cp(d, Op, x0, ui,  dims, cl, 
               alpha, beta, delta, tau, mu, tau_seg=0.7,
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
    tau_seg : float, optional
        Step size for the primal variable in the segmentation solver. Default is 0.7.
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
    Dop = Gradient(dims=dims, edge=True, dtype='float64', kind='forward')
    l1 = L21(ndim=len(dims), sigma=alpha)
    v = cp.zeros(ncl * msize)

    if bregman:
        p = cp.zeros(msize)
        q = cp.zeros(ncl * msize)

    if utrue is not None:
        xhist, xsnr, xerr = [], [], []

    if plotflag:
        fig, axs = plt.subplots(2, niter, figsize=(4 * niter, 10))
    
    for iiter in range(niter):
        print('Iteration %d...' % iiter)
        ui_old, v_old = ui.copy(), v.copy()

        #############
        # Inversion #
        #############
        if iiter == 0:
            l2 = L2(Op=Op, b=d, niter=l2niter, warm=True, x0=cp.zeros(np.prod(dims)))
        
        else:
            # define misfit term
            v = v.reshape((msize, ncl))         
            L1op = VStacklop([Op] + [Diagonal(cp.sqrt(2.*delta)*cp.sqrt(v[:, icl])) for icl in range(ncl)])
            d1 = cp.hstack([d.ravel(), -cp.sqrt(2.*delta)*(cp.sqrt(v).T).ravel() * ((ui[:, None] - cl).T).ravel()])
            l2 = L2(Op=L1op, b=d1, niter=l2niter, warm=True, q=p if bregman else None, alpha=-alpha if bregman else None)

        # solve
        gu = Dop * cp.asnumpy(ui)
        l1prec = l1.precomposition(a=1., b=gu)
        l1prec.b = cp.asarray(l1prec.b)
        if utrue is not None:
            du = PrimalDual(proxf=l2, proxg=l1prec, A=Dop,
                    tau=tau, mu=mu, theta=1., x0=x0, niter=pdniter,
                callback=lambda xx:callbackx(xx.copy(), ui.copy(), utrue.ravel(), xhist, xsnr, xerr), 
                show=False)
        else:
            du = PrimalDual(proxf=l2, proxg=l1prec, A=Dop, 
                tau=tau, mu=mu, theta=1., x0=x0, niter=pdniter,
                show=show)
        ui += du

        if bregman:
            l2_grad = L2(Op=(Op if iiter == 0 else L1op), b=(d.ravel() if iiter == 0 else d1))
            p -= cp.real((1. / alpha) * l2_grad.grad(du))

        if plotflag:
            ui_np = cp.asnumpy(ui).real.reshape(dims)
            if niter==1:
                axs[0].imshow(ui_np, 'gray')
                axs[0].axis('tight')
            else:
                axs[0, iiter].imshow(ui_np, 'gray')
                axs[0, iiter].axis('tight')

        ################
        # Segmentation #
        ################
        v, vcl = Segment_(ui, cl, 2 * delta, 2 * beta, tau_seg=tau_seg, z=(-beta * q if bregman else None),
                        niter=segmentniter, callback=None, show=show,
                        kwargs_simplex=dict(maxiter=bisectniter, call=False, engine='cuda'))

        # Update q
        if bregman:
            q -= (delta / beta) * ((ui.ravel() - cl[:, np.newaxis]) ** 2).ravel()

        if plotflag:
            vcl_np = cp.asnumpy(vcl).reshape(dims)
            if niter==1:
                axs[1].imshow(vcl_np, 'gray')
                axs[1].axis('tight')
            else:    
                axs[1, iiter].imshow(vcl_np, 'gray')
                axs[1, iiter].axis('tight')

        # Monitor cost functions
        v_diff = float(cp.linalg.norm(v.ravel() - v_old.ravel()))
        u_diff = float(cp.linalg.norm(ui.ravel() - ui_old.ravel()))
        print('f=', L2(Op=Op, b=d.ravel())(ui))
        print('||v-v_old||_2=', v_diff)
        print('||m-m_old||_2=', u_diff)

        if v_diff < tolstop:
            break

    if utrue is None:
        return du, ui, v, vcl
    else:
        return du, ui, v, vcl, xerr, xsnr

#################################################################3
# test

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from pylops import BlockDiag, Gradient
from pylops.utils.typing import NDArray

from pyproximal import L21, Simplex, VStack
from pyproximal.optimization.primaldual import PrimalDual


def Segment_(
    y: NDArray,
    cl: NDArray,
    sigma: float,
    alpha: float,
    tau_seg: float,
    clsigmas: Optional[NDArray] = None,
    z: Optional[NDArray] = None,
    niter: int = 10,
    x0: Optional[NDArray] = None,
    callback: Optional[Callable[[NDArray], None]] = None,
    show: bool = False,
    kwargs_simplex: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArray, NDArray]:
    r"""Primal-dual algorithm for image segmentation

    Perform image segmentation over :math:`N_{cl}` classes using the
    general version of the first-order primal-dual algorithm [1]_.

    Parameters
    ----------
    y : :obj:`np.ndarray`
        Image to segment (must have 2 or more dimensions)
    cl : :obj:`numpy.ndarray`
        Classes
    sigma : :obj:`float`
        Positive scalar weight of the misfit term
    alpha : :obj:`float`
        Positive scalar weight of the regularization term
    tau_seg : :obj:`float`
        Step size for the primal variable in the segmentation Primal-Dual solver
    clsigmas : :obj:`numpy.ndarray`, optional
        Classes standard deviations
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    x0 : :obj:`numpy.ndarray`, optional
        Initial vector
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    kwargs_simplex : :obj:`dict`, optional
        Arbitrary keyword arguments for
        :py:func:`pyproximal.Simplex` operator

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Classes probabilities. This is a vector of size :math:`N_{dim} \times
        N_{cl}` whose columns contain the probability for each pixel to be in
        the class :math:`c_i`
    cl : :obj:`numpy.ndarray`
        Estimated classes. This is a vector of the same size of the input data
        ``y`` with the selected classes at each pixel.

    Notes
    -----
    This solver performs image segmentation over :math:`N_{cl}` classes solving
    the following nonlinear minimization problem using the general version of
    the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} \frac{\sigma}{2} \mathbf{x}^T \mathbf{f} +
        \mathbf{x}^T \mathbf{z} + \frac{\alpha}{2}||\nabla \mathbf{x}||_{2,1}

    where :math:`X=\{ \mathbf{x}: \sum_{i=1}^{N_{cl}} x_i = 1,\; x_i \geq 0 \}`
    is a simplex and :math:`\mathbf{f}=[\mathbf{f}_1, ...,
    \mathbf{f}_{N_{cl}}]^T` with :math:`\mathbf{f}_i = |\mathbf{y}-c_i|^2/\sigma_i`.
    Here :math:`\mathbf{c}=[c_1, ..., c_{N_{cl}}]^T` and
    :math:`\mathbf{\sigma}=[\sigma_1, ..., \sigma_{N_{cl}}]^T` are vectors
    representing the optimal mean and standard deviations for each class.

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    kwargs_simplex = {} if kwargs_simplex is None else kwargs_simplex

    dims = y.shape
    ndims = len(dims)
    dimsprod = np.prod(np.array(dims))
    ncl = len(cl)

    # Data (difference between image and center of classes)
    g = sigma / 2.0 * (y.reshape(1, dimsprod) - cl[:, np.newaxis]) ** 2
    if clsigmas is not None:
        g /= clsigmas[:, np.newaxis]
    g = g.ravel()

    # Gradient operator
    sampling = 1.0
    Gop = Gradient(
        dims=dims, sampling=sampling, edge=False, kind="forward", dtype="float64"
    )
    Gop = BlockDiag([Gop] * ncl)

    # Simplex and L21 proximal operators
    simp = Simplex(
        dimsprod * ncl, radius=1, dims=(ncl, dimsprod), axis=0, **kwargs_simplex
    )
    l21 = VStack(
        [L21(ndim=ndims, sigma=0.5 * alpha)] * ncl, nn=[ndims * dimsprod] * ncl
    )

    # Steps
    L = 8.0 / sampling**2
    tau = tau_seg
    mu = 1.0 / (tau * L)

    # Inversion
    x: NDArray = PrimalDual(
        simp,
        l21,
        Gop,
        tau=tau,
        mu=mu,
        z=g if z is None else g + z,
        theta=1.0,
        x0=np.zeros_like(g) if x0 is None else x0,
        niter=niter,
        callback=callback,
        show=show,
        returny=False,
    )
    x = x.reshape(ncl, dimsprod).T
    cl = np.argmax(x, axis=1)
    cl = cl.reshape(dims)

    return x, cl