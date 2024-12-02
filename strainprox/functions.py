import time
import numpy as np
from pylops import Diagonal, Gradient, VStack as VStacklop
from pylops.utils.backend import get_array_module, to_numpy
from pyproximal import L21, L2
import matplotlib.pyplot as plt
from strainprox.utils import *
from pyproximal.optimization.bregman import *
from pyproximal.optimization.segmentation import *

def PrimalDual2(proxf, proxg, A, b, x0, tau, mu, 
                y0=None, z=None, theta=1., niter=10,
               gfirst=True, callback=None, callbacky=False, 
               returny=False, show=False):
    r"""Primal-dual algorithm

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    mu : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    theta : :obj:`float`
        Scalar between 0 and 1 that defines the update of the
        :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
        special case that represents the semi-implicit classical Arrow-Hurwicz
        algorithm
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
    returny : :obj:`bool`, optional
        Return also ``y``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Primal-dual algorithm can be expressed by the following recursion
    (``gfirst=True``):

    .. math::

        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k})\\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k+1} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k)

    where :math:`\tau \mu \lambda_{max}(\mathbf{A}^H\mathbf{A}) < 1`.

    Alternatively for ``gfirst=False`` the scheme becomes:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k) \\
        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k+1})

    .. [1] A., Chambolle, and T., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120-145. 2011.

    """
    ncp = get_array_module(x0)

    # check if tau and mu are scalars or arrays
    fixedtau = fixedmu = False
    if isinstance(tau, (int, float)):
        tau = tau * ncp.ones(niter, dtype=x0.dtype)
        fixedtau = True
    if isinstance(mu, (int, float)):
        mu = mu * ncp.ones(niter, dtype=x0.dtype)
        fixedmu = True

    if show:
        tstart = time.time()
        print('Primal-dual: min_x f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Additional vector (z): %s\n'
              'tau = %s\t\tmu = %s\ntheta = %.2f\t\tniter = %d\n' %
              (type(proxf), type(proxg), type(A),
               None if z is None else 'vector', str(tau[0]) if fixedtau else 'Variable',
               str(mu[0]) if fixedmu else 'Variable', theta, niter))
        head = '   Itn       x[0]          f           g          z^x       J = f + g + z^x'
        print(head)

    x = x0.copy()
    xhat = x.copy()
    y = y0.copy() if y0 is not None else ncp.zeros(A.shape[0], dtype=x.dtype)
    for iiter in range(niter):
        xold = x.copy()
        if gfirst:
            y = proxg.proxdual(y + mu[iiter] * (A.matvec(xhat) + b) , mu[iiter])
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter])
            xhat = x + theta * (x - xold)
        else:
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter])
            xhat = x + theta * (x - xold)
            y = proxg.proxdual(y + mu[iiter] * A.matvec(xhat), mu[iiter])

        # run callback
        if callback is not None:
            if callbacky:
                callback(x, y)
            else:
                callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(A.matvec(x))
                pf = 0. if type(pf) == bool else pf
                pg = 0. if type(pg) == bool else pg
                zx = 0. if z is None else np.dot(z, x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])), pf, pg, zx, pf + pg + zx)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    if not returny:
        return x
    else:
        return x, y
    
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
        Initial background model. Must have the same dimensions as `d`.
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
        If True, enables Bregman iterations to enhance convergence. Default is Flse.
    mtrue : np.ndarray, optional
        True model for comparison. If provided, metrics like RRE and PSNR are computed.
        Must have the same dimensions as `d`.
    plotflag : bool, optional
        If True, visualizes intermediate results for each iteration. Default is True.
    show : bool, optional
        If True, prints iteration logs for solvers. Default is False.

    Returns
    -------
    ui : np.ndarray
        Final inverted time-strains.
    vcl : np.ndarray
        Estimated segmentation (assigned class for each model point).
    rre : list or None
        Relative Reconstruction Error (RRE) for each iteration, if `mtrue` is provided.
    psnr : list or None
        Peak Signal-to-Noise Ratio (PSNR) for each iteration, if `mtrue` is provided.

    Notes
    -----
    - The algorithm alternates between inversion (updating `ui`) and segmentation 
      (updating `v` and `vcl`) for `niter` iterations.
    - The use of Bregman iterations (`bregman=True`) might in some cases improve 
      performance by recovering the constrast that usually loss in standard TV
      regularization.
    """

    print('Working with alpha=%f,  beta=%f,  delta=%f' % (alpha, beta, delta))
    
    msize = x0.size
    ncl = len(cl)

    # TV regularization term
    Dop = Gradient(dims=dims, edge=True, dtype='float32', kind='forward')
    l1 = L21(ndim=2, sigma=alpha)
    b = Dop * ui
    v = np.zeros(ncl * msize)

    if bregman:
        p = np.zeros(msize)
        q = np.zeros(ncl * msize)
    
    # u_hist = []
    # v_hist = []
    rre = psnr = None
    if utrue is not None:
        rre = np.zeros(niter)
        psnr = np.zeros(niter)

    if plotflag:
        fig, axs = plt.subplots(2, niter, figsize=(4 * niter, 10))

    for iiter in range(niter):
        print('Iteration %d...' % iiter)
        ui_old, v_old = ui.copy(), v.copy()

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
        du = PrimalDual2(l2, l1, Dop, b, x0=x0,
                         tau=tau, mu=mu, theta=1., niter=pdniter,
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

        # Monitor quality of reconstruction
        if utrue is not None:
            rre[iiter] = RRE(utrue.ravel(), ui.ravel())
            psnr[iiter] = SNR(utrue.ravel(), ui.ravel())
            print('RRE=', rre[iiter])
            print('PSNR=', psnr[iiter])

        # Check stopping criterion
        if np.linalg.norm(v.ravel()-v_old.ravel()) < tolstop:
            break

    return ui, vcl, rre, psnr


