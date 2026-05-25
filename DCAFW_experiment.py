from cmath import e
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple
import array
import copy
Array = np.ndarray

def set_thesis_plot_style():
    plt.rcParams.update({
        "figure.figsize": (2.2, 1.65),
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "lines.linewidth": 0.9,
    "lines.markersize": 2.2,
    })                              #recommendated by Chatgpt for the plot format

@dataclass
class Finite_Sum_DC_Problem:
    n: int
    H : object
    grad_H: object
    g_i: object
    grad_g_i: object
    lmo: object
    phi: object
    grad_phi: object
    grad_phi_i: object
    full_grad_G: object
    diameter: object

def batch_schedule(t:int, b_0:int = 64, p:float = 2+1e-5) -> int:         # the inceasing batch size b_t used for this experiment
    return int(np.ceil(b_0 * (t+1)**p))

@dataclass
class DCAFW:
    epsilon = 1e-9          # used to define the K_max
    T: int = 40             # the outer iterations
    b_0: int = 50           # used to define b_t as multiplier
    p: float = 2+1e-5       # used to define b_t as power
    L_H: float = 1.0        # the smoothness constant of H
    batch_schedule : Callable[[int, int, float], int] = lambda t,b_0,p: batch_schedule(t,b_0=b_0, p=p)   # the batch size schedule, defined as a function of t, b_0 and p
    stop_tolerance :float = 1e-12        # used for stopping criterion, for safety but not necessary
    seed_1: int = 14        # 
    seed_2: int =15


@dataclass
class SFW:
    T: int = 1000            # iterations for SFW
    batch_size: int = None  # in Reddi theorem 2: b = T
    gamma: float = None    # in Reddi theorem 2: gamma can be set to \sqrt{1/T}
    seed : int = 104


#-------------------
# SDCAFW algorithm with the first stopping criterion
#-------------------

def dcafw_algorithm1(prob:Finite_Sum_DC_Problem, x0:Array, cfg: DCAFW) -> Dict[str,Any]:
    rng = np.random.default_rng(cfg.seed_1)
    x = np.array(x0,dtype=float, copy=True)
    Kmax= int(np.ceil((8*cfg.L_H*prob.diameter*prob.diameter/cfg.epsilon)+1))   #define k_max using the setting in the paper

    lmo_calls = 0
    def lmo_call_counting(v:Array) -> Array:
        nonlocal lmo_calls            #used to count the total LMO calls
        lmo_calls += 1
        return prob.lmo(v)
    
    hist: Dict[str,list[Any]]={
        "time": [],
        "x": [],
        "b_t": [],
        "inner_steps": [],
        "stop_gap": [],
        "stop_rhs": [],
        "phi_value": [],
        "phi_fw_gap": [],
        "lmo_calls": [],
    }
    metric_time=0.0
    start= time.perf_counter()            # put start time outside the loop to record the cumulative running time of the algorithm
    phi_value = np.zeros(cfg.T)
    last_gap = np.zeros(cfg.T)             #we record frank-wolfe gap at stopping iteration for each outer iteraiont t
    last_rhs = np.zeros(cfg.T)             #we record the RHS of the stopping criterion at stopping iteration for each outer iteraiont t

    for t in range(cfg.T):
        b_t = min(cfg.batch_schedule(t, cfg.b_0, cfg.p), prob.n)   #define the increasing batch size following the recommendation from the paper and hope it not exceed n, which is the total number of samples in the finite sum problem
        idx = rng.integers(0, prob.n, size=b_t)     #sample a batch of indices with replacement

        s_t = np.zeros_like(x)             #define the gradient estimator s_t as the sample average of the gradients of g_i at x for the sampled indices
        for i in idx:
            s_t += prob.grad_g_i(x, i)
        s_t /= b_t

        H_xt=prob.H(x)             #compute the function value of H at x_t
        x_t = x.copy()             #receive the x_{t-1} from the last inner loop as the initial point for the inner solver in this inner loop
        zk = x.copy()              #define the inner loop warm-stating, using z_0=x_{t-1}
        inner_k = 0                #used to count the inner iterations, reset for each outer iteration t
        for k in range(Kmax):
            if k == Kmax-1:
                break
            grad_con_surr = prob.grad_H(zk) - s_t     #compute the gradient of the convex surrogate at x_t 
            vk = lmo_call_counting(grad_con_surr)      #count LMO and solve LMO

            d = vk - zk
            d2 = float(np.dot(d, d))
            gap = float(np.dot(grad_con_surr,-d))    #compute the FW gap at xk, the LHS of the stopping criterion
            short_step_criterion = float(cfg.L_H * d2)     #the denominator term in the short step expreesion

            #now we use gap and short_step_criterion to determine the short step value being gap/short_step_criterion or 1

            if gap >= short_step_criterion:
                gamma = 1.0
            else:
                gamma = float(gap / short_step_criterion)
            
            if d2 <= 1e-18:     # the fw_gap can be very small, we are interested in this fw_gap
                print("fwgap =", f"{gap:.15e}")
            
            rhs = float(H_xt - prob.H(zk) + np.dot(s_t, zk-x_t))
            rhs_with_tolerance = rhs + cfg.stop_tolerance     #adding tolerance to the rhs for safety, just a numerical trick, not totally consistent with criterion definitio nin the paper, but it does not change the nature of the stopping criterion

            inner_k += 1
            last_gap[t],last_rhs[t] = gap, rhs     #record the both side at stopping criterion

            if rhs < 0:                           # safety check, but with short step, it should not happen
                if gap <= cfg.stop_tolerance:
                    break
            
            else:                       
                if gap <= rhs_with_tolerance:     #the stopping criterion checking!!
                    break
            
            zk = zk + gamma * d      #update using short step
            
        x = zk

        t_0 = time.perf_counter()
        phi_value[t] = prob.phi(x)        #compute the phi value at x_t for this outer iteration t
        grad_phi = prob.grad_H(x) - prob.full_grad_G(x)     
        v_phi = prob.lmo(grad_phi)
        phi_fw_gap = float(np.dot(grad_phi, x- v_phi))     #calculate the true fw gap for phi at each outer iterations x_t except x_0
        metric_time += time.perf_counter() - t_0           #record the time for computing the metric, which should not appear in plotting the running time for algorithm
        
        hist["time"].append(time.perf_counter() - start - metric_time)     #record the cumulative running time of the algorithm, excluding the time for computing the metric
        hist["x"].append(x.copy())
        hist["b_t"].append(b_t)
        hist["inner_steps"].append(inner_k)
        hist["stop_gap"].append(last_gap[t])
        hist["stop_rhs"].append(last_rhs[t])
        hist["phi_value"].append(phi_value[t])
        hist["phi_fw_gap"].append(phi_fw_gap)
        hist["lmo_calls"].append(lmo_calls)

        print(f"[t={t:03d}] bt={b_t:<6d} inner={inner_k:<4d} "
                   f"LMO={lmo_calls:<7d}  stop_gap={last_gap[t]:.3e} rhs={last_rhs[t]:.3e} Phi={phi_value[t]:.10f} Phi_gap={phi_fw_gap:.3e}")
    return hist

#-------------------
# SDCAFW algorithm with the second stopping criterion
#-------------------

def dcafw_algorithm2(prob:Finite_Sum_DC_Problem, x0:Array, cfg: DCAFW) -> Dict[str,Any]:
    rng = np.random.default_rng(cfg.seed_2)
    x = np.array(x0,dtype=float, copy=True)
    Kmax= int(np.ceil((8*cfg.L_H*prob.diameter*prob.diameter/cfg.epsilon)+1))   

    lmo_calls = 0
    def lmo_call_counting(v:Array) -> Array:
        nonlocal lmo_calls            
        lmo_calls += 1
        return prob.lmo(v)
    
    hist: Dict[str,list[Any]]={
        "time": [],
        "x": [],
        "b_t": [],
        "inner_steps": [],
        "stop_gap": [],
        "stop_rhs": [],
        "best_inner_gap": [],
        "phi_value": [],
        "phi_fw_gap": [],
        "lmo_calls": [],
    }
    metric_time=0.0
    start= time.perf_counter()
    phi_value = np.zeros(cfg.T)          
    last_gap = np.zeros(cfg.T)             
    last_rhs = np.zeros(cfg.T)  
    best_inner_gap = np.zeros(cfg.T)      # we record the best gap inside the inner loop for each outer iteration t         

    for t in range(cfg.T):
        b_t = min(cfg.batch_schedule(t, cfg.b_0, cfg.p), prob.n)   
        idx = rng.integers(0, prob.n, size=b_t)     

        s_t = np.zeros_like(x)             
        for i in idx:
            s_t += prob.grad_g_i(x, i)
        s_t /= b_t

        H_xt=prob.H(x)             
        x_t = x.copy()             
        zk = x.copy()              #warm-starting for inner loop
        inner_k = 0

        best_gap_t = np.inf        # we record the best gap inside the inner loop
        best_zk_t = zk.copy()        # we record the zk corresponding to the best gap inside the inner loop
        best_k_t = 0                  # we record the inner iteration k corresponding to the best gap inside the inner loop
        hit_the_kmax= False             # we check whether we hit the inner iteration limitation kmax
        
        for k in range(Kmax):
            grad_con_surr = prob.grad_H(zk) - s_t     
            vk = lmo_call_counting(grad_con_surr)      

            d = vk - zk
            d2 = float(np.dot(d, d))
            gap = float(np.dot(grad_con_surr,-d))    
            short_step_criterion = float(cfg.L_H * d2)     

            if gap >= short_step_criterion:
                gamma = 1.0
            else:
                gamma = float(gap / short_step_criterion)      

            is_best_gap_so_far = (gap <= best_gap_t)     # check whether the current gap is the best gap

            if k == 0:                   # record the best_everything at the first inner iteration
                best_gap_t = gap
                best_k_t = k
                #best_zk_t = zk.copy()  already use initial point to define best_zk_t
            else:
                if is_best_gap_so_far:     # update the best_everything if we find a better gap
                    best_gap_t = gap
                    best_k_t = k
                    best_zk_t = zk.copy()
            if k < Kmax-1:
                if d2 <= 1e-18:         # same as in algorithm 1, we are interested in the fw_gap when it is very small
                    print("fwgap =", f"{gap:.15e}")
                
                rhs = float(H_xt - prob.H(zk) + np.dot(s_t, zk-x_t))
                rhs_with_tolerance = rhs + cfg.stop_tolerance
                inner_k += 1
                last_gap[t],last_rhs[t],best_inner_gap[t] = gap, rhs, best_gap_t  # record all informations for stopping criterion 2

                if rhs < 0:
                    stop = (gap <= cfg.stop_tolerance)
                else:
                    stop = (gap <= rhs_with_tolerance) and is_best_gap_so_far   # the stopping criterion 2, we only stop when the gap is smaller than the rhs and it is the best gap we have seen in this inner loop

                if stop:
                    break
                else:
                    zk = zk + gamma * d      #update using short step
                
            else:
                hit_the_kmax = True
        if hit_the_kmax:               # we output the best_z_k_t if we hit the kmax, following the thesis
            print(f"warning: hit Kmax={Kmax} at t={t}, best_gap_t={best_gap_t:.3e},at k={best_k_t}")
            x = best_zk_t
        else:
            x = zk
        
        t_0 = time.perf_counter()
        phi_value[t] = prob.phi(x)         # doing the same metric recording as in algorithm 1
        grad_phi = prob.grad_H(x) - prob.full_grad_G(x)     
        v_phi = prob.lmo(grad_phi)
        phi_fw_gap = float(np.dot(grad_phi, x- v_phi))    
        metric_time += time.perf_counter() - t_0
        
        hist["time"].append(time.perf_counter() - start - metric_time)     #record the cumulative running time of the algorithm, excluding the time for computing the metric
        hist["x"].append(x.copy())
        hist["b_t"].append(b_t)
        hist["inner_steps"].append(inner_k)
        hist["stop_gap"].append(last_gap[t])
        hist["stop_rhs"].append(last_rhs[t])
        hist["phi_value"].append(phi_value[t])
        hist["phi_fw_gap"].append(phi_fw_gap)
        hist["lmo_calls"].append(lmo_calls)
        hist["best_inner_gap"].append(best_inner_gap[t])

        print(f"[t={t:03d}] bt={b_t:<6d} inner={inner_k:<4d} "
                   f"LMO={lmo_calls:<7d}  stop_gap={last_gap[t]:.3e} rhs={last_rhs[t]:.3e} Phi={phi_value[t]:.10f} Phi_gap={phi_fw_gap:.3e}")
    return hist

#-----------------------------
#  nonconvex SFW algorithm (reddi)
#-----------------------------

def sfw_reddi(prob:Finite_Sum_DC_Problem, x0:Array, cfg: SFW) -> Dict[str,Any]:
    rng = np.random.default_rng(cfg.seed)
    x_t = np.array(x0,dtype=float, copy=True)                #initialization
    T = int(cfg.T)
    b = int(cfg.batch_size) if cfg.batch_size is not None else T
    start = time.perf_counter()
    metric_time = 0.0

    gamma = float(cfg.gamma) if cfg.gamma is not None else float(1/np.sqrt(T))
    # in the paper, gamma is defined by gamma = sqrt(2(Phi(x_0)-\Phi(x^*))/(TLD^2\beta))
    # and beta is chosen by beta >= 2(Phi(x_0)-\Phi(x^*))/LD^2
    # for simplicity we choose beta = 2(Phi(x_0)-\Phi(x^*))/LD^2
    # then gamma = sqrt(1/T)

    lmo_calls = 0
    def lmo_call_counting(v:Array) -> Array:
        nonlocal lmo_calls            
        lmo_calls += 1
        return prob.lmo(v)
    
    hist: Dict[str,list[Any]]={
        "time": [],
        "x": [],
        "phi_value": [],
        "phi_fw_gap": [],
        "lmo_calls": [],
    }

    for t in range(T):
        idx= rng.integers(0,prob.n,size=b)
        grad_estimator = np.zeros_like(x_t)               # evaluate the batch size averaged gradient at x
        for i in idx:
            grad_estimator += prob.grad_phi_i(x_t,i)
        grad_estimator /= b
        
        v_t = lmo_call_counting(grad_estimator)
        x_t = x_t + gamma*(v_t-x_t)

    
        t_0= time.perf_counter()
        phi= prob.phi(x_t)
        grad_phi_true = prob.grad_phi(x_t)
        v_phi=prob.lmo(grad_phi_true)
        phi_fw_gap = float(np.dot(grad_phi_true,x_t-v_phi))
        metric_time +=time.perf_counter()-t_0


        hist["time"].append(time.perf_counter()-start-metric_time)
        hist["lmo_calls"].append(lmo_calls)
        hist["phi_value"].append(phi)
        hist["phi_fw_gap"].append(phi_fw_gap)
        hist["x"].append(x_t.copy())

        print(f"[SFW t={t:03d}] b={b:<6d} gamma={gamma:.3e} "
                    f"LMO={lmo_calls:<7d} Phi={phi:.10f} Phi_gap={phi_fw_gap:.3e} ")
    
    # since the paper output random iterate. we keep it here only for consistency and reference
    # for plotting, we still use each iterate x_t for comparison
    if len(hist["x"]) == 0:
        raise RuntimeError("no iterates x_t history")
    a = int(rng.integers(0,len(hist["x"])))
    hist["random_output"] =hist["x"][a].copy()
    return hist



#-------------
#  L1 ball LMO solver
#--------------

def lmo_l1_ball(c:Array, tau:float)-> Array:
    v = np.zeros_like(c)
    i = int(np.argmax(np.abs(c)))                 # find the maximal index i where |c_i|>= all other |c_j|
    if c[i] != 0.0:
        v[i]=-tau*np.sign(c[i])                   # the minimizer is v = -sign(c_i)*tau*e_i,where e_i is the unit vector
    return v


#-------------
# Plotting function (generated by chatgpt)
#-------------
def _prep_y_log(y: np.ndarray, eps: float = 1e-16) -> np.ndarray:        # was not used in this experiment, just to have the completeness
    y = np.asarray(y, dtype=float)
    y = np.where(np.isfinite(y), y, np.nan)
    return np.maximum(y, eps)  

def plot_metric_histories_vs_lmo_calls(
    histories,
    labels,
    key,
    title,
    ylabel,
    figure_id=None,
    logy=False,
):
    if len(histories) != len(labels):
        raise ValueError(f"len(histories)={len(histories)} but len(labels)={len(labels)}")

    if figure_id is not None:
        plt.figure(figure_id)
        plt.clf()
    else:
        plt.figure()

    styles = [
        dict(marker="o", linestyle="-"),
        dict(marker="s", linestyle="--"),
        dict(marker="^", linestyle="-."), 
        dict(marker="d", linestyle=":"),
        dict(marker="x", linestyle="-"),
    ]

    for i, (hist, label) in enumerate(zip(histories, labels)):
        y = np.array(hist[key], dtype=float)
        x = np.array(hist["lmo_calls"], dtype=float)

        if logy:
            y = _prep_y_log(y)

        st = styles[i % len(styles)]
        plt.plot(x, y, label=label, linewidth=0.9, alpha=0.9, markevery=1, **st)

    if logy:
        plt.yscale("log")

    plt.xlabel("Cumulative LMO calls")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

def plot_metric_histories_vs_time(
    histories,
    labels,
    key,
    title,
    ylabel,
    figure_id=None,
    logy=False,
):
    if len(histories) != len(labels):
        raise ValueError(f"len(histories)={len(histories)} but len(labels)={len(labels)}")

    if figure_id is not None:
        plt.figure(figure_id)
        plt.clf()
    else:
        plt.figure()

    styles = [
        dict(marker="o", linestyle="-"),
        dict(marker="s", linestyle="--"),
        dict(marker="^", linestyle="-."),
        dict(marker="d", linestyle=":"),
        dict(marker="x", linestyle="-"),
    ]

    for i, (hist, label) in enumerate(zip(histories, labels)):
        y = np.array(hist[key], dtype=float)
        x = np.array(hist["time"], dtype=float)

        if logy:
            y = _prep_y_log(y)

        st = styles[i % len(styles)]
        plt.plot(x, y, label=label, linewidth=2, alpha=0.9, markevery=1, **st)

    if logy:
        plt.yscale("log")

    plt.xlabel("Running time (seconds)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
   
    plt.legend(
    fontsize=4.8,
    framealpha=0.9,
    borderpad=0.2,
    labelspacing=0.2,
    handlelength=1.3,
    handletextpad=0.35,
    markerscale=0.75,)

#---------------------
#-- we define a initial point for all algorithms
#---------------------
x_ini = array.array('d', [
    3.71895994e-03, -9.09642658e-03, -4.49240667e-03, 9.88997075e-03,
    -1.12588500e-02, 1.65077477e-03, -4.80416555e-03, -3.86043540e-04,
    9.20523126e-03, -2.22313367e-03, -4.33098799e-03, -1.03476349e-02,
    2.77810857e-03, -8.27889085e-03, -1.56292305e-02, 4.85894318e-03,
    6.46501625e-03, -1.57461579e-01, 2.39361935e-02, 1.44536983e-02,
    4.42038669e-04, -1.21434715e-02, 2.51155854e-02, 5.15099690e-03,
    7.65060051e-03, -1.55190064e-01, -4.25961801e-04, 8.34759903e-03,
    2.16078866e-02, -5.18532637e-03, 7.70902072e-03, -3.99418190e-03,
    7.06472018e-03, -3.43776234e-03, -8.96184722e-03, 1.13477207e-02,
    -1.70907376e-02, -9.66191512e-03, 1.03664011e-03, 1.71697594e-03,
    1.04820637e-02, -6.56306856e-05, 1.46769485e-02, 1.68184026e-03,
    -3.44048065e-03, 3.85338150e-03, -9.63065101e-03, -4.54261962e-03,
    -9.47220858e-03, -5.28121459e-03, -1.42133836e-02, -3.58814449e-03,
    -5.29145401e-03, -1.01949232e-02, 1.04693563e-02, -1.10663135e-02,
    4.60000929e-03, 2.29976954e-02, -2.58985636e-04, -7.48311250e-04,
    7.79849896e-04, 2.17335013e-03, -6.28242200e-03, 7.46996091e-03,
    6.20470887e-03, -1.07502299e-02, 2.49825502e-03, 7.48775809e-03,
    1.83060015e-02, 9.06901634e-03, 1.00132112e-02, 2.95992624e-03,
    6.98582470e-03, 1.99902133e-02, 9.01943895e-04, -6.08776654e-03,
    -1.25534584e-02, -8.77501382e-03, -1.52189651e-02, 2.03543528e-02,
    2.99812310e-03, -1.85523765e-02, 1.44040088e-02, -1.52330755e-03,
    -8.93473593e-03, -1.25604600e-02, 1.31514065e-03, -1.02090190e-01,
    -1.37063783e-02, -1.88076394e-02, -3.10804170e-03, -1.46717313e-02,
    2.01627583e-03, -4.81493410e-03, -1.32887230e-03, 1.41403522e-02,
    1.69694983e-02, -1.77927324e-04, 1.06239311e-02, -8.65468911e-03,
    -8.24639519e-03, 2.81705922e-03, 1.91965218e-02, -8.89247580e-03,
    4.30166156e-03, -7.76522377e-03, 1.55657659e-04, -7.65224253e-03,
    7.63697356e-04, -5.91547936e-03, 3.62318730e-03, 6.06528189e-01,
    -1.20923576e-03, 6.86611289e-03, 3.58328015e-03, 3.63333361e-04,
    -1.03975133e-02, 6.63301824e-03, -4.29502608e-03, -8.08835230e-03,
    9.98196856e-03, -4.70986687e-03, -5.48313855e-01, 1.56095372e-02,
    3.93507549e-03, 3.62348030e-03, 4.60815015e-03, -6.40934507e-03,
    -1.84469634e-02, -2.54001326e-02, -9.13738893e-03, -1.01574920e-01,
    -1.95492118e-03, -1.22474731e-02, 2.00561394e-03, -7.99124990e-03,
    1.03827638e-02, -3.10564337e-02, 7.15263605e-03, 1.44950606e-02,
    1.12476971e-02, 1.08096826e-02, -1.32700715e-02, 1.32824572e-03,
    -6.72680195e-04, 1.26179883e-03, -4.34843590e-03, -1.23362550e-02,
    -6.85835721e-03, 2.88846745e-03, -1.03325190e+00, 4.33903306e-04,
    -1.72456461e-04, -1.21030039e-02, -3.90017643e-03, 2.87133137e-02,
    6.90646759e-04, -2.58997793e-03, -1.71524389e-02, -3.30399779e-03,
    -6.89110728e-03, -8.98665313e-01, -2.96883983e-03, -4.76896303e-03,
    8.55812086e-03, 5.87537668e-03, 7.88944084e-03, -6.23368857e-04,
    -1.52852481e-03, -1.17592038e-03, -1.51049289e+00, -1.51681568e-01,
    -4.60490946e-04, -8.14619316e-03, -1.52055257e+00, -1.55563375e-02,
    5.24473095e-03, 1.95660060e-02, 3.68184381e-03, -3.56079425e-03,
    9.69331643e-01, -8.57088438e-03, 8.69496008e-03, 1.80277008e-03,
    -1.15468093e-02, 1.21556142e-02, 9.56581798e-03, 3.73852652e-03,
    -2.59335107e-03, -4.37704858e-02, 3.80357274e-03, -5.90979907e-03,
    1.11387688e-02, 1.27531524e-03, -1.09608238e-02, -3.23099641e-03,
    -9.28925881e-03, 6.19589323e-05, -8.34759961e-03, -6.59965880e-03
])   # for radius of l1 ball = 30, d= 200, x_ini is inside the feasible set.


#----------------------------
# sparse PCA with cauchy penalty problem
#----------------------------

def sparse_pca_cauchy_pen_problem(
        Z: Array,
        lam:float,
        delta: float,
        tau: float,
        decomposition: str = "v1",
        seed_x0: int = 0
) -> Tuple[Finite_Sum_DC_Problem,Array,Dict[str,float]] :
    n,d = Z.shape
    Sigma = np.dot(Z.T,Z)/ n
    L_averaged_variance = float(np.max(np.linalg.eigvalsh(Sigma)))       # we calculate the smooth constant of -(1/(2n)) ||Zx||^2
    L_penalty = float(2.0*lam/(delta**2))                                #smooth constant of lambda * sum_j log(1 + x_j^2 / delta^2)
    L_phi = float(L_averaged_variance + L_penalty)

    def lmo(c:Array)->Array:
        return lmo_l1_ball(c,tau=tau)
    
    #we need to manually disable one of the initialization x_0 choices
    #x_0 = x_ini

    rng0= np.random.default_rng(seed_x0)     # the random initialization that was used as the second part of experiment 1 in the thesis!
    u = rng0.normal(size=d)
    x_0=(tau/(np.sum(np.abs(u))+1e-3))*u
    
    def penalty(x:Array)->float:               # cauchy penalty term lambda * sum_j log(1 + x_j^2 / delta^2)
        return float(lam*np.sum(np.log1p(x*x/delta**2)))
    
    def grad_penalty(x:Array)->Array:          # gradient of cauchy penalty term (.....,2*lam*x_j/(delta^2+x_j^2),.....)
        return 2*lam*(x/(delta**2+x * x))
    
    def phi(x:Array)->float:                   #Phi(x) = -(1/(2n)) ||Zx||^2 + lambda * sum_j log(1 + x_j^2 / delta^2)
        Z_times_x = np.dot(Z,x)
        neg_averaged_variance = -(1/2)* float(np.dot(Z_times_x,Z_times_x)/n)
        return neg_averaged_variance + penalty(x)
    
    def grad_phi(x:Array)->Array:              # gradient of Phi(x)
        return -(np.dot(Sigma,x)) + grad_penalty(x)
    
    def grad_phi_i(x:Array,i:int)->Array:      # first term of phi can be written as (1/n) * sum_i (1/2)*(z_i^T*x)^2 = （1/n)*sum_i phi_i, gradient of phi is z_i^T*z_i*x
        z_i= Z[i,:]       #i-th row of Z
        return -(float(np.dot(z_i,x))*z_i) + grad_penalty(x)

    if decomposition == "v1":      # the first decomposition in the thesis
         # H_v1(x) = lam/delta^2 ||x||^2
         # g_i(x) = 1/2 (z_i^T x)^2 + lam/delta^2 ||x||^2 - penalty(x)
         # by the formulars above we construct each term in finite-sum-dc-problem class
        def H(x:Array)->float:
            return float((lam/(delta**2))*np.dot(x,x))
        
        def grad_H(x:Array)->Array:
            return 2*(lam/(delta**2))*x
        
        def g_i(x:Array,i:int)->float:
            z_i= Z[i,:]       
            return 0.5*(float(np.dot(z_i,x)**2)) + H(x) - penalty(x)
        
        def grad_g_i(x:Array,i:int)->Array:
            z_i= Z[i,:]       
            return float(np.dot(z_i,x))*z_i + grad_H(x) - grad_penalty(x)
        
        def full_grad_G(x:Array)->Array:
            gradient = np.zeros_like(x)
            for i in range(n): 
                gradient += grad_g_i(x,i)    
            return gradient/n
        
        L_smooth_H = L_penalty        # the smooth constant of H(x) is the same as the smooth constant of penalty(x)

    elif decomposition == "v2":     # the second decomposition in the thesis
        # H_v2(x) = (L_phi/2) ||x||^2
        # g_i(x) = 1/2 (z_i^T x)^2 + (L_phi/2)||x||^2 - penalty(x)
        def H(x:Array)->float:
            return float((L_phi/2)*np.dot(x,x))
        
        def grad_H(x:Array)->Array:
            return L_phi*x
        
        def g_i(x:Array,i:int)->float:
            z_i= Z[i,:]       
            return 0.5*(float(np.dot(z_i,x)**2)) + (L_phi/2)*np.dot(x,x) - penalty(x)
        
        def grad_g_i(x:Array,i:int)->Array:
            z_i= Z[i,:]       
            return float(np.dot(z_i,x))*z_i + L_phi*x - grad_penalty(x)
        
        def full_grad_G(x:Array)->Array:
            gradient = np.zeros_like(x)
            for i in range(n): 
                gradient += grad_g_i(x,i)    
            return gradient/n
        L_smooth_H = L_phi        # smooth constant of H in this decomposition is L_phi
    
    else:
        raise ValueError("decomposition must be either 'v1' or 'v2'")
    
    info = {
        "n": n,
        "d" : d,
        "lam": lam,
        "delta": delta,
        "tau": tau,
        "L_averaged_variance": L_averaged_variance,
        "L_penalty": L_penalty,
        "L_phi": L_phi,
        "diameter": float(2.0*tau),
        "L_smooth_H": L_smooth_H
    }
    
    prob = Finite_Sum_DC_Problem(
        n=n,
        H=H,
        grad_H=grad_H,
        g_i=g_i,
        grad_g_i=grad_g_i,
        lmo=lmo,
        phi=phi,
        grad_phi=grad_phi,
        grad_phi_i=grad_phi_i,
        full_grad_G=full_grad_G,
        diameter=float(2.0*tau)
    )
    
    return prob,x_0,info

#--------------------
# run the experiment
#--------------------

def run_experiment(
    n: int = 300000,
    d: int = 200,
    lam: float = 0.5,
    delta: float = 0.2,
    tau: float = 30.0,
    cfg: Optional[DCAFW]= None,
    sfw_cfg: Optional[SFW] = None,
    seed_data: int = 1000,           #103                # the seed for matrix Z generation   103 for fixed x_ini, 1000 for random x_0
    seed_x0: int = 100,                              # the seed for random initialization of x0, (not for x_ini)
    seed_for_5_algorithms: int = [84,85,86,87,88],         # the seed for 5 algorithms: algothm1+v1, algorithm1+v2, algorithm2+v1, algorithm2+v2, SFW
) -> Tuple[Dict[str,Any],Dict[str,Any],Dict[str,Any],Dict[str,Any],Dict[str,Any]]:
    
    rng = np.random.default_rng(seed_data)
    Z = rng.normal(size=(n,d))
    Z = Z - Z.mean(axis=0,keepdims=True)     #columns are features, we do column-wise centering
  
    scale = 5.0   #5     # scale the matrix to make the principal component more obvious.
    Z = np.sqrt(scale)*Z

    prob1,x_0,info1 = sparse_pca_cauchy_pen_problem(Z, lam=lam, delta=delta, tau=tau, decomposition="v1",seed_x0=seed_x0)

    prob2,x_0b,info2= sparse_pca_cauchy_pen_problem(Z, lam=lam, delta=delta, tau=tau, decomposition="v2",seed_x0=seed_x0)
    # since we use the same seed_x0, here we have x_0 = x_0b.
    print("Parameters of the problem:")
    print(f"n={info1['n']} d={info1['d']} lam={lam} delta={delta} tau={tau}")
    print(f"L_averaged_variance={info1['L_averaged_variance']} L_penalty={info1['L_penalty']} L_phi={info1['L_phi']}")

    if cfg is None:
        cfg = DCAFW()
    if sfw_cfg is None:
        sfw_cfg = SFW()
    
    cfg1_alg1 = copy.deepcopy(cfg)         #algo 1 + v1
    cfg1_alg1.L_H= float(info1["L_smooth_H"])       # we overwrite the L_H in class
    cfg1_alg1.seed_1=seed_for_5_algorithms[0]        # we overwrite the corresponding seed for algorithm in class
    hist1= dcafw_algorithm1(prob1,x_0,cfg1_alg1)

    cfg2_alg1 = copy.deepcopy(cfg)        #algo 1 + v2
    cfg2_alg1.L_H= float(info2["L_smooth_H"])
    cfg2_alg1.seed_1=seed_for_5_algorithms[1]
    hist2= dcafw_algorithm1(prob2,x_0b,cfg2_alg1)

    cfg1_alg2 = copy.deepcopy(cfg)         #algo2 + v1
    cfg1_alg2.L_H= float(info1["L_smooth_H"])
    cfg1_alg2.seed_2=seed_for_5_algorithms[2]
    hist3= dcafw_algorithm2(prob1,x_0,cfg1_alg2)
 
    cfg2_alg2 = copy.deepcopy(cfg)        #algo2 + v2
    cfg2_alg2.L_H= float(info2["L_smooth_H"])
    cfg2_alg2.seed_2=seed_for_5_algorithms[3]
    hist4= dcafw_algorithm2(prob2,x_0b,cfg2_alg2)

    cfg_sfw = copy.deepcopy(sfw_cfg)
    cfg_sfw.seed=seed_for_5_algorithms[4]  

    hist_sfw = sfw_reddi(prob1,x_0,cfg_sfw)    #it doesnt matter we call prob1 or prob2, because sfw_reddi doesn't use the functions in if syntax for different decompositions    

    return hist1,hist2,hist3,hist4,hist_sfw


if __name__ == "__main__":

    h1, h2, h3, h4, h_sfw = run_experiment()

    RESULTS_DIR = Path(__file__).resolve().parent / "Plots_SDCFW"    # generated by chatgpt to store the results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"result_{timestamp}_{uuid.uuid4().hex[:6]}"

    labels_all = [
        "v1 + Alg1",
        "v2 + Alg1",
        "v1 + Alg2",
        "v2 + Alg2",
        "SFW",
    ]

    set_thesis_plot_style()
    
    # we plot two plots for each metric (function value vs time, true_fw_gap vs cumulative lmo calls)
    plot_metric_histories_vs_time(
        [h1, h2, h3, h4, h_sfw],
        labels_all,
        key="phi_value",
        title="Function value vs Running Time",
        ylabel=r"Function value $\Phi(x_t)$",
        figure_id=4,
        logy=False,
    )
    plt.figure(4)
    fname4 = RESULTS_DIR / f"{base_name}_function_value_vs_iterations.png"
    plt.tight_layout(pad=0.15)
    plt.savefig(fname4.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.savefig(fname4.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()

    plot_metric_histories_vs_lmo_calls(
        [h1, h2, h3, h4, h_sfw],
        labels_all,
        key="phi_fw_gap",
        title=r"Frank-Wolfe Gap of $\Phi$ vs LMO Calls",
        ylabel=r"FW gap",
        figure_id=5,
        logy=True,
    )
    plt.figure(5)
    fname5 = RESULTS_DIR / f"{base_name}_phi_fw_gap_vs_lmo_calls.png"
    plt.tight_layout(pad=0.15)
    plt.savefig(fname5.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.savefig(fname5.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()






