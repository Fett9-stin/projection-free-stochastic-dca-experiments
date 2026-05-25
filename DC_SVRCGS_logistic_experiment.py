import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Iterable
Array = np.ndarray
import copy


#-----------------
#-- PLOT SETTING
#-----------------

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
    })


#---------------------------------
#-- central seeding manager
#---------------------------------
@dataclass
class SeedBank:
    """
    Central seed manager.
    - base_seed controls everything.
    - produces independent RNG streams for named components.
    - optional overrides let you pin specific parts.
    """
    base_seed: int
    names: Iterable[str] = ("data", "ncgs_vr", "prox_dc_vr_scgs", "fcgs",'dc_vr_scgs')
    overrides: Optional[Dict[str, int]] = None

    def __post_init__(self):
        self.names = tuple(self.names)
        ss = np.random.SeedSequence(self.base_seed)
        children = ss.spawn(len(self.names))

        # store seeds (uint32) + RNGs
        self.seeds: Dict[str, int] = {}
        self.rngs: Dict[str, np.random.Generator] = {}

        for name, child in zip(self.names, children):
            seed32 = int(child.generate_state(1, dtype=np.uint32)[0])
            self.seeds[name] = seed32
            self.rngs[name] = np.random.default_rng(seed32)

        # apply overrides if provided
        if self.overrides:
            for name, seed in self.overrides.items():
                self.seeds[name] = int(seed)
                self.rngs[name] = np.random.default_rng(int(seed))

    def rng(self, name: str) -> np.random.Generator:
        return self.rngs[name]

#-----------------------
#--LMO solver and projection functions
#------------------------

def projection_onto_l1_ball(s,tau=1.0):    
# used for sigunlar value vector projection in the next nuclar ball projection function
# we use soft-thresholding proj(s) = sign(s)*max(0,|s|-lambda) where sum(max(0,|s|-lambda))=tau
    s = np.asarray(s, dtype=np.float64)
    
    if tau<=0:      #trivial or non-exisiting case, for stability we output zero vector
        return np.zeros_like(s)
    
    s_abs = np.abs(s)
    
    if s_abs.sum() <= tau:
        return s.copy()
    ordered_s_abs = np.sort(s_abs)[::-1]           # lambda = (sum_1^j odered_s_abs[j]-tau)/j ,where j = max{j:ordered_s_abs[j]-(sum_1^j odered_s_abs[j]-tau)/j >0}
    cumulative_sum_vector = np.cumsum(ordered_s_abs)-tau
    index = np.arange(1,s_abs.size+1)
    candidates = cumulative_sum_vector/index
    
    acceptable_candidates = ordered_s_abs - candidates > 0

    if not np.any(acceptable_candidates):
        raise RuntimeError(f"no acceptable candidates found in projection_onto_l1_ball(s,tau={tau})")
    
    lam = candidates[acceptable_candidates][-1]      # find lambda, by choosing the last index such that accept_candidates[index]==True
    return np.maximum(s_abs-lam,0)*np.sign(s)


def lmo_l1_ball(c:Array, tau:float)-> Array:
    v = np.zeros_like(c)
    i = int(np.argmax(np.abs(c)))                 # find the maximal index i where |c_i|>= all other |c_j|
    if c[i] != 0.0:
        v[i]=-tau*np.sign(c[i])                   # the minimizer is v = -sign(c_i)*tau*e_i,where e_i is the unit vector
    return v


#--------------------------------
#-- Logistic Regression with Cauchy Penalty Problem
#-------------------------------

class LogisticCauchyPenaltyProblem:
    def __init__(self, P, y, lam=0.5,delta=0.2, L=None,margin=1e-3,tau=10.0):
        self.P = np.asarray(P,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        self.n ,self.d = self.P.shape
        self.lam = float(lam)
        self.delta=float(delta)
        self.tau=float(tau)
        self.margin = float(margin)

        if L is None:    # smooth constant of the penalty function
            L = 2.0*self.lam / (self.delta**2)+ self.margin
        self.L = float(L)

    def diameter(self):
        return 2.0*self.tau
    
    def sigmoid(self,z):
        z = np.asarray(z,dtype=np.float64)
        sigmoid_z = np.empty_like(z)
        positive = z>= 0 
        sigmoid_z[positive] = 1.0/(1.0+np.exp(-z[positive]))
        sigmoid_z[~positive] = np.exp(z[~positive])/(1.0+np.exp(z[~positive]))
        return sigmoid_z

    
    #----------- we recall the definition of H in the thesis
    #------------H(x) = 1/n sum_i log(1+exp(-yi pi^T x))+ L/2 \|x\|^2
    def H_value(self,x):
        x= np.asarray(x,dtype=np.float64)
        z= -(self.y*(self.P @ x))
        return float(np.mean(np.logaddexp(0.0,z))+0.5*self.L*np.dot(x,x))
    
    def H_full_grad(self,x):
        x= np.asarray(x,dtype=np.float64)
        z = -(self.y*(self.P @ x))
        s = self.sigmoid(z)
        g = -(self.P.T @ (self.y*s)) / self.n
        return g + self.L*x
    
    def H_batch_grad(self,x,idx):
        x= np.asarray(x,dtype=np.float64)
        P = self.P[idx]
        y = self.y[idx]
        z = -(y*(P @ x))
        s = self.sigmoid(z)
        g = -(P.T @ (y*s)) / len(idx)
        return g + self.L*x
    
    #---------------------
    #-----
    #------------------------

    def R_component(self,x):
        return np.log1p((x*x)/self.delta**2)
    
    def R_grad(self,x):
        return (2.0*x)/(self.delta**2+x*x)
    
    #---------------------
    #----
    #----------
    def G_value(self,x):
        x= np.asarray(x,dtype=np.float64)
        return float(0.5*self.L*np.dot(x,x)) - float(self.lam*np.sum(self.R_component(x)))
    
    def G_full_grad(self,x):
        x= np.asarray(x,dtype=np.float64)
        return self.L*x - self.lam*self.R_grad(x)
    
    def G_batch_grad(self,x,idx):
        x= np.asarray(x,dtype=np.float64)
        return self.G_full_grad(x)
    
    def phi(self,x):
        x= np.asarray(x,dtype=np.float64)
        return self.H_value(x) - self.G_value(x)
    
    def phi_full_grad(self,x):
        x= np.asarray(x,dtype=np.float64)
        return self.H_full_grad(x) - self.G_full_grad(x)
    
    def lipschitz_constants(self):
        L_logistic= 0.25*float(np.mean(np.sum(self.P *self.P,axis=1)))
        L_H = L_logistic + self.L
        L_G = self.lam*(2.0/(self.delta**2))+self.L
        return float(L_H),float(L_G),float(L_H+L_G)

def make_logistic_cauchy_pen_problem(n=800,d=100,scale=2.0,seed=0):
    rng= np.random.default_rng(seed)
    P = rng.standard_normal((n,d))
    P *= (scale/(np.linalg.norm(P,axis=1,keepdims=True)+1e-12))
    y = rng.choice([-1.0,1.0],size=n)
    return P,y



#-------------------------------
#-- Algorithms: Prox_dc_vr_scgs, dc_vr_scgs, NCGS_VR, FCGS
#-------------------------------
@dataclass
class Prox_dc_vr_scgs:
    T_outer = 20
    bB = 64
    mB = 30
    mu0 = 0.6
    mu_decay = 0.97
    delta0 = 1
    delta_decay = 0.5
    delta_min = 1e-8
    eta_scale = 0.7
    max_inner = 2500
    max_cndgB = 500
    check_gap_every = 10

@dataclass
class Dc_vr_scgs:
    T_outer = 20
    bB = 64
    mB = 30
    mu0 = 0.6
    mu_decay = 0.97
    delta0 = 1
    delta_decay = 0.5
    delta_min = 1e-8
    eta_scale = 0.7
    max_inner = 2500          #1200?
    max_cndgB = 500
    check_gap_every = 10

@dataclass
class NCGS_VR:
    T = 1500
    mA = None #int(np.ceil(n**(1/3))),
    bA = None #int(np.ceil(n ** (2/3))),
    lam = None   # 1/(3*L_phi)
    eta_ncgs = None  #1/T
    max_cndgA = 500
    log_everyA = 10


@dataclass
class FCGS:
    K= 1500
    qc= None
    bc = None
    eta_FCGS = None
    lam= None
    max_cndgC = 500
    log_everyC = 10


def ncgs_vr(problem, x0, cfg=NCGS_VR, rng=None, seed=1):
    rng = np.random.default_rng(seed) if rng is None else rng
    n= problem.n
    _,_, L_phi = problem.lipschitz_constants()
    if cfg.eta_ncgs is None:
        eta_ncgs = float(1/cfg.T)
    if cfg.lam is None:
        lam = 1/(3*L_phi)
    if cfg.mA is None:
        mA = int(np.ceil(n**(1/3)))
    if cfg.bA is None:
        bA = int(np.ceil(n ** (2/3)))
    
    x = x0.copy()

    times,function_value = [],[]
    start = time.perf_counter()
    metric_time = 0.0
    
    iteration = 0
    S = int(np.ceil(cfg.T / mA))      
    found = False

    for s in range(S):
        x_tilde = x.copy()
        full_grad_tilde = problem.phi_full_grad(x_tilde)
        for t in range(mA):
            if iteration >= cfg.T:
                found = True     # we set found also to break the outer loop s
                break
            idx = rng.integers(0,n,size=bA)
            v = (problem.H_batch_grad(x,idx)-problem.H_batch_grad(x_tilde,idx))
            v-= (problem.G_batch_grad(x,idx)-problem.G_batch_grad(x_tilde,idx))
            v+= full_grad_tilde
            x_new,cndg_info = cndg_l1(v,x,lam=lam,eta=eta_ncgs,tau=problem.tau,max_cndg=cfg.max_cndgA)

            if (t+1) % cfg.log_everyA == 0 or t==0:
                print(f"[NCGS_VR] t={t:5d} cndg_gap= {cndg_info['gap']:.3e} hit_max={cndg_info['hit_max']}")

            x = x_new
            iteration +=1
            if (t+1) % cfg.log_everyA == 0 or t==0:
                t0=time.perf_counter()
                phi_value = problem.phi(x)
                metric_time += time.perf_counter()-t0

                times.append(time.perf_counter()-start-metric_time)
                function_value.append(phi_value)
        if found:
            break
    info = {
        'lam': lam,
        'eta': eta_ncgs,
        'L_phi': L_phi,
        'mA': mA,
        'bA': bA,
        'T': cfg.T,
    }

    return x, np.array(times), np.array(function_value), info
                
                

def fcgs(problem, x0, cfg=FCGS, rng=None, seed=1):
    rng = np.random.default_rng(seed) if rng is None else rng
    n= problem.n
    _,_, L_phi = problem.lipschitz_constants()
    if cfg.eta_FCGS is None:
        eta_FCGS = float(1/cfg.K)
    if cfg.lam is None:
        lam = 1.0/(3.0*L_phi)
    if cfg.qc is None:
        qc = int(np.sqrt(n))
    if cfg.bc is None:
        bc = qc
    
    xk = x0.copy()
    x_previous = x0.copy()
    v = problem.phi_full_grad(xk)

    times, function_value = [],[]
    start = time.perf_counter()
    metric_time = 0.0
    
    for k in range(cfg.K):
        if k % qc == 0:
            v = problem.phi_full_grad(xk)
        else:
            idx = rng.integers(0,n,size=bc)
            v +=(problem. H_batch_grad(xk,idx)-problem.H_batch_grad(x_previous,idx))
            v -=(problem. G_batch_grad(xk,idx)-problem.G_batch_grad(x_previous,idx))
        
        x_new, cndg_info = cndg_l1(v, xk,lam=lam, eta=eta_FCGS, tau = problem.tau,max_cndg=cfg.max_cndgC)

        stopping_gap = cndg_info['gap']
        if (k+1) % cfg.log_everyC == 0 or k== 0:
            print(f"[FCGS]: k={k}, stopping_gap={stopping_gap}, hit_max={cndg_info['hit_max']}")
        
        x_previous, xk = xk, x_new

        if (k+1)% cfg.log_everyC == 0 or k== 0:
            t0= time.perf_counter()
            phi_value = problem.phi(xk)
            metric_time += time.perf_counter() - t0

            times.append(time.perf_counter() - start - metric_time)
            function_value.append(phi_value)

    info = {
        "lam" : lam,
        'L_phi': L_phi,
        "eta": eta_FCGS,
        'K': cfg.K,
        'qc': qc,
        'bc': bc
    }
    return xk, np.array(times), np.array(function_value), info


def prox_dc_vr_scgs(problem, x0 , cfg=Prox_dc_vr_scgs, rng=None, seed=1):
    rng = np.random.default_rng(seed) if rng is None else rng
    n = problem.n
    L_H, _, _ = problem.lipschitz_constants()
    L_sur = 0.0
    fw_gap_from_first_iteration = 0.0
    
    

    D = problem.diameter()

    x = x0.copy()
    times, function_value = [], []
    start = time.perf_counter()
    metric_time = 0.0
    
    
    for t in range(cfg.T_outer):
        mu_t = cfg.mu0 * (cfg.mu_decay ** t)      # we construct a decrasing squence mu_t = mu_0 * mu_decay^t
        L_sur = L_H + mu_t                # for hat{phi}_t(x)=H(x) - <nabla G(x_t),x-x_t> + mu_t||x-x_t||^2, L_sur = L_H + mu_t
           
        #delta_t = 1/((t+1)**2*(t+2)**2)
        delta_t= 1/((t+1)*(t+2))           # the choice in theorem 4.13 of the thesis
        
     
        
        t0 = time.perf_counter()                 #metric part, calculate the function values
        Phi_value = problem.phi(x)
        metric_time += time.perf_counter() - t0
        
        #stopping_at_2000 = time.perf_counter() - start - metric_time
        #if stopping_at_2000 >= 2000:
        #    print(f"Stopping at t={t} due to reaching time limit: {stopping_at_2000:.2f} seconds")
        #    break

        times.append(time.perf_counter() - start - metric_time)
        function_value.append(Phi_value)

        g_t = problem.G_full_grad(x)
        x_center = x.copy()

        delta_s=0.0
        x_inner = x.copy()         # x0 <---- x_t according to the algorithm
        y_inner = x_inner.copy()   # y0 <---- x_t

        found = False
        inner_it = 0
        epochs = int(np.ceil(cfg.max_inner / cfg.mB))   #theoretically we dont set this epochs bound, but for practice we do
        for s in range(epochs):
            if found==True:
                break
            '''if gap <= delta_min:  # gap \leq delta_t:
                break'''
            
            w_tilde = y_inner.copy()
            fullH_tilde = problem.H_full_grad(w_tilde)

            D_s_square = D**2*L_sur/ (mu_t*2**s)
            #N_s= int(np.ceil(2*np.sqrt(6*L_sur/mu_t)))
            N_s = int(np.ceil(np.sqrt(32*(L_sur/mu_t))))
            delta_s = max(cfg.delta_min, cfg.delta0 * (cfg.delta_decay ** s))
            for k in range(N_s):    #N_s -1 ????????
                
                #if inner_it >= N_s:#or gap <= delta_min:
                   # print(f"Stopping at inner_it={inner_it} due to reaching max_inner={max_inner},delta_t={delta_t:.3e}, gap={gap:.3e}")
                   # break

                # ---- CGS extrapolation point z_k = (1-gamma_k) y + gamma_k x ----
                gamma_k = 2.0 / (k + 2.0)  # in (0,1], gives gamma_1=1,... inner_it+2
                lam_t= (k+1.0) / (3.0 * L_sur)
                eta_k = 8.0 * L_sur * delta_s / (mu_t * N_s * (k+1.0))  
                #eta_k = 2*L_sur*D_s_square / (N_s * (k+1.0))
                z_k = (1.0 - gamma_k) * y_inner + gamma_k * x_inner
                

                idx = rng.integers(0, n, size=800)

                # SVRG for H, evaluated at z_k
                vH = (problem.H_batch_grad(z_k, idx) - problem.H_batch_grad(w_tilde, idx)) + fullH_tilde

                # Surrogate gradient estimator at z_k: ∇H(z_k) - g_t + μ (z_k - x_center)
                v = vH - g_t + mu_t * (z_k - x_center)
                
                x_new,fw_gap_from_first_iteration= cndg_l1_gap0_checking(v, x_inner,z=z_k, lam=lam_t, eta=eta_k , tau=problem.tau, max_cndg=cfg.max_cndgB)  
                ####### I add a z_k here, because I'm worried about the dual gap at the first iteration of CG should be at z_k,
                #####    otherwise it might look like fw_gap_from_f_i = \<v, x_inner - g_j\> is neither a gap at x_inner nor a gap at z_k, which is weird. 
                if k == N_s-1:
                    print(f"oh {s} {N_s} {fw_gap_from_first_iteration}")
        

                if fw_gap_from_first_iteration <= delta_t:
                    print(f"outer{t}, epoch {s}: current_N_s={N_s}")
                    print(f"Stopping at total_inner_it={inner_it} due to small gap={fw_gap_from_first_iteration:.3e} <= delta_t={delta_t:.3e}")
                    found = True
                    break
                # y_k = (1-gamma_k) y_{k-1} + gamma_k x_k
                y_inner = (1.0 - gamma_k) * y_inner + gamma_k * x_new

                x_inner = x_new
                inner_it += 1

        x = z_k   # theoretically we should output y_inner here, but not align with our outer loop, see the description in the note of overleaf
    
    return x, np.array(times),np.array(function_value)



def dc_vr_scgs(problem, x0, cfg=Dc_vr_scgs, rng=None, seed=1):
    rng = np.random.default_rng(seed) if rng is None else rng
    n = problem.n
    L_H, _, _ = problem.lipschitz_constants()
    L_sur = 0.0
    fw_gap_from_first_iteration = 0.0
    
    
    
    #throw away the strong convexity, let the proximal term to be trivial
    mu0=0

    D= problem.diameter()


    x = x0.copy()
    times, gm2,function_value = [], [],[]
    start = time.perf_counter()
    metric_time = 0.0
    
    for t in range(cfg.T_outer):
        mu_t = mu0 * (cfg.mu_decay ** t)
        #delta_t = 1/((t+1))   #????????????100/((t+1)*(t+2))
        #delta_t = 1/((t+1)**2*(t+2)**2)
        delta_t= 1/((t+1)*(t+2))
        L_sur = L_H + mu_t

        t0 = time.perf_counter()
        phi_val = problem.phi(x)
        metric_time += time.perf_counter() - t0

        times.append(time.perf_counter() - start - metric_time)
        function_value.append(phi_val)


        g_t = problem.G_full_grad(x)
        x_center = x.copy()

        delta_s = 0
        x_inner = x.copy()         # x0 <---- x_t
        y_inner = x_inner.copy()   # y0 <---- x_t

        found = False
        inner_it = 0
        epochs = int(np.ceil(cfg.max_inner / cfg.mB))  #theoretically we dont set this epochs bound, but for practice we do
        for s in range(epochs):
            if found==True:
                break
            '''if gap <= delta_min:  # gap \leq delta_t:
                break'''
            
            
            w_tilde = y_inner.copy()
            fullH_tilde = problem.H_full_grad(w_tilde)
            #N_s= int(np.ceil(2*np.sqrt(6*L_sur/mu_t)))
            delta_s = max(cfg.delta_min, cfg.delta0 * (cfg.delta_decay ** s))
            #N_s = int(np.ceil(2**(((s+1)/2+2))))
            N_s = 50
            #print(N_s)
            for k in range(N_s): #something off here!!!!!!!!!!!!!!!!
                if k == N_s-1:
                    print(f"oh {t} | {s} {N_s} {fw_gap_from_first_iteration}")

                # ---- CGS extrapolation point z_k = (1-gamma_k) y + gamma_k x ----
                gamma_k = 2.0 / (k + 2.0)  # in (0,1], gives gamma_1=1,... inner_it+2
                lam_t= (k+1.0) / (3.0 * L_sur)

                eta_k = 8.0 * L_sur * delta_s / (N_s * (k+1.0))
                #eta_k = 2*L_sur*D**2 / (N_s * (k+1.0))
                #eta_k = 2*L_sur*delta_s / (N_s * (k+1.0))
                
                z_k = (1.0 - gamma_k) * y_inner + gamma_k * x_inner
                
                idx = rng.integers(0, n, size=800)   #200

                # SVRG for H, evaluated at z_k
                vH = (problem.H_batch_grad(z_k, idx) - problem.H_batch_grad(w_tilde, idx)) + fullH_tilde

                # Surrogate gradient estimator at z_k: ∇H(z_k) - g_t + μ (z_k - x_center)
                v = vH - g_t + mu_t * (z_k - x_center)
                
                x_new,fw_gap_from_first_iteration= cndg_l1_gap0_checking(v, x_inner,z=z_k, lam=lam_t, eta=eta_k, tau=problem.tau, max_cndg=cfg.max_cndgB)  
                
                if fw_gap_from_first_iteration <= delta_t:
                        print(f"outer{t}, epoch {s}: current_N_s={N_s} ") #batch_size={batch_size}, control_batch_size={constant_control_batch_size}
                        print(f"Stopping at total_inner_it={inner_it} due to small gap={fw_gap_from_first_iteration:.3e} <= delta_t={delta_t:.3e}")
                        gap = fw_gap_from_first_iteration
                        found = True
                        break

                # y_k = (1-gamma_k) y_{k-1} + gamma_k x_k
                y_inner = (1.0 - gamma_k) * y_inner + gamma_k * x_new

                x_inner = x_new
                inner_it += 1

        x = z_k   # theoretically for scgs we should output y_inner here, but not align with our outer loop, see the description in the note of overleaf
        
    return x, np.array(times), np.array(function_value)
#----------------
#--SCGS condg subproblem solvers
#----------------

def cndg_l1(v, x, lam, eta, tau, max_cndg):
    u= x.copy()
    inv_lam= 1.0/lam
    last_gap = None
    iterations = 0

    for l in range(0, max_cndg):
        gradient = v + inv_lam*(u-x)
        s = lmo_l1_ball(gradient, tau=tau)
        gap = float(np.dot(gradient,u-s))
        last_gap = gap
        iterations += 1

        if gap <= eta:
            return u, {"gap": last_gap, "iterations": iterations, "hit_max": False}
        
        d = s-u
        L_times_d2 = inv_lam*float(np.dot(d,d))
        if L_times_d2 <= 1e-18:
            break
        alpha_l = min(1.0, np.dot(gradient,-d)/L_times_d2)
        u = u + alpha_l*d
    
    return u , {"gap": last_gap, "iterations": iterations, "hit_max": True}

        


def cndg_l1_gap0_checking(v,x,z,lam,eta,tau,max_cndg):
    u = x.copy()
    inv_lam = 1.0/lam
    gap0 = None

    for l in range(0,max_cndg):
        gradient = v + inv_lam*(u-x)
        s = lmo_l1_ball(gradient, tau=tau)
        gap = float(np.dot(gradient,u-s))
        
        if l == 0:
            gap0 = float(np.dot(gradient,z-s))

        if gap <= eta:
            return u,gap0
        
        d = s-u
        L_times_d2 = inv_lam*float(np.dot(d,d))
        if L_times_d2 <= 1e-18:
            break

        alpha_l = min(1.0, np.dot(gradient,-d)/L_times_d2)
        u = u + alpha_l*d
    
    return u,gap0

#-------------------
#-- run experiment
#-------------------

def run_experiment(base_seed=0):
    set_thesis_plot_style()
    print("Starting run...")
    sb = SeedBank(base_seed)
    
    n,d = 2000000,100
    P,y = make_logistic_cauchy_pen_problem(n=n,d=d,scale=5.0,seed=sb.seeds["data"])
    prob = LogisticCauchyPenaltyProblem( P,y,lam=0.5,delta=0.2,tau=10.0)
    
    x0 = np.random.default_rng(11).standard_normal(d)
    x0 = projection_onto_l1_ball(x0,tau=prob.tau)

    cfg_Ncgs_vr = copy.deepcopy(NCGS_VR)
    cfg_Fcgs = copy.deepcopy(FCGS)
    cfg_prox_dc = copy.deepcopy(Prox_dc_vr_scgs)
    cfg_dc = copy.deepcopy(Dc_vr_scgs)


    xA, tA,function_value_A, infoA = ncgs_vr(
        prob, x0,
        cfg_Ncgs_vr, rng=sb.rng("ncgs_vr")
    )

    xB, tB,function_value_B = prox_dc_vr_scgs(
        prob, x0,
        cfg_prox_dc,rng=sb.rng("prox_dc_vr_scgs")
    )

    xB_2,tB_2,function_value_B_2 = dc_vr_scgs(
        prob, x0,
        cfg_dc,rng=sb.rng("dc_vr_scgs")
    )
    # --- FCGS (Algorithm 4, Gao & Huang 2020 supp.) ---

    xC, tC, function_value_C, infoC = fcgs(
        prob, x0,
        cfg_Fcgs,rng=sb.rng("fcgs")
    )
    

    #Plot 
    plt.figure(1)
    plt.plot(tA, function_value_A, marker='o', linewidth=1.5, label='NCGS-VR')
    plt.plot(tB, function_value_B, marker='s', linewidth=0.8, label='Prox-DC + VR-SCGS')
    plt.plot(tC, function_value_C, marker='^', linewidth=1.5, label='FCGS')
    plt.plot(tB_2, function_value_B_2, marker='x', color='black',linewidth=0.8, label='DC + VR-SCGS')
    plt.xlabel("CPU time (s)")
    plt.ylabel("function value")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.xlim(0, 70) # just for plot the frist 70 secs. we can delete this line, 
    plt.tight_layout()
    # --- unique filename (absolute path next to this file) ---
    RESULTS_DIR = Path(__file__).resolve().parent / "Plots_logistic"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"result_{stamp}_seed{base_seed}_{uuid.uuid4().hex[:6]}"
   

    # 3. Save Figure 2
    plt.figure(1)  
    fname2 = RESULTS_DIR / f"{base_name}_value.png"
    plt.tight_layout(pad=0.15)
    plt.savefig(fname2.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.savefig(fname2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("Saved plot to:", fname2.resolve())


    return {"t_ncgs": tA, "info_ncgs": infoA, "t_prox_dc": tB,  "t_fcgs": tC,  "info_fcgs": infoC}


if __name__ == "__main__":
    print("Done. Showing plot...")
    out = run_experiment(base_seed=20)   #base_seed=8
    plt.show()