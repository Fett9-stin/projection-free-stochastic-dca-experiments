

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Iterable
from scipy.sparse.linalg import  svds, ArpackNoConvergence
import warnings
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
#--LMO solver
#------------------------

def frobenius_inner_product(A, B):   # <A,B>_F= Sum(Aij*Bij)
    return np.sum(A * B)

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

    


def projection_onto_nuclear_ball(X, tau=1.0): 
# find the closest matrix to X using the frobenius distance
# solving 1/2 ||X-Z||_F^2 on ||Z||_* <= tau
    X = np.asarray(X, dtype=np.float64)
    u, s, vh = np.linalg.svd(X, full_matrices=False)           # we use a compact SVD which fulfills U_compact diag(s) V_compact ^T = U SIGMA V^T(fullSVD)
    if np.sum(s) <= tau:
        return X.copy()
    s_proj= projection_onto_l1_ball(s,tau=tau)
    return (u*s_proj) @ vh     #cheaper than u @ np.diag(s) @ vh, but do the same thing

def lmo_nuclear_ball_full_svd(X,tau=1.0):
    X = np.asarray(X, dtype=np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)    

    u1 = U[:,0]
    v1 = Vt[0,:]

    return np.outer(-tau*u1,v1)


def lmo_nuclear_ball(X,tau=1.0,tol=1e-6,maxiter=300,v0=None):
    X= np.asarray(X, dtype=np.float64)
    try:                      # use a partial SVD to solve the subproblem
        U,_,Vt=svds(
            X,
            k=1,
            which='LM',
            tol=tol,
            maxiter=maxiter,
            v0=v0,
            solver='arpack',
            return_singular_vectors=True)
        u1=U[:,0]
        v1=Vt[0,:]
        return np.outer(-tau*u1,v1)
    except (ArpackNoConvergence, np.linalg.LinAlgError, ValueError) as e:   # if partial svd fails, use full svd
        warnings.warn(
            f"svds failed with exception {e}, using lmo_nuclear_ball_full_svd instead",
            RuntimeWarning
        )
        return lmo_nuclear_ball_full_svd(X,tau=tau)


#--------------
#-- Matrix completion problem
#--------------


class MatrixCompletionExperiment:

    def __init__(self,m,n_cols,omega_rows,omega_cols,omega_values, tau=1.0, mu=0.15, gamma=3.0,svd_epsilon=1e-12):
        self.m = int(m)
        self.n_cols = int(n_cols)
        self.omega_rows = np.asarray(omega_rows,dtype=np.int64)
        self.omega_cols = np.asarray(omega_cols,dtype=np.int64)
        self.omega_values = np.asarray(omega_values,dtype=np.float64)
        self.n = int(len(self.omega_values))
        self.d = self.m * self.n_cols
        self.tau = float(tau)
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.svd_epsilon = float(svd_epsilon)
    
    def diameter(self):
        return 2.0*self.tau
    
    def lipschitz_constants(self):
        L_H= float(1.0/ self.n)
        L_G= float(self.mu*(self.gamma**2))
        return L_H,L_G,float(L_H+L_G)
    
    #----we define H(X) = 1/2|Omega| sum_{(i,j)\in Omega} (X_{ij}-M_{ij})^2
    #----and calculate its gradient and batch_gradient
    def H_value(self,X):
        X= np.asarray(X, dtype=np.float64)
        return 0.5*np.sum((X[self.omega_rows,self.omega_cols]-self.omega_values)**2)/self.n
    
    def H_full_grad(self,X):
        X= np.asarray(X, dtype=np.float64)
        grad = np.zeros_like(X)

        grad[self.omega_rows,self.omega_cols] = (X[self.omega_rows,self.omega_cols]-self.omega_values)/self.n

        return grad
    
    def H_batch_grad(self,X,idx):
        X= np.asarray(X, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)

        rows= self.omega_rows[idx]
        cols= self.omega_cols[idx]
        values= self.omega_values[idx]

        grad = np.zeros_like(X)
        subtract = X[rows,cols]-values
        np.add.at(grad, (rows,cols), subtract)
        
        grad /= float(len(idx))

        return grad
    
    #-----we define G as mu*sum psi(sigma_l(X))
    #-----and calculate function value and its gradient

    def psi_value(self,s):      #formular was shown in the thesis
        s= np.asarray(s, dtype=np.float64)
        return self.gamma*s - np.log1p(self.gamma*s)
    
    def psi_grad(self,s):       
        s= np.asarray(s, dtype=np.float64)
        return self.gamma - self.gamma/(1.0+self.gamma*s)
    
    def G_value(self,X):
        X= np.asarray(X, dtype=np.float64)
        s = np.linalg.svd(X, compute_uv=False,full_matrices=False)      # we only need the singular values here
        return self.mu * float(np.sum(self.psi_value(s)))
    
    def G_full_grad(self,X):      #nabla G(X) = U diag(mu psi'(sigma_l(X))) V^T    
        X= np.asarray(X, dtype=np.float64)
        U,s,Vt = np.linalg.svd(X,full_matrices=False)
        coeff = self.mu * self.psi_grad(s)
        return (U*coeff[None,:])@Vt
    
    def G_batch_grad(self,X,idx):
        return self.G_full_grad(X)
    
    #--------calculate function value and gradient of Phi = H- G

    def phi(self,X):
        return self.H_value(X) - self.G_value(X)
    
    def phi_grad(self,X):
        return self.H_full_grad(X) - self.G_full_grad(X)


def make_matrix_completion_problem(
        m=500,
        n_cols=300,
        rank=6,
        obs_fraction=0.25,
        noise_std=0.08,
        singular_values=[25.0, 18.0, 12.0, 8.0, 4.0, 2.0],
        mu=0.15,
        gamma=3.0,
        tau=None,
        seed=None
        ):
    rng = np.random.default_rng(seed)
    U,_=np.linalg.qr(rng.normal(size=(m,rank)))       #using gaussian to generate matrices with orthogonormal columns
    V,_=np.linalg.qr(rng.normal(size=(n_cols,rank)))

    singular_values=np.asarray(singular_values,dtype=np.float64)    #should be some predefined singular values for the M we want to recover
    
    M = (U*singular_values)@ V.T

    total_entries= m * n_cols
    observations_size = int(np.ceil(obs_fraction * total_entries))
    observed_flat_idx= rng.choice(total_entries,size = observations_size, replace=False)      #choosing the observed entries

    observed_rows = observed_flat_idx // n_cols   # convert the flat selected index into matrix coordinates using map k = i*n_cols + j where k is the observed flat index
    observed_cols = observed_flat_idx % n_cols

    observed_values = M[observed_rows,observed_cols].copy()
    observed_values += noise_std * rng.normal(size=observations_size)
    
    if tau is None:
        tau= 1.8*sum(singular_values)

    prob = MatrixCompletionExperiment(
        m=m,
        n_cols=n_cols,
        omega_rows=observed_rows,
        omega_cols=observed_cols,
        omega_values=observed_values,
        tau = tau,
        mu=mu,
        gamma=gamma,
    )
    
    return prob




#-----------------
#-- Algorithms: Prox_dc_vr_scgs, dc_vr_scgs, NCGS_VR, FCGS
#-----------------
@dataclass
class Prox_dc_vr_scgs:
    T_outer = 20          #20
    bB = 64
    mB = 30
    mu0 = 0.6
    mu_decay = 0.97
    delta0 = 1
    delta_decay = 0.5
    delta_min = 1e-8
    eta_scale = 0.7
    max_inner = 1200
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
    max_inner = 1200
    max_cndgB = 500
    check_gap_every = 10

@dataclass
class NCGS_VR:
    T = 200      # the paper states T should divide mA, but for experiment, we dont follow this too strictly
    mA = None #int(np.ceil(n**(1/3))),
    bA = None #int(np.ceil(n ** (2/3))),
    lam = None   # 1/(3*L_phi)
    eta_ncgs = None   #1/T
    max_cndgA = 500
    log_everyA = 10


@dataclass
class FCGS:
    K= 200                   #200
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
        full_grad_tilde = problem.phi_grad(x_tilde)
        for t in range(mA):
            if iteration >= cfg.T:
                found = True     # we set found also to break the outer loop s
                break
            idx = rng.integers(0,n,size=bA)
            v = (problem.H_batch_grad(x,idx)-problem.H_batch_grad(x_tilde,idx))
            v-= (problem.G_batch_grad(x,idx)-problem.G_batch_grad(x_tilde,idx))
            v+= full_grad_tilde
            x_new,cndg_info = cndg_nuclear(v,x,lam=lam,eta=eta_ncgs,tau=problem.tau,max_cndg=cfg.max_cndgA)

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
        qc = int(np.sqrt(n))    #the computational result of sqrt(n) might not be integer, but we only keep the integer part
    if cfg.bc is None:
        bc = qc
    
    xk = x0.copy()
    x_previous = x0.copy()
    v = problem.phi_grad(xk)

    times, function_value = [],[]
    start = time.perf_counter()
    metric_time = 0.0
    
    for k in range(cfg.K):
        if k % qc == 0:
            v = problem.phi_grad(xk)
        else:
            idx = rng.integers(0,n,size=bc)
            v +=(problem. H_batch_grad(xk,idx)-problem.H_batch_grad(x_previous,idx))
            v -=(problem. G_batch_grad(xk,idx)-problem.G_batch_grad(x_previous,idx))
        
        x_new, cndg_info = cndg_nuclear(v, xk,lam=lam, eta=eta_FCGS, tau = problem.tau,max_cndg=cfg.max_cndgC)

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
        L_sur = L_H + mu_t                # for hat{phi}_t(x)=H(x) -G(x_t)- <nabla G(x_t),x-x_t> + mu_t/2 ||x-x_t||^2, L_sur = L_H + mu_t
        
        #delta_t = max(delta_min, delta0 * (delta_decay ** t))    
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

        x_inner = x.copy()
        y_inner = x_inner.copy()   # we use the empirical initialization, the warm starting in the experiment section

        found = False
        inner_it = 0
        epochs = int(np.ceil(cfg.max_inner / cfg.mB))
        for s in range(epochs):
            if found==True:         # used for the stopping criterion to also break from the epoch loop
                break
            
            w_tilde = y_inner.copy()
            fullH_tilde = problem.H_full_grad(w_tilde)

            D_s_square = D**2*L_sur/ (mu_t*2**s)
            #N_s= int(np.ceil(2*np.sqrt(6*L_sur/mu_t)))
            N_s = int(np.ceil(np.sqrt(32*(L_sur/mu_t))))
            
            for k in range(N_s):    

                
                gamma_k = 2.0 / (k + 2.0)  
                lam_t= (k+1.0) / (3.0 * L_sur)     
                eta_k = 2*L_sur*D_s_square / (N_s * (k+1.0))
                z_k = (1.0 - gamma_k) * y_inner + gamma_k * x_inner  # CGS extrapolation point 
        

                idx = rng.integers(0, n, size=200)

                # the batch sized part of v_k
                vH = (problem.H_batch_grad(z_k, idx) - problem.H_batch_grad(w_tilde, idx)) + fullH_tilde

                # the full definition of v_k in DC_SVRCGS
                v = vH - g_t + mu_t * (z_k - x_center)
                
                x_new,fw_gap_from_first_iteration= cndg_nuclear_gap0_checking(v, x_inner,z=z_k, lam=lam_t, eta=eta_k , tau=problem.tau, max_cndg=cfg.max_cndgB)  
                # We compute gap_0 using z_k, thats why it appears as an input of cndg_nuclear_gap0_checking
                if k == N_s-1:
                    print(f"oh {s} {N_s} {fw_gap_from_first_iteration}")
                # if the inner loop hits the max iteration, we want to record the gap_0

                if fw_gap_from_first_iteration <= delta_t:
                    print(f"outer{t}, epoch {s}: current_N_s={N_s}")         #control_N_s={constant_control_N_s},control_batch_size={constant_control_batch_size}
                    print(f"Stopping at total_inner_it={inner_it} due to small gap={fw_gap_from_first_iteration:.3e} <= delta_t={delta_t:.3e}")
                    found = True
                    break
                # y_k = (1-gamma_k) y_{k-1} + gamma_k x_k
                y_inner = (1.0 - gamma_k) * y_inner + gamma_k * x_new

                x_inner = x_new
                inner_it += 1

        x = z_k   # in normal CGS/SCGS we should output y_inner here, but not align with our outer loop, in DC_SVRCGS we should output z_k
    
    return x, np.array(times),np.array(function_value)



def dc_vr_scgs(problem, x0, cfg = Dc_vr_scgs, rng=None, seed=1):
    rng = np.random.default_rng(seed) if rng is None else rng
    n = problem.n
    L_H, _, _ = problem.lipschitz_constants()
    L_sur = 0.0
    fw_gap_from_first_iteration = 0.0
    
    #throw away the strong convexity, let the proximal term to be trivial
    mu0=0

    D= problem.diameter()


    x = x0.copy()
    times, function_value = [], []
    start = time.perf_counter()
    metric_time = 0.0
    
    for t in range(cfg.T_outer):
        mu_t = mu0 * (cfg.mu_decay ** t)
        #delta_t = 1/((t+1))   #100/((t+1)*(t+2)) could be considered
        #delta_t = 1/((t+1)**2*(t+2)**2)
        delta_t= 1/((t+1)*(t+2))
        L_sur = L_H + mu_t
        


        t0 = time.perf_counter()
        phi_val = problem.phi(x)
        metric_time += time.perf_counter() - t0

        times.append(time.perf_counter() - start - metric_time)
        function_value.append(phi_val)

        g_t = problem.G_full_grad(x)   # \nabla G(x_t)
        x_center = x.copy()

        delta_s = 0
        x_inner = x.copy()
        y_inner = x_inner.copy()   # warm-starting, the empirical initialization

        found = False
        inner_it = 0
        epochs = int(np.ceil(cfg.max_inner / cfg.mB))
        for s in range(epochs):
            if found==True:
                break
            
            
            w_tilde = y_inner.copy()
            fullH_tilde = problem.H_full_grad(w_tilde)
            #N_s= int(np.ceil(2*np.sqrt(6*L_sur/mu_t)))
            delta_s = max(cfg.delta_min, cfg.delta0 * (cfg.delta_decay ** s))
            #N_s = int(np.ceil(2**(((s+1)/2+2))))
            N_s = 20
            #print(N_s)
            for k in range(N_s): 
                if k == N_s-1:
                    print(f"oh {t} | {s} {N_s} {fw_gap_from_first_iteration}")

                gamma_k = 2.0 / (k + 2.0)  
                lam_t= (k+1.0) / (3.0 * L_sur)
                eta_k = 8.0 * L_sur * delta_s / (N_s * (k+1.0))
                #eta_k = 2*L_sur*D**2 / (N_s * (k+1.0))
                
                z_k = (1.0 - gamma_k) * y_inner + gamma_k * x_inner

                idx = rng.integers(0, n, size=200)   #200

                vH = (problem.H_batch_grad(z_k, idx) - problem.H_batch_grad(w_tilde, idx)) + fullH_tilde

                # v_k in DC_SVRCGS
                v = vH - g_t + mu_t * (z_k - x_center)
                
                x_new,fw_gap_from_first_iteration= cndg_nuclear_gap0_checking(v, x_inner,z=z_k, lam=lam_t, eta=eta_k, tau=problem.tau, max_cndg=cfg.max_cndgB)  
                
                if fw_gap_from_first_iteration <= delta_t:
                        print(f"outer{t}, epoch {s}: current_N_s={N_s}")   #,control_N_s={constant_control_N_s}, control_batch_size={constant_control_batch_size}
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
#--SCGS cndg subproblem solvers
#----------------

def cndg_nuclear(v, x , lam, eta, tau=1.0, max_cndg=300):
    u = x.copy()
    inv_lam = 1.0 / lam
    last_gap = None
    iterations = 0
    
    for l in range(0,max_cndg):
        gradient = v + inv_lam*(u-x)
        s = lmo_nuclear_ball(gradient, tau=tau)
        gap = float(frobenius_inner_product(gradient,u-s))
        last_gap= gap
        iterations += 1

        if gap <= eta:
            return u,{"gap": last_gap, "iterations": iterations, "hit_max": False}
        
        d = s - u
        L_times_d2 = inv_lam * float(np.sum(d * d))
        if L_times_d2 <= 1e-18:    # For numerical safety of construct alpha_l. also in this case the fwgap should be extremely small
            break
        alpha_l = min(1.0, frobenius_inner_product(gradient,-d)/L_times_d2)         #short step and exact line search concides here, bc the subproblem is quadratic
        u = u + alpha_l * d
        
    return u,{"gap": last_gap, "iterations": iterations, "hit_max": True}
        

def cndg_nuclear_gap0_checking(v, x ,z, lam, eta, tau=1.0, max_cndg=300):
    u = x.copy()
    inv_lam = 1.0 / lam
    gap0= None
    
    for l in range(0,max_cndg):
        gradient = v + inv_lam*(u-x)
        s = lmo_nuclear_ball(gradient, tau=tau)
        gap = float(frobenius_inner_product(gradient,u-s))

        if l== 0:
            gap0 = float(frobenius_inner_product(gradient,z-s))

        if gap <= eta:
            print("hit_max? NO, cndg early stopped")
            return u, gap0
        
        d = s - u
        L_times_d2 = inv_lam * float(np.sum(d * d))
        if L_times_d2 <= 1e-18:
            break
        alpha_l = min(1.0, frobenius_inner_product(gradient,-d)/L_times_d2)         #short step and exact line search concides here, bc the subproblem is quadratic
        u = u + alpha_l * d
    print("cndg hit_max? YES")
    return u,gap0


#-------------------
#-- run experiment
#-------------------

def run_experiment(base_seed=0):
    set_thesis_plot_style()
    print("Starting run...")
    sb = SeedBank(base_seed)
  
    prob = make_matrix_completion_problem(seed=sb.seeds["data"])

    n=prob.n # acutally n = int(np.ceil(prob.m * prob.n_cols * prob.obs_fraction))

    #----choose initial point for all algorithms-----
    # we abandoned the random initialization of x_0
    #rng0 = np.random.default_rng(10)         #tough initialization for matrix completion
    #x0_raw = rng0.standard_normal((prob.m, prob.n_cols))
    #x0 = project_to_nuclear_ball(x0_raw, tau=prob.radius)

    x0 = np.zeros((prob.m, prob.n_cols))  #matrix completion initialization
 
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
    RESULTS_DIR = Path(__file__).resolve().parent / "Plots_matrix_completion"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"result_{stamp}_seed{base_seed}_{uuid.uuid4().hex[:6]}"
   

    # 3. Save Figure to file
    plt.figure(1)  
    fname2 = RESULTS_DIR / f"{base_name}_value.png"
    plt.tight_layout(pad=0.15)
    plt.savefig(fname2.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.savefig(fname2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("Saved plot to:", fname2.resolve())


    return {"t_ncgs": tA, "info_ncgs": infoA, "t_prox_dc": tB,  "t_fcgs": tC,  "info_fcgs": infoC}


if __name__ == "__main__":
    print("Done. Showing plot...")
    out = run_experiment(base_seed=14)   
    plt.show()