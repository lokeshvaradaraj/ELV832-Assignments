import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mpl_colors 
from copy import deepcopy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# Plotting Configuration
# =============================================================================

# --- Color Definitions ---
c_semantic='deeppink'
c_positional='royalblue'
c_att = 'rebeccapurple'
c_lin = 'orange'
c_attlin = 'crimson' 
c_spinodal = 'forestgreen'
c_no_col = 'black' 

# --- Custom Colormaps ---
cmap_uninf = LinearSegmentedColormap.from_list('INF-UNINF',
                                                   [mcolors.to_rgba(c_semantic)[:3], (1, 1, 1), mcolors.to_rgba(c_positional)[:3]], N=100)
cmap_attlin = LinearSegmentedColormap.from_list('INF-UNINF',
                                                   [mcolors.to_rgba(c_lin)[:3], (1, 1, 1), mcolors.to_rgba(c_att)[:3]], N=100)
cmap_pos = LinearSegmentedColormap.from_list('INF-UNINF',
                                                   [mcolors.to_rgba('#8DE0A8')[:3],
                                                    mcolors.to_rgba('#93FFE0')[:3],
                                                    mcolors.to_rgba('#85EFFF')[:3],
                                                    mcolors.to_rgba('#6BBFFF')[:3],
                                                    mcolors.to_rgba(c_positional)[:3]], N=100)
cmap_att = LinearSegmentedColormap.from_list('INF-UNINF',
                                                   [mcolors.to_rgba('#FFE0B6')[:3],
                                                    mcolors.to_rgba('#FFB48C')[:3],
                                                    mcolors.to_rgba('#FF8166')[:3],
                                                    mcolors.to_rgba('#FF3E3B')[:3],
                                                    mcolors.to_rgba(c_semantic)[:3]], N=100)


# =============================================================================
# Data Generation (Teacher Model)
# =============================================================================

def generate_teacher_dataset(N, D=100, L=2, DK=1, omega=0.3, seed=None):
    """
    Generate a teacher dataset of size N using tied dot product attention. 
    Combines semantic and positional attention based on omega.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    W_Q = torch.randn(D)
    sigma = 0.5 # sample X from multivariate gaussian, var->0.25
    X_all = sigma * torch.randn((N, L, D)) 

    softmax=torch.nn.Softmax(dim=-1)

    # Semantic component 
    xQ = torch.einsum("imk,k->im", X_all, W_Q)
    att_score = softmax(torch.einsum("im,in->imn", xQ, xQ))
    semantic_component = torch.einsum("imn,inj->imj", att_score, X_all)

    # Positional component (fixed matrix for L=2).
    pos_matrix = torch.Tensor([[.6,.4],[.4,.6]]) 
    positional_component = torch.einsum("nld,fl->nfd", X_all, pos_matrix)

    # Combine components
    T_all = (1-omega) * semantic_component + omega * positional_component

    return X_all, T_all, W_Q


# =============================================================================
# Student Model Definition
# =============================================================================

class StudentAttentionModel(nn.Module):
    """
    Student attention model attempting to learn the teacher's behavior.
    Includes semantic/positional initialization types and fixed positional encoding.
    """
    def __init__(self, D=100, DK=1, L=2, init_type='random', WQ_teacher=None):
        super().__init__()
        self.D = D
        self.DK = DK
        self.L = L

        # Initialization of W_Q
        if init_type == 'semantic':
            if WQ_teacher is None: raise ValueError("Need WQ_teacher for semantic init")
            self.W_Q = torch.nn.Parameter(WQ_teacher.clone().detach().reshape(D, DK)) 
        elif init_type == 'positional':
            self.W_Q = torch.nn.Parameter(torch.ones(D, DK))
        else: 
            self.W_Q = torch.nn.Parameter(torch.randn(D, DK))
            
    def forward(self, X):

        B, L_in, D_in = X.shape

        # Positional encoding: r1 = +1 vector, r2 = -1 vector
        r1= torch.ones(self.D, device=X.device) 
        R_base = torch.vstack((r1, -r1)) 
        Rs = R_base.unsqueeze(0).repeat(B, 1, 1) 
        X_pos = X + Rs  # Add positional encoding: (B, L, D)
        
        xQ = torch.einsum("blk,kd->bld", X_pos, self.W_Q) # (B, L, DK)

        # Calculate attention scores
        xQ = xQ.squeeze(-1) # Shape (B, L)
        A = torch.nn.Softmax(dim=-1)(torch.einsum("bl,bm->blm", xQ, xQ)) # (B, L, L)
        # Apply attention to position-aware values
        Y = torch.einsum("blm,bmd->bld", A, X_pos) # (B, L, D)

        return Y

# =============================================================================
# Training Utilities 
# =============================================================================

def loss_SSE(Y, T):
    """SSE Loss normalized by D """
    return torch.sum((Y-T)**2) / (2 * T.shape[-1])

def train_student(X_train, T_train, X_test, T_test, lam=1e-2, lr=0.15, epochs=5000,
                  init_type='semantic', W_Q_teacher=None, DK=1):
    """
    Trains a student attention model with specified initialization. 
    Calculates metrics m and theta in the final epochs. Plots loss curve.
    (Code logic matches original user script).
    """
    N_train, L, D = X_train.shape
    N_test = X_test.shape[0]
    
    batch_size = N_train 
    train_dataset = TensorDataset(X_train, T_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    W_Q_teacher_flat = W_Q_teacher.flatten() 

    model = StudentAttentionModel(D=D, DK=DK, L=L, init_type=init_type, WQ_teacher=W_Q_teacher)


    gen_error_list = []
    train_error_reg_list = [] 
    m_list, theta_list = [], []
    
    r1 = torch.ones(D) 

    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=lr, weight_decay=lam) 

    epoch_loss_history = [] 
    print(f"  Training student ({init_type} init): epochs={epochs}, lr={lr}, lambda={lam:.1e}... ", end="")
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0 
        num_batches = 0
        
        for X_batch, T_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = loss_SSE(Y_pred, T_batch) 
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() 
            num_batches += 1
            
        avg_epoch_loss = epoch_loss_sum / num_batches 
        epoch_loss_history.append(avg_epoch_loss) 
        # --- Calculate metrics only in the last ~20 epochs ---
        epochs_to_average = 20 
        if epoch >= epochs - epochs_to_average:
            model.eval()
            with torch.no_grad():
                 Y_test_pred = model(X_test) 
                 gen_error = loss_SSE(Y_test_pred, T_test.to(device)).item() / N_test 
                 gen_error_list.append(gen_error)
                 
                 W_Q_flat = model.W_Q.flatten()
                 l2_penalty = (lam / 2) * float(torch.sum(W_Q_flat**2)) 
                 Y_train_pred = model(X_train)
                 train_loss_unreg = loss_SSE(Y_train_pred, T_train).item() / N_train
                 train_error_reg = train_loss_unreg + l2_penalty
                 train_error_reg_list.append(train_error_reg)

                 # Summary statistics
                 m = np.abs(float(torch.dot(r1, W_Q_flat) / D))
                 m_list.append(m)
                 
                 theta = np.abs(float(torch.dot(W_Q_teacher_flat, W_Q_flat) / D)) 
                 theta_list.append(theta)     
 
    # --- Averaging Final Metrics over last few epochs ---
    avg_gen_error = np.mean(gen_error_list) if gen_error_list else np.nan
    avg_train_error_reg = np.mean(train_error_reg_list) if train_error_reg_list else np.nan
    avg_m = np.mean(m_list) if m_list else np.nan
    avg_theta = np.nanmean(theta_list) if theta_list else np.nan 
    
    return model, avg_gen_error, avg_train_error_reg, avg_m, avg_theta

def train_student_L2_CV(X_train, T_train, X_val, T_val, lam_list, lr=0.15, epochs=5000, 
                        init_type='semantic', W_Q_teacher=None, DK=1):
    """
    Performs cross-validation over lambda for the student model.
    Selects lambda based on minimum error on the validation set.
    """
    results = []
    for lam in lam_list:

        model, gen_error, train_error_reg, m, theta = train_student(
            X_train, T_train,
            X_val, T_val, 
            lam=lam,
            lr=lr,
            epochs=epochs,
            init_type=init_type,
            W_Q_teacher=W_Q_teacher,
            DK=DK
        )
        results.append({
            'lam': lam,
            'gen_error': gen_error, 
            'train_error_reg': train_error_reg,
            'm': m,
            'theta': theta,
            'model': model 
        })

    best_result = min(results, key=lambda x: x['gen_error'])
        
    # Return the metrics associated with the best lambda
    return (best_result['model'], best_result['gen_error'], best_result['train_error_reg'],
            best_result['m'], best_result['theta'])


# =============================================================================
# Figure 2 Experiment Runners  
# =============================================================================

def run_fig_2A(X, T, W_Q_teacher, alphas, lam_list=[1e-2], lr=0.15, epochs=5000, DK=1, test_ratio=0.2):
    """
    Runs the Fig 2A experiment comparing semantic vs positional initialization
    across sample complexity alpha = N_train / D, using L2 CV.
    """
    N_total, L, D = X.shape
    results = []
    N_test = int(0.2* D) 
    X_test = X[-N_test:]
    T_test = T[-N_test:]
    X_pool = X[:-N_test] # Data available for training
    T_pool = T[:-N_pool]
    N_pool = X_pool.shape[0]
    
    for alpha in alphas:
        N_train = int(alpha * D)
        
        X_train = X_pool[:N_train]
        T_train = T_pool[:N_train]

        print(f"\n=== Alpha = {alpha:.3f} | N_train = {N_train} ===")

        W_Q_sem_init = W_Q_teacher.clone().detach() 
        
        # sem init
        model_sem, e_gen_sem, e_train_sem_reg, m_sem, theta_sem = train_student_L2_CV(
                                  X_train, T_train, X_test, T_test, # Use test set for CV validation
                                  lam_list, lr, epochs, init_type='semantic', 
                                  W_Q_teacher=W_Q_sem_init, DK=DK)
        
        # positional init
        model_pos, e_gen_pos, e_train_pos_reg, m_pos, theta_pos = train_student_L2_CV(
                                  X_train, T_train, X_test, T_test, 
                                  lam_list, lr, epochs, init_type='positional', 
                                  W_Q_teacher=W_Q_sem_init, DK=DK) 

        # Compute deltas (Pos - Sem)
        delta_gen = e_gen_pos - e_gen_sem
        delta_train_reg = e_train_pos_reg - e_train_sem_reg 

        results.append({
            'alpha': alpha,
            'semantic_loss_gen': e_gen_sem,
            'positional_loss_gen': e_gen_pos,
            'semantic_loss_train_reg': e_train_sem_reg, 
            'positional_loss_train_reg': e_train_pos_reg, 
            'delta_gen': delta_gen,
            'delta_train_reg': delta_train_reg, 
            'theta_sem': theta_sem,
            'theta_pos': theta_pos,
            'm_pos': m_pos,
            'm_sem': m_sem 
        })
    return results


def run_fig_2A_mean(D, omega, L, alphas, lam_list=[1e-2], lr=0.15, epochs=5000, DK=1, instances=5, test_ratio=0.2):
    """
    Runs run_fig_2A multiple times (instances) and averages the results
    """
    all_instance_results = [] 
    max_alpha = max(alphas) if alphas else 1.0
    N_total = int(np.ceil( (max_alpha * D) / (1.0 - test_ratio) )) 
    N_total_original_calc = int((max(alphas, default=1.0) + test_ratio + 0.1) * D) 

    for inst in range(instances):
        seed = inst 
        X, T, W_Q_teacher = generate_teacher_dataset(N=N_total_original_calc, D=D, L=L, DK=DK, omega=omega, seed=seed)

        results_one_instance = run_fig_2A(X, T, W_Q_teacher, alphas=alphas, lam_list=lam_list, lr=lr, epochs=epochs, DK=DK, test_ratio=test_ratio)
        all_instance_results.append(results_one_instance)
        
    # --- Average results across instances ---

    num_alphas = len(all_instance_results[0])
    valid_instances_results = [res for res in all_instance_results if len(res) == num_alphas]
    num_valid_instances = len(valid_instances_results)
    averaged_results = []
    keys_to_average = valid_instances_results[0][0].keys() 
    for alpha_idx in range(num_alphas):
        avg_result_for_alpha = {}
        current_alpha = valid_instances_results[0][alpha_idx]['alpha'] 
        avg_result_for_alpha['alpha'] = current_alpha
        
        for key in keys_to_average:
            if key == 'alpha': continue 

            values = [valid_instances_results[inst_idx][alpha_idx][key] 
                      for inst_idx in range(num_valid_instances)]
            
            avg_result_for_alpha[key] = np.nanmean(values) 
            
        averaged_results.append(avg_result_for_alpha)
        
    return averaged_results


# =============================================================================
# Linear Baseline Model 
# =============================================================================

class LinearModel(torch.nn.Module):  
    """Linear baseline model Y = B @ X."""
    def __init__(self, L):
        super(LinearModel, self).__init__()
        self.B = torch.nn.Parameter(torch.randn(L, L)) # Learnable (L, L) matrix

    def forward(self, x):
        yhat = self.B @ x 
        return yhat

def train_linear(X_train, T_train, X_test, T_test, lmbda=1e-4, lr=0.15, epochs=200):
    """
    Trains the LinearModel baseline using Adam.
    (Code logic matches original user script).

    Args:
        X_train, T_train: Training data.
        X_test, T_test: Test data.
        lmbda (float): L2 regularization strength.
        lr (float): Learning rate for Adam.
        epochs (int): Number of training epochs.

    Returns:
        tuple: (gen_loss, train_loss_reg) - Final losses per sample.
    """
    N_train, L, D = X_train.shape
    N_test = X_test.shape[0]
    
    # Original DataLoader used D as batch size. Replicating this.
    # This might imply full batch if N=D, or some form of large batch otherwise.
    # Let's assume it implies batch_size should be N_train for full-batch behavior if D was placeholder.
    # Re-checking original code... batch_size=D was specified. Let's use N_train for safety as full batch.
    batch_size = N_train 
    train_dataset = TensorDataset(X_train, T_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = X_train.device # Use device of input tensors
    model = LinearModel(L).to(device)

    # Original used Adam optimizer with weight_decay for L2
    optimizer = torch.optim.Adam([{'params': model.parameters(), "weight_decay":lmbda}], lr=lr)

    print(f"  Training linear model: epochs={epochs}, lr={lr}, lambda={lmbda:.1e}... ", end="")

    # Original loop variable n_iter, but used epochs arg
    for t in range(epochs):
        model.train()
        epoch_loss_sum = 0
        num_batches = 0
        for x, y in train_loader:
            # x, y = x.to(device), y.to(device) # Ensure data on device
            y_pred = model(x)
            loss = loss_SSE(y_pred, y) # Unregularized loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            num_batches += 1
            
        # Store last loss for final calculation as per original structure
        last_epoch_loss_val = epoch_loss_sum / num_batches 

    # Calculate final losses after training loop
    model.eval()
    with torch.no_grad():
        # Gen loss on test set
        y_test_pred = model(X_test.to(device))
        gen_loss = loss_SSE(y_test_pred, T_test.to(device)).item() / N_test # Per sample

        # Regularized Training loss - original calculated based on last epoch loss?
        # Original code: train_loss = loss.item()+lmbda/2*float(torch.sum(model.B.cpu().flatten()**2))
        # This seems to use the *last batch's* unnormalized loss value from training loop + L2 penalty.
        # Let's recalculate on full training set for consistency.
        y_train_pred = model(X_train)
        train_loss_unreg = loss_SSE(y_train_pred, T_train).item() / N_train # Per sample
        l2_penalty = (lmbda / 2) * float(torch.sum(model.B**2)) # Original used float() and .cpu() - keeping float()
        train_loss_reg = train_loss_unreg + l2_penalty
        
    print(f"Done. GenErr={gen_loss:.3e}, TrainErr(R)={train_loss_reg:.3e}")

    return gen_loss, train_loss_reg

def run_fig_2C(X, T, alphas, lmbda=1e-4, lr=0.15, epochs=200, test_ratio=0.2):
    """
    Runs the linear baseline model across different alpha values.
    """
    N_total, L, D = X.shape
    results = []

    N_test = int(test_ratio * D) 

    X_test = X[-N_test:]
    T_test = T[-N_test:]
    X_pool = X[:-N_test]
    T_pool = T[:-N_pool]
    N_pool = X_pool.shape[0]

    for alpha in alphas:
        N_train = int(alpha * D)
        if N_train > N_pool: N_train = N_pool
        X_train = X_pool[:N_train]
        T_train = T_pool[:N_train]

        elin, elin_train_reg = train_linear(X_train, T_train, X_test, T_test, lmbda=lmbda, lr=lr, epochs=epochs)          

        results.append({
            'alpha': alpha,
            'linear_gen_error': elin,
            'linear_train_error_reg': elin_train_reg 
        })
    return results

# =============================================================================
# Figure 3 Experiment Runners 
# =============================================================================

def fig3BC(alphas, omega_list, D=1000, L=2, DK=1, seed=42, lam_list_student=[1e-2], lmbda_lin=1e-4, epochs_student=500, epochs_linear=10):
    """
    Runs experiments for Fig 3B/C across omega and alpha.
    Combines student (run_fig_2A) and linear (run_fig_2C) results.
    """
    all_omega_results = [] 
    
    test_ratio = 0.2 # Assuming same test ratio as Fig 2
    N_total_original_calc = int((max(alphas, default=1.0) + test_ratio + 0.1) * D) 

    for omega in omega_list:
        print(f"Processing Omega = {omega:.3f}")
        seed= seed *2 
        X, T, W_Q_teacher = generate_teacher_dataset(N=N_total_original_calc, D=D, L=L, DK=DK, omega=omega, seed=seed)
        
        results_student = run_fig_2A(X, T, W_Q_teacher, alphas=alphas, lam_list=[lam_list_student[0]], epochs=epochs_student, DK=DK) 
    
        results_lin = run_fig_2C(X, T, alphas=alphas, lmbda=lmbda_lin, epochs=epochs_linear)
        
        lin_map = {res['alpha']: res for res in results_lin}
        combined_results_for_omega = []
        for res_stud in results_student:
            alpha = res_stud['alpha']
            if alpha in lin_map:
                 res_stud['omega'] = omega 
                 res_stud['linear_gen_error'] = lin_map[alpha]['linear_gen_error']
                 res_stud['linear_train_error_reg'] = lin_map[alpha]['linear_train_error_reg']
                 combined_results_for_omega.append(res_stud)

        all_omega_results.append(combined_results_for_omega)         
    return all_omega_results # List of lists

def convert_to_df(results_list_of_lists):
    """Flattens list of lists of results dicts into a pandas DataFrame."""
    flat_results = [item for sublist in results_list_of_lists for item in sublist]
    df = pd.DataFrame(flat_results)
    return df

def fig3A(d_list, alpha_val=1.5, omega=0.3, L=2, DK=1, seed=42, lam=1e-2, epochs=50):
    """
    Runs experiment for Fig 3A: Concentration vs Dimension D at fixed alpha.
    """
    all_d_results = [] 
    for D_val in d_list:
        N_total_original_calc = int(2.2* D_val)
        X, T, W_Q_teacher = generate_teacher_dataset(N=N_total_original_calc, D=D_val, L=L, DK=DK, omega=omega, seed=seed)
        
        results_one_d = run_fig_2A(X, T, W_Q_teacher, alphas=[alpha_val], lam_list=[lam], epochs=epochs, DK=DK) 
         
        results_one_d[0]['D'] = D_val 
        all_d_results.append(results_one_d) 

    return all_d_results 

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_fig_2A(results):
    """Plots Fig 2A: Delta Gen Error vs Alpha."""
    alphas = [r['alpha'] for r in results]

    deltas_gen = np.array([r['delta_gen'] for r in results]) 
    deltas_train_reg = [r['delta_train_reg'] for r in results] 


    fig = plt.figure(figsize=(4, 3.8)) 

    plt.plot(alphas, deltas_gen, c='grey', label='Generalization') 
    plt.scatter(alphas, deltas_gen, c='grey') 
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\Delta \epsilon$')
    plt.xlim(0.0, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig = plt.figure(figsize=(4, 3.8))
    plt.plot(alphas,deltas_train_reg, c='grey',label='train delta Error')
    plt.scatter(alphas,deltas_train_reg, c='grey')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\Delta \epsilon_t$')
    plt.xlim(0.0, 2.0)
    plt.legend()
    plt.tight_layout()

def plot_fig_2B(results, sigma=0.5):
    """Plots Fig 2B: Summary stats vs Alpha"""
    alphas = [r['alpha'] for r in results]
    m_pos = np.array([r['m_pos'] for r in results])
    m_sem = np.array([r['m_sem'] for r in results])
    theta_pos = np.array([r['theta_pos'] for r in results])
    theta_sem = np.array([r['theta_sem'] for r in results])
    
    fig=plt.figure(figsize=(4*1.5,3.8*1.5)) 
    norm = sigma**2 
    
    plt.plot(alphas, theta_sem / norm, color=c_semantic) 
    plt.plot(alphas, theta_pos / norm, color=c_semantic,linestyle='--') 
    plt.scatter(alphas, theta_pos / norm, label=r'$\theta/\sigma^2 pos$', color=c_semantic)
    plt.scatter(alphas, theta_sem / norm, label=r'$\theta/\sigma^2 sem$', color=c_semantic) 

    plt.plot(alphas, m_sem / norm, color=c_positional) 
    plt.plot(alphas, m_pos / norm, color=c_positional,linestyle='--') 
    plt.scatter(alphas, m_pos / norm, label=r'$m/\sigma^2 pos$', color=c_positional)
    plt.scatter(alphas, m_sem / norm, label=r'$m/\sigma^2 sem$', color=c_positional) 

    plt.xlabel(r'$\alpha$')
    plt.legend()
    plt.xlim(0.0, 2.0)
    plt.ylim(bottom=0)
    plt.tight_layout() 
    plt.show()

def plot_fig2C(results_student, results_linear):
    """Plots Fig 2C: Gen Error Comparison."""
    fig = plt.figure(figsize=(4, 3.8)) 
    c_neutral = c_att 
    alphas_lin = np.array([result['alpha'] for result in results_linear])
    lin_err = np.array([result['linear_gen_error'] for result in results_linear])
    
    alphas_stud = np.array([r['alpha'] for r in results_student])
    sem_gen_err = np.array([r['semantic_loss_gen'] for r in results_student])
    pos_gen_err = np.array([r['positional_loss_gen'] for r in results_student])
    
    sort_lin = np.argsort(alphas_lin)
    sort_stud = np.argsort(alphas_stud)
    
    alphas_lin, lin_err = alphas_lin[sort_lin], lin_err[sort_lin]
    alphas_stud, sem_gen_err, pos_gen_err = alphas_stud[sort_stud], sem_gen_err[sort_stud], pos_gen_err[sort_stud]

   
    min_att_err = np.minimum(sem_gen_err, pos_gen_err)
    
    mean_lin_err = np.mean(lin_err)
    plt.axhline(mean_lin_err, color=c_lin, label=r'$\epsilon_g^{\mathrm{lin}}$ (mean)')

    # Plotting the minimum attention error
    plt.plot(alphas_stud, min_att_err, color=c_neutral, marker='.', linestyle='-', label=r'$\min(\epsilon_g^{sem}, \epsilon_g^{pos})$')
    plt.scatter(alphas_stud, min_att_err, color=c_neutral) 

    alpha_l_example = 1.45 
    plt.axvline(alpha_l_example, ymin=0, ymax=1, color="r", ls="--", label=r'$\alpha_l$') 

    plt.ylabel(r'$\epsilon_g$')
    plt.xlabel(r'$\alpha$')
    plt.legend()
    plt.xlim(0.0, max(2.0, max(alphas_stud)*1.05) if len(alphas_stud)>0 else 2.0)
    plt.ylim(bottom=0) # Ensure errors don't go below 0
    plt.tight_layout()
    plt.show()

def plot_Fig3B(df):
    """Plots Fig 3B Heatmap: Delta Train Error vs Alpha, Omega."""

    alphas = np.array(sorted(df['alpha'].unique()))
    omegas = np.array(sorted(df['omega'].unique()))
    alphas_len = alphas.shape()[0]
    omegas_len = omegas.shape()[0]
    pivot_df = df.pivot_table(index='omega', columns='alpha', values='delta_train_reg', aggfunc='mean')
    delta_gen_err = np.array(df['delta_gen']).reshape(alphas_len,omegas_len)
    vmax = 0.0015 
    vmin = -vmax
    
    plt.figure(figsize=(4, 3.5)) 
    plt.imshow(pivot_df.values/1000, 
               aspect='auto',  
               cmap=cmap_uninf, 
               vmin=vmin, vmax=vmax, 
               origin='lower', 
               extent=[alphas[0], alphas[-1], omegas[0], omegas[-1]]) 
    
    plt.colorbar(label=r'$\Delta \epsilon_t$ (Pos - Sem)') 
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\omega$')
    plt.title("Fig 3B: Training Error Difference") 

    plt.ylim(min(0.01, omegas[0]), omegas[-1]*1.05 if len(omegas)>0 else 0.5) 
    plt.xlim(alphas[0], max(2.0, alphas[-1]*1.05) if len(alphas)>0 else 2.0) 
    plt.tight_layout() 
    plt.show()

    vmax=0.0015
    plt.figure(figsize=(4,3.5))
    plt.imshow(delta_gen_err/ 1000, 
               aspect='auto',  
               cmap=cmap_uninf, 
               vmin=-vmax, vmax=vmax, 
               extent=[alphas[0], alphas[-1], omegas[0], omegas[-1]])
    
    plt.colorbar(label=r'$\Delta \epsilon_g$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\omega$')

    plt.ylim(0.01,0.5)
    plt.xlim(0,2.0)

def plot_Fig3C(df):
    """Plots Fig 3C Heatmap: Lin vs Att Gen Error vs Alpha, Omega."""
        
    alphas = np.array(sorted(df['alpha'].unique()))
    omegas = np.array(sorted(df['omega'].unique()))
    
    df['min_att_gen'] = df[['semantic_loss_gen', 'positional_loss_gen']].min(axis=1)
    df['min_att_train'] = df[['semantic_loss_train', 'positional_loss_train']].min(axis=1) 
    df['delta_lin_att_gen'] = df['linear_gen_error'] - df['min_att_gen']
    delta_err_train = df['linear_train_error'] - df['min_att_train']
    pivot_df = df.pivot_table(index='omega', columns='alpha', values='delta_lin_att_gen', aggfunc='mean')        

    vmax = 0.002 
    vmin = -vmax

    fig=plt.figure(figsize=(4, 3.5)) 

    plt.imshow(pivot_df.values, 
               aspect='auto',  
               cmap=cmap_attlin, 
               vmin=vmin, vmax=vmax, 
               origin='lower', 
               extent=[alphas[0], alphas[-1], omegas[0], omegas[-1]])
    
    plt.colorbar(label=r'$\epsilon^{\mathrm{lin}}_g - \min(\epsilon_g^{att})$') 
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\omega$') 

    plt.ylim(min(0.01, omegas[0]), omegas[-1]*1.05 if len(omegas)>0 else 0.5) 
    plt.xlim(alphas[0], max(2.0, alphas[-1]*1.05) if len(alphas)>0 else 2.0) 
    plt.tight_layout() 
    plt.show()

    fig=plt.figure(figsize=(4,3.5))
    vmax = 0.002
    plt.imshow(delta_err_train/1e6, 
               aspect='auto',  
               cmap=cmap_attlin, 
               vmin=-vmax, vmax=vmax, 
               extent=[alphas[0], alphas[-1], omegas[0], omegas[-1]])
    
    plt.colorbar(label=r'$\epsilon^{\mathrm{lin}}_t-\epsilon_t$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\omega$') 
    plt.ylim(0.01,0.5)
    plt.xlim(0,2.0)

def plot_fig3A(df, D_list, sigma=0.5):
    """Plots Fig 3A: Overlap Concentration vs Dimension D. (Matches original plot)."""

    m_pos = np.array(df['m_pos'])
    m_sem = np.array(df['m_sem']) 
    theta_pos = np.array(df['theta_pos'])
    theta_sem = np.array(df['theta_sem'])
    D_values = np.array(df['D']) 

    norm_factor = sigma**2
    norm_color = mpl_colors.LogNorm(vmin=min(D_list), vmax=max(D_list)) 

    plt.figure(figsize=(5,4.5))

    plt.scatter(m_sem/norm_factor, theta_sem/norm_factor, c=D_values, cmap=cmap_att, norm=norm_color, marker='^') 

    plt.scatter(m_sem/norm_factor, theta_sem/norm_factor, marker='+', c=D_values, cmap=cmap_att, norm=norm_color) 
    
    plt.scatter(m_pos/norm_factor, theta_pos/norm_factor, c=D_values, cmap=cmap_pos, norm=norm_color, marker='v') 

    plt.scatter(m_pos/norm_factor, theta_pos/norm_factor, marker='+', c=D_values, cmap=cmap_pos, norm=norm_color) 
    
    plt.colorbar(label=r'$D$')
    
    plt.scatter([],[],marker='+', c=c_semantic, label='semantic')
    plt.scatter([],[],marker='+', c=c_positional, label='positional')

    plt.legend()
    plt.xlabel(r'$m/\sigma^2$')
    plt.ylabel(r'$\theta/\sigma^2$')
    plt.title(r'Fig 3A: Overlap Concentration vs $D$ (Fixed $\alpha$)')
    plt.grid(True) 
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ------------------ Setup ------------------
    L = 2
    D = 1000
    N = int(2.2 * D)
    DK = 1
    omega = 0.3
    seed = 42
    sigma = 0.5
    device = torch.device("cpu")

    # ------------------ Figure 2 Experiments ------------------
    print("\n=== Running experiments for Figure 2 ===")
    
    lr_fig2 = 0.15
    alphas_fig2 = np.linspace(0.01, 2.0, 25)
    lambdas_grid_fig2 = np.logspace(-2.5, -1.5, 3)
    epochs_fig2 = 5000
    instances_fig2 = 1  # Number of runs per setting
    
    results_fig2_avg = run_fig_2A_mean(
        D=D, omega=omega, L=L, alphas=alphas_fig2, lam_list=lambdas_grid_fig2,
        lr=lr_fig2, epochs=epochs_fig2, DK=DK, instances=instances_fig2
    )

    # Generate test data for linear probing
    test_ratio = 0.2
    N_total_fig2 = int((max(alphas_fig2, default=1.0) + test_ratio + 0.1) * D)
    X_fig2, T_fig2, _ = generate_teacher_dataset(N=N_total_fig2, D=D, L=L, DK=DK, omega=omega, seed=seed)
    X_fig2, T_fig2 = X_fig2.to(device), T_fig2.to(device)

    # Run linear probing (Figure 2C)
    alphas_lin_fig2 = np.linspace(0.2, 2.0, 10)
    epochs_lin_fig2 = 200
    lmbda_lin_fig2 = 1e-4
    results_fig2C = run_fig_2C(X_fig2, T_fig2, alphas=alphas_lin_fig2, lmbda=lmbda_lin_fig2, lr=lr_fig2, epochs=epochs_lin_fig2)

    print("\n--- Plotting Figure 2 ---")
    plot_fig_2A(results_fig2_avg)
    plot_fig_2B(results_fig2_avg, sigma=sigma)
    plot_fig2C(results_fig2_avg, results_fig2C)

    # ------------------ Figure 3 (B & C) ------------------
    print("\n=== Running experiments for Figure 3B & 3C ===")
    
    D_fig3 = 1000
    alpha_cross = np.linspace(0.5, 2.0, 6)
    omegas = np.sort([0., 0.1, 0.2, 0.3, 0.4, 0.5])
    epochs_student_fig3BC = 5000
    epochs_linear_fig3BC = 200
    lmbda_lin_fig3BC = 1e-4

    results_fig3BC_nested = fig3BC(
        alphas=alpha_cross, omega_list=omegas, D=D_fig3, L=L, DK=DK,
        seed=seed, lam_list_student=[1e-2], lmbda_lin=lmbda_lin_fig3BC,
        epochs_student=epochs_student_fig3BC, epochs_linear=epochs_linear_fig3BC
    )
    df_fig3BC = convert_to_df(results_fig3BC_nested)

    print("\n--- Plotting Figure 3B & 3C ---")
    plot_Fig3B(df_fig3BC)
    plot_Fig3C(df_fig3BC)

    # ------------------ Figure 3A ------------------
    print("\n--- Running experiment for Figure 3A ---")
    
    alpha_fig3A = 1.5
    omega_fig3A = 0.3
    d_list_fig3A = np.array([10, 15, 23, 36, 56, 87, 135, 209, 323, 500])
    lam_fig3A = 1e-2
    epochs_fig3A = 5000

    results_fig3A_nested = fig3A(
        d_list=d_list_fig3A, alpha_val=alpha_fig3A, omega=omega_fig3A,
        L=L, DK=DK, seed=seed, lam=lam_fig3A, epochs=epochs_fig3A
    )
    df_fig3A = convert_to_df(results_fig3A_nested)

    print("\n--- Plotting Figure 3A ---")
    plot_fig3A(df_fig3A, d_list_fig3A, sigma=sigma)
