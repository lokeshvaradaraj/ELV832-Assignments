import os
import gc
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm import tqdm
import multiprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

def generate_data(D=10000, teacher_seed=42, alpha_min=0.2, alpha_max=2):
    '''
    Teacher model (binary perceptron) from which data is generated.
    Generated data stored using numpy memmaps, float16 precision is used to save memory.
    '''
    print("Initializing data generation...")
    key = random.PRNGKey(teacher_seed)
    w_teacher = 2 * random.bernoulli(key, 0.5, shape=(D,)) - 1

    N_total = int((alpha_max + alpha_min) * D)
    print('Creating memory-mapped data...')
    X = np.memmap('./X_data.dat', dtype='float16', mode='w+', shape=(N_total, D))
    y = np.memmap('./y_data.dat', dtype='float16', mode='w+', shape=(N_total,))

    batch_size = min(D,int(1e4))
    num_batches = int((N_total + batch_size - 1) // batch_size)
    key = random.split(key, num_batches)

    @jit
    def process_batch(subkey, w_teacher):
        X_batch = (random.normal(subkey, (batch_size, D))).astype(jnp.float16)
        y_batch = jnp.sign(X_batch @ w_teacher).astype(jnp.float16)
        y_batch = jnp.where(y_batch == 0, 1, y_batch)
        return X_batch, y_batch

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N_total)
        current_batch_size = end - start
        print(f"Processing batch {start}-{end}")
        X_batch, y_batch = process_batch(key[i], w_teacher)
        X[start:end] = np.asarray(X_batch[:current_batch_size])
        y[start:end] = np.asarray(y_batch[:current_batch_size])

    del w_teacher, key, X_batch, y_batch
    gc.collect()
    print("Data generation completed.")

def calculate_cross_entropy_loss_batched(y_true, y_pred, batch_size=1024):
    '''
    compute CE loss in batches 
    '''
    epsilon = 1e-15
    total_loss = 0.0
    total_samples = len(y_true)
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_true = y_true[i:batch_end]
        batch_pred = y_pred[i:batch_end]
        
        batch_pred = np.clip(batch_pred, epsilon, 1 - epsilon)
        batch_loss = -(batch_true * np.log(batch_pred) + (1 - batch_true) * np.log(1 - batch_pred))
        total_loss += np.sum(batch_loss)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')

def batch_predict_proba(model, X, batch_size=1024):
    '''
    computes probability of class +1 in batches. 
    '''
    total_samples = len(X)
    probas = np.zeros(total_samples)
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch = X[i:batch_end]
        probas[i:batch_end] = model.predict_proba(batch)[:, 1]
    return probas

def batch_predict(model, X, batch_size):
    '''
    Predicts the class given X using a trained logistic regression model in batches.
    '''
    total_samples = len(X)
    predictions = np.zeros(total_samples, dtype=int)
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch = X[i:batch_end]
        predictions[i:batch_end] = model.predict(batch)
    
    return predictions

def train_model_for_alpha(alpha, D, teacher_seed, X_train, y_train, X_test, y_test, kf, reg_values, batch_size, num_epochs, n_splits):
    '''
    Trains a logistic regression model for given sample complexity (alpha).
    Hyperparameter tuning is done for the l2 regularisation hyperparameter using K-fold CV.
    MSE on test dataset computed and returned along with best l2 regularisation hyperparameter and alpha.
    '''
    n_samples = int(alpha * D)
    X_subset, y_subset = X_train[:n_samples], y_train[:n_samples]
    
    best_model = None
    best_loss = float('inf')
    best_reg_value = None
    
    for reg in tqdm(reg_values, desc=f"Alpha {alpha:.3g}"):
        avg_loss = 0
        # Logistic regression model with l2 regularisation
        model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=reg,
            max_iter=1,
            tol=None,
            random_state=teacher_seed,
            warm_start=True,
            n_jobs=-1,
            learning_rate='optimal'
        )
        
        for train_idx, val_idx in kf.split(X_subset):
            for epoch in range(num_epochs):
                for batch_start in range(0, len(train_idx), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_idx))
                    batch_train_idx = train_idx[batch_start:batch_end]
                    
                    if len(batch_train_idx) > 0:
                        model.partial_fit(
                            X_subset[batch_train_idx],
                            y_subset[batch_train_idx],
                            classes=[-1, 1]
                        )
            
            prob_pred = batch_predict_proba(model, X_subset[val_idx], batch_size=batch_size)
            val_labels_binary = (y_subset[val_idx] + 1) // 2
            loss = calculate_cross_entropy_loss_batched(val_labels_binary, prob_pred, batch_size=batch_size)
            avg_loss += loss / n_splits
        # chooses the model with the best l2 reg hyperparameter giving the least CE loss for the validation set
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_reg_value = reg
            best_model = model
    
    # compute MSE for the test dataset
    y_test_pred = batch_predict(best_model, X_test, batch_size=batch_size)
    test_error = np.mean((y_test - y_test_pred) ** 2)
    print(f'alpha: {alpha:3g}, test_error: {test_error:3g}')
    return alpha, best_reg_value, test_error

def optimized_training(D, teacher_seed, alpha_max=2, alpha_min=0.2, alpha_step=0.2, 
                        n_splits=3, batch_size=4096, num_epochs=5):
    '''
    We run our experiment for alpha ranging from alpha_min - alpha_max with alpha_step step size.
    Parallelisation is done for optimised training. We return the alphas and  test_errors for each alpha. 
    '''
    N_total = int((alpha_max + alpha_min) * D)
    X = np.memmap('./X_data.dat', dtype='float16', mode='r', shape=(N_total, D))
    y = np.memmap('./y_data.dat', dtype='float16', mode='r', shape=(N_total,))
    
    train_size = int(alpha_max * D)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    alphas = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
    # l2 regularisation hyperparameter over which we do K-fold CV
    reg_values = np.logspace(-2, 0, 5)
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    # parallelised code
    with parallel_backend("loky"):
        results = Parallel(n_jobs=-1)(
            delayed(train_model_for_alpha)(alpha, D, teacher_seed, X_train, y_train, X_test, y_test, kf, reg_values, batch_size, num_epochs, n_splits)
            for alpha in tqdm(alphas, desc="Training Progress")
        )
        
    alphas, best_reg_values, test_errors = zip(*results)
    
    for alpha, best_reg, error in results:
        print(f"Alpha: {alpha:.3g} | Test Error: {error:.3g}")
    
    return alphas, test_errors

def plot_results(alphas, errors_list, labels, D, name):
    '''
    Plots sample complexity vs generalization error and saves the plot in figures.
    '''
    plt.figure(figsize=(10, 6))
    for errors, label in zip(errors_list, labels):
        plt.plot(alphas, errors, 'o-', label=label)
    plt.xlabel('Sample Complexity (α)')
    plt.ylabel('Generalization Error (MSE)')
    plt.title(f'Generalization Error vs α (D={D})')
    plt.grid(True)
    plt.ylim(0, )
    plt.legend()

    # saving plots
    os.makedirs('./figures', exist_ok=True)
    filename = f'./figures/generalization_error_D{D}_{name}.png'
    plt.savefig(filename)

    print(f"Plot saved as {filename}")

def D1e5_experiment():
    # Experiment parameters
    D = int(1e5)    
    teacher_seed = 24

    generate_data(D=D, teacher_seed=teacher_seed)
    
    alphas, errors = optimized_training(D=D, teacher_seed=teacher_seed)
    plot_results(
        alphas, 
        [errors], 
        [f'seed = {teacher_seed}'], 
        D=D, 
        name='teacher'
    )
    os.remove('./X_data.dat')
    os.remove('./y_data.dat')

def D1e4_experiment():
    # Experiment parameters
    D = int(1e4)    
    teacher_seed1,teacher_seed2 = 24,72
    generate_data(D=D, teacher_seed=teacher_seed1)
    alphas, errors1 = optimized_training(D=D, teacher_seed=teacher_seed1)

    os.remove('./X_data.dat')
    os.remove('./y_data.dat')

    generate_data(D=D, teacher_seed=teacher_seed2)
    _, errors2 = optimized_training(D=D, teacher_seed=teacher_seed2)

    plot_results(alphas, [errors1, errors2], [f'Teacher 1, seed = {teacher_seed1}', f'Teacher 2, seed = {teacher_seed2}'], D=D, name='combined')
    plot_results(alphas, [errors1], [f'Teacher 1, seed = {teacher_seed1}'], D=D, name='teacher 1')
    plot_results(alphas, [errors2], [f'Teacher 2, seed = {teacher_seed2}'], D=D, name='teacher 2')
    os.remove('./X_data.dat')
    os.remove('./y_data.dat')

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    D1e4_experiment()
    D1e5_experiment()