import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import gc
import os

def generate_data(D=10000, teacher_seed=42, alpha_min=0.2, alpha_max=2):
    """
    Generates dataset using memory-mapped files.
    Creates memory-mapped data files with features and labels based on a random teacher model.
    """
    np.random.seed(teacher_seed)
    
    # Generate teacher weights
    w_teacher = 2 * np.random.binomial(1, 0.5, size=D) - 1 # (D,)

    # Calculate total number of samples
    n_total = int((alpha_max + alpha_min) * D)
    
    # Create memory-mapped files for X and y
    x_filepath = './X_data.dat'
    y_filepath = './y_data.dat'
    
    X = np.memmap(x_filepath, dtype='float16', mode='w+', shape=(n_total, D)) #(N,D)
    y = np.memmap(y_filepath, dtype='float16', mode='w+', shape=(n_total,))   #(N,)

    # Process data in batches
    batch_size = min(1e4,D)
    num_batches =int( (n_total + batch_size - 1) // batch_size)
    
    for i in range(num_batches):
        # Calculate current batch parameters
        start =int(i * batch_size)
        end = int(min((i + 1) * batch_size, n_total))
        current_batch_size = int(end - start)
        
        # Generate batch data
        X_batch = np.random.normal(0, 1, (current_batch_size, D)).astype(np.float16)
        y_batch = np.sign(X_batch @ w_teacher).astype(np.float16)
        y_batch[y_batch == 0] = 1  # Ensure no zero labels
        
        # Store batch data
        X[start:end] = X_batch
        y[start:end] = y_batch
    
    # Clean up memory
    del w_teacher
    gc.collect()
    
    return x_filepath, y_filepath

def run_experiment(D=10000, teacher_seed=42):
    """
    Trains logistic regression models on subsets of data of varying size and evaluates test error.
    """
    # Calculate total and train/test split
    n_total = int(2.2 * D)
    X = np.memmap('./X_data.dat', dtype='float16', mode='r', shape=(n_total, D))
    y = np.memmap('./y_data.dat', dtype='float16', mode='r', shape=(n_total,))
    
    # Split dataset
    X_train, X_test = X[:2*D], X[2*D:]
    y_train, y_test = y[:2*D], y[2*D:]
    
    # Clean up memory
    del X, y
    gc.collect() 
    
    # Define sample complexity range
    alphas = np.arange(0.2, 2.2, 0.2)
    test_errors = []

    # Hyperparameter grid for Logistic Regression
    regularization_params = np.logspace(-3, 3, 7) 

    for alpha in alphas:
        print("alpha = ",alpha)
        n = int(alpha * D)
        X_subset = X_train[:n]
        y_subset = y_train[:n]

        # Train Logistic Regression with cross-validation
        model = LogisticRegressionCV(
            Cs=regularization_params, 
            cv=3, 
            penalty='l2', 
            solver='saga',
            max_iter=100000, 
            tol=1e-3, 
            random_state=teacher_seed, 
            n_jobs=-1
        )
        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_test)

        # Compute classification error (MSE)
        test_error = np.mean(np.square(y_test - y_pred)) 
        test_errors.append(test_error)

    return alphas, test_errors

def plot_results(alphas, errors_list, labels, D, name):
    """
    Creates a plot of test errors (MSE) for given sample complexity values and saves it.
    """
    # Create figures directory if it doesn't exist
    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    plt.figure(figsize=(10, 6))
    for errors, label in zip(errors_list, labels):
        plt.plot(alphas, errors, 'o-', label=label)
    
    plt.xlabel('Sample Complexity (α)')
    plt.ylabel('Test Error')
    plt.title(f'Generalization Error vs α (D={D})')
    plt.grid(True)
    plt.ylim(0,)
    plt.legend()

    # Save figure 
    filename = f'./figures/generalization_error_D{D}_{name}.png'
    plt.savefig(filename)
    plt.close()

def D1e5_experiment():
    # Experiment parameters
    D = int(1e5)    
    teacher_seed = 42 

    generate_data(D=D, teacher_seed=teacher_seed)
    
    alphas, errors = run_experiment(D=D, teacher_seed=teacher_seed)
    plot_results(
        alphas, 
        [errors], 
        [f'seed = {teacher_seed}'], 
        D=D, 
        name='teacher'
    )
    os.remove('X_data.dat')
    os.remove('y_data.dat')
    
def D1e4_experiment():
    # Experiment parameters
    D = int(3e1)    
    teacher_seed1,teacher_seed2 = 24,72
    generate_data(D=D, teacher_seed=teacher_seed1)
    alphas, errors1 = run_experiment(D=D, teacher_seed=teacher_seed1)

    os.remove('X_data.dat')
    os.remove('y_data.dat')

    generate_data(D=D, teacher_seed=teacher_seed2)
    _, errors2 = run_experiment(D=D, teacher_seed=teacher_seed2)

    plot_results(alphas, [errors1, errors2], [f'Teacher 1, seed = {teacher_seed1}', f'Teacher 2, seed = {teacher_seed2}'], D=D, name='combined')
    plot_results(alphas, [errors1], [f'Teacher 1, seed = {teacher_seed1}'], D=D, name='teacher 1')
    plot_results(alphas, [errors2], [f'Teacher 2, seed = {teacher_seed2}'], D=D, name='teacher 2')
    os.remove('X_data.dat')
    os.remove('y_data.dat')

D1e4_experiment()
# D1e5_experiment() (runs into memory issues)
