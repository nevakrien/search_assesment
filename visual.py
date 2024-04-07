import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Function to calculate binomial test p-value with higher precision
def calculate_p_value(N, k, Q, X):
    success_prob = k / N
    p_value = stats.binom_test(X, Q, success_prob, alternative='greater')
    return p_value, success_prob

# Function to generate a PDF plot with the beta distribution and mark expected success probability
def plot_pdf(success_prob, Q, X, ax=None):
    if ax is None:
        ax = plt.gca()

    # Generate a range of probability values
    P = np.linspace(0, 1, 1000)

    # Calculate beta distribution parameters
    alpha, beta = X + 1, Q - X + 1
    pdf = stats.beta.pdf(P, alpha, beta)

    ax.plot(P, pdf, label='Beta Distribution PDF')
    ax.axvline(x=success_prob, color='red', linestyle='--', label=f'Expected Success Probability ({success_prob:.4f})')
    #ax.set_xlabel('Probability of Successful Retrieval')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

# Main function to plot experiments
def plot_experiments(experiments):
    # Plotting
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10,len(experiments) * 5))

    for i, (name, params) in enumerate(experiments.items()):
        # Calculating p-value and success probability with more precision
        p_value, success_prob = calculate_p_value(**params)
        p_value_str = f"{p_value:.2e}" if p_value < 0.0001 else f"{p_value:.4f}"

        # Plotting PDF with expected success probability
        plot_pdf(success_prob, params['Q'], params['X'], ax=axes[i])

        # Adding experiment details to the plot
        axes[i].set_title(f"{name} - k: {params['k']}, p-value: {p_value_str}", fontsize=10)

    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


# # Example Data
# experiments = {
#     'english':{'k': 1, 'N': 10570, 'Q': 10570, 'X': 4506},
#     'hebrew':{'k': 1, 'N': 7455, 'Q': 7455, 'X': 2329},
#     'hebrew (multi lang)':{'k': 1, 'N': 7455, 'Q': 7455, 'X': 1671},
#     #'json_dump english': {'k': 1, 'N': 10570, 'Q': 10570, 'X': 4307},
#     #'json_dump hebrew': {'k': 1, 'N': 7455, 'Q': 7455, 'X': 1169},#1690
# }

experiments = {
    #'bge-large-en-v1.5':{'k': 1, 'N': 10570, 'Q': 10570, 'X': 1608},
    #'sentence-transformers-alephbert':{'k': 1, 'N': 7455, 'Q': 7455, 'X': 588},
    'heBERT':{'k': 1, 'N': 7455, 'Q': 7455, 'X': 416},
    'mymodel':{'k': 1, 'N': 7455, 'Q': 7455, 'X': 15},
    #'base-bert':{'k': 1, 'N': 10570, 'Q': 10570, 'X': 681},
}
# Run the function with your experiments
plot_experiments(experiments)
