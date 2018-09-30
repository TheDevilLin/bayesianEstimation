import numpy as np
import math
import matplotlib.pyplot as plt


def flip_coins(coins):
    # 0 = tails, 1 = heads
    heads = np.sum(np.random.randint(2, size=coins))
    tails = coins - heads
    return np.array([tails, heads])


def flip_coins_manually(tails, heads):
    return np.array([tails, heads])


def likelihood(theta, flip_coins_result):
    likelihood_result = []
    heads = flip_coins_result[1]
    tails = flip_coins_result[0]
    for x in range(len(theta)):
        likelihood_result.append(math.pow(theta[x], heads) * math.pow(1 - theta[x], tails))

    return np.array(likelihood_result)


def marginal_probability(likelihood, prior):
    return np.sum(likelihood * prior)


def posterior(likelihood, prior):
    return np.true_divide(likelihood * prior, marginal_probability(likelihood, prior))


def plot_graph(prior, likelihood, posterior):
    xpos = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    ypos = np.arange(0.0, 1.1, 0.2)

    plt.bar(xpos, prior.tolist(), align='center')
    plt.yticks(ypos)
    plt.xlabel('Theta')
    plt.title('Prior')
    plt.show()

    plt.bar(xpos, likelihood.tolist(), align='center')
    plt.yticks(ypos)
    plt.xlabel('Theta')
    plt.title('Likelihood')
    plt.show()

    plt.bar(xpos, posterior.tolist(), align='center')
    plt.yticks(ypos)
    plt.xlabel('Theta')
    plt.title('Posterior')
    plt.show()

    print("Maximum Likelihood Estimation:\t Theta =", np.argmax(likelihood) / 10)
    print("Maximum a Posterior Estimation:\t Theta =", np.argmax(posterior) / 10)


def plot_posterior(posterior, flip_coins_result, index):
    xpos = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    ypos = np.arange(0.0, 1.1, 0.2)
    plt.bar(xpos, posterior.tolist(), align='center')
    plt.yticks(ypos)
    plt.xlabel('Theta')
    plt.title('Result for Index #{} : {} heads, {} tails'.format(index, flip_coins_result[1], flip_coins_result[0]))
    plt.show()
    print("Maximum a Posterior Estimation:\t Theta =", np.argmax(posterior) / 10)


def simulate_flip_coins(coins, times, prior):
    for x in range(1, times+1):
        flip_coins_result = flip_coins(coins)
        likelihood_result = likelihood(theta, flip_coins_result)
        posterior_result = posterior(likelihood_result, prior)
        if x % 10 == 0:
            plot_posterior(posterior_result, flip_coins_result, x)


if __name__ == '__main__':
    # number of coins
    coins = 5

    # theta range
    theta = np.arange(0, 1.1, 0.1)

    # Default Prior
    prior1 = np.array([1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11])
    prior2 = np.array([0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01])

    # Prior 1 with plots
    print("========== Prior 1 Begin ==========")
    print("Result for 2 heads, 8 tails")
    flip_coins_result = flip_coins_manually(8, 2)
    likelihood_result = likelihood(theta, flip_coins_result)
    posterior_result = posterior(likelihood_result, prior1)
    plot_graph(prior1, likelihood_result, posterior_result)
    print("========== Prior 1 Ended ==========\n")

    # Prior 2 with plots
    print("========== Prior 2 Begin ==========")
    print("Result for 2 heads, 8 tails")
    flip_coins_result = flip_coins_manually(8, 2)
    likelihood_result = likelihood(theta, flip_coins_result)
    posterior_result = posterior(likelihood_result, prior2)
    plot_graph(prior2, likelihood_result, posterior_result)
    print("========== Prior 2 Ended ==========\n")

    # Simulate 50 times prior 1
    print("========== Prior 1 Begin ==========")
    simulate_flip_coins(10, 50, prior1)
    print("========== Prior 1 Ended ==========\n")

    # Simulate 50 times prior 2
    print("========== Prior 2 Begin ==========")
    simulate_flip_coins(10, 50, prior2)
    print("========== Prior 2 Ended ==========\n")


