import matplotlib.pyplot as plt
import pandas as pd

# Read the data
results = pd.read_pickle("results_0.p")
print(results)

i = 0
for run in range(9):
    # Plot the beginning and end for the first agent
    lexicon_start = results["Lexicon"].iloc[i]
    lexicon_end = results["Lexicon"].iloc[i + 2]

    # Skip all the data from each individual word and the second agent
    i += 16

    # Plot the beginning first
    for word_index in range(results["N_words"].iloc[0]):
        exemplars = lexicon_start[word_index][0]
        plt.scatter(*zip(*exemplars))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

    # Plot the end state for all simulations
    for word_index in range(results["N_words"].iloc[0]):
        exemplars = lexicon_end[word_index][0]
        plt.scatter(*zip(*exemplars))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()