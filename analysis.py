import matplotlib.pyplot as plt
import pandas as pd

# Read the data
results = pd.read_pickle("results_0.p")
print(results)

# Plot the beginning
i = 0
for run in range(9):
    lexicon_beginning = results["Lexicon"][i]
    lexicon_end = results["Lexicon"][i + 7]

    # Skip all the data from each individual word and the second agent
    i += 8

    for word_index in range(results["N_words"][0]):
        exemplars = lexicon_beginning[word_index][0]
        plt.scatter(*zip(*exemplars))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

    # Plot the end state for all simulations
    for word_index in range(results["N_words"][0]):
        exemplars = lexicon_end[word_index][0]
        plt.scatter(*zip(*exemplars))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()