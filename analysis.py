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
        # centroid = results["Centroid"].loc[results["Word"] == word_index and results["Agent"] == 1 and results["State"]=="End", "Centroid"]

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

    # Plot the end state for all simulations
    centroid_list = []
    average_distance_list = []
    for word_index in range(results["N_words"].iloc[0]):
        exemplars = lexicon_end[word_index][0]
        plt.scatter(*zip(*exemplars))

        results_slice = results[i-16:i]
        centroid = results_slice.loc[(results_slice["Word"] == word_index) & (results_slice["Agent"] == 1) &
                                           (results_slice["State"] == "End"), "Centroid"]
        centroid_list.append(centroid.tolist())
        average_distance = results_slice.loc[(results_slice["Word"] == word_index) & (results_slice["Agent"] == 1) &
                                           (results_slice["State"] == "End"), "Average_distance"]
        average_distance_list.append(average_distance.tolist())
    print(centroid_list)
    print(average_distance_list)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()