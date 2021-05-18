import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from math import hypot
from itertools import combinations

# Read the data
results = pd.read_pickle("results_0.p")
# print(results)
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_10000/results_0.p")
# print(results["Anti_ambiguity_bias"])

i = 0

# 9 is the number of simulation runs
for run in range(20):
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
    # print(centroid_list)

    average_distance_list = list(chain.from_iterable(average_distance_list))
    print("Average SD: ", sum(average_distance_list)/len(average_distance_list))

    # total_distance = 0
    # n = 1
    # for centroid in centroid_list:
    #     x = centroid[0][0]
    #     y = centroid[0][1]
    #     print(x)
    #     distance = ((x - centroid2[0]) ** 2) + ((y - centroid2[1]) ** 2)
    #     total_distance += math.sqrt(distance)
    #     n += 1
    # average_distance2 = total_distance / n


    def distance(p1, p2):
        """Euclidean distance between two points."""
        x1, y1 = p1
        x2, y2 = p2
        return hypot(x2 - x1, y2 - y1)


    centroid_list = list(chain(*centroid_list))
    print(centroid_list)
    centroid_distances = [distance(*combo) for combo in combinations(list_of_coords, 2)]


    # How to get to know which coordinates belongs to which word in space? (upper left, upper right etc.)
    # The two smallest x values and the two smallest y values? Like the one with the smallest x value and the highest y
    # is the left upper one

    # centroid_x = [item[0] for item in centroid_list]
    # centroid_y = [item[1] for item in centroid_list]
    #
    # sorted_x = sorted(centroid_x)
    # sorted_y = sorted(centroid_y)

    # index_smallest_x = centroid_x.index(sorted(centroid_x)[0])
    # index_small_x = centroid_x.index(sorted(centroid_x)[1])
    #
    # index_smallest_y = centroid_y.index(sorted(centroid_y)[0])
    # index_small_y = centroid_y.index(sorted(centroid_y)[1])

    # left_upper = centroid_x.index(sorted_x[0])

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()