import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from math import hypot
from itertools import combinations

# Read the data

# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_10000/results_0.p")
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Results/20_runs_10000/"
#                          "results_0.p")
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Results/20_runs_10000/"
#                          "results_20_10000_False_0.02.p")
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Results/"
#                          "results_20_10000_True_0.069.p")
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Results/20_2500/"
#                          "results_20_2500_True_0.02.p")
# results = pd.read_pickle("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Results/20_runs_4000/"
#                          "results_0.p")
results = pd.read_pickle("results_3_1000_True.p")

# ======================================================================================================================

# Define the Euclidean distance measure between two points in a 2D space
def distance(p1, p2):
    """Euclidean distance between two points."""

    x1, y1 = p1
    x2, y2 = p2
    return hypot(x2 - x1, y2 - y1)


# ======================================================================================================================

# Start the analysis of the results data selected

# Initialise some empty lists to put in the average distances between the centroids and the two dimensional SD of the
# word categories
average_centroid_distances = []
average_sd = []

averages_com_runs = []
averages_meta_runs = []

for run in range(results.iloc[-1]["Simulation_run"] + 1):

    n_rounds = results.iloc[0]["N_rounds"]
    n_states = (n_rounds / 500) + 1

    # The number of states for the older results (before saving lexicons after every 500 rounds)
    # n_states = 2

    n_words = results["N_words"].iloc[0]
    end = int(2 * n_words * n_states)
    start_position = run * end
    end_position = ((run+1) * end) - 2

    # The end position if you want an intermediate result
    # end_position = start_position + 43

    # print(results.iloc[end_position])

    lexicon_start = results["Lexicon"].iloc[start_position]
    lexicon_end = results["Lexicon"].iloc[end_position]

    # Plot the beginning first
    # for word_index in range(n_words):
    #     exemplars = lexicon_start[word_index][0]
        # plt.scatter(*zip(*exemplars))
        # centroid = results["Centroid"].loc[results["Word"] == word_index and results["Agent"] == 1 and
    # results["State"]=="End", "Centroid"]

    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # plt.show()

# ======================================================================================================================

    # Plot the end state for all simulations
    centroid_list = []
    average_distance_list = []
    for word_index in range(n_words):
        exemplars = lexicon_end[word_index][0]
        plt.scatter(*zip(*exemplars))

        centroid = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                     (results["State"] == "End") & (results["Simulation_run"] == run), "Centroid"]
        centroid_list.append(centroid.tolist())
        average_distance = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                       (results["State"] == "End") & (results["Simulation_run"] == run),
                                       "Average_distance"]

        # The centroids and average distance measures for intermediate rounds
        # centroid = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
        #                        (results["N_rounds"] == 4000) & (results["Simulation_run"] == run), "Centroid"]
        # centroid_list.append(centroid.tolist())
        # average_distance = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
        #                                (results["N_rounds"] == 4000) & (results["Simulation_run"] == run),
        #                                "Average_distance"]

        average_distance_list.append(average_distance.tolist())
    # print(centroid_list)

    average_distance_list = list(chain.from_iterable(average_distance_list))
    # print("Average SD: ", sum(average_distance_list)/len(average_distance_list))

    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # # plt.show()
    # plt.savefig("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_start/Wedel_4000/20_runs/"
    #             "amb_false/" + str(run + 1) + ".pdf")
    # plt.clf()

    # print(centroid_list)
    centroid_list = list(chain(*centroid_list))
    # print(centroid_list)
    centroid_distances = [distance(*combo) for combo in combinations(centroid_list, 2)]
    # print(centroid_distances)
    average_centroid_distances.append(sum(centroid_distances) / len(centroid_distances))
    # print("Average centroid distance: ", average_centroid_distances)
    average_sd.append(sum(average_distance_list) / len(average_distance_list))

# ======================================================================================================================

# Plot the average centroids distance
# r = list(range(1, results.iloc[-1]["Simulation_run"] + 2))
# plt.bar(x=r, height=average_centroid_distances)
# plt.ylim(0, 50)
# plt.xticks(r)
# plt.xlabel("Simulation run")
# plt.ylabel("Average centroids distance")
# # plt.show()
# plt.savefig("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_start/Wedel_4000/20_runs/"
#             "amb_false/centroid.pdf")
# plt.clf()

# Plot the average SD for a two dimensional space
# r = list(range(1, results.iloc[-1]["Simulation_run"] + 2))
# plt.bar(x=r, height=average_sd)
# plt.ylim(0, 5)
# plt.xticks(r)
# plt.xlabel("Simulation run")
# plt.ylabel("Average distance of exemplars to centroids")
# # plt.show()
# plt.savefig("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_start/Wedel_4000/20_runs/"
#             "amb_false/sd.pdf")
# plt.clf()

# ======================================================================================================================

# Distance to the middle of the space for the different types of words: communicative vs metacommunicative (continuers)

    com_words = []
    meta_words = []
    distances_com = []
    distances_meta = []
    for word_index in range(n_words):
        if lexicon_end[word_index][1] == "C":
            exemplars = lexicon_end[word_index][0]
            com_words.append(exemplars)
            for exemplar in exemplars:
                distance_exemplar = distance(exemplar, [50, 50])
                distances_com.append(distance_exemplar)
        else:
            exemplars = lexicon_end[word_index][0]
            meta_words.append(exemplars)
            for exemplar in exemplars:
                distance_exemplar = distance(exemplar, [50, 50])
                distances_meta.append(distance_exemplar)

    average_distance_com = sum(distances_com)/len(distances_com)
    average_distance_meta = sum(distances_meta)/len(distances_meta)

    print("Com average distance: ", average_distance_com)
    print("Meta average distance: ", average_distance_meta)

    # Average over all runs
    averages_com_runs.append(average_distance_com)
    averages_meta_runs.append(average_distance_meta)

average_com_runs = sum(averages_com_runs)/len(averages_com_runs)
average_meta_runs = sum(averages_meta_runs)/len(averages_meta_runs)
print("Com average distance over all runs: ", average_com_runs)
print("Meta average distance over all runs: ", average_meta_runs)

# ======================================================================================================================

# 2D SD measure across all simulations: SD of first run to every other run compared for the communicative and
# metacommunactive words


# ======================================================================================================================

# Plot the distinct words over all the simulation runs per word (one plot per word to see how they move in the space)


# ======================================================================================================================

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
