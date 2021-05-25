import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from math import hypot
from itertools import combinations

# Read in the data

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
results = pd.read_pickle("Results/results_20_4000_True_2.p")


# ======================================================================================================================

# Define the Euclidean distance measure between two points in a 2D space
def distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    :param p1: list; the first point in the 2D space
    :param p2: list; the second point in the 2D space
    :return: float; the Euclidean distance
    """

    # Get the coordinates for both points
    x1, y1 = p1
    x2, y2 = p2

    # Return the Euclidean distance between the 2D points
    return hypot(x2 - x1, y2 - y1)


# ======================================================================================================================

# Start the analysis of the results data selected

# Initialise some empty lists to put in the average distances between the centroids and the two dimensional SD of the
# word categories
average_centroid_distances = []
average_sd = []

# Initialise empty lists for the average distance of the communicative and meta-communicative words to the middle of the
# 2D space
averages_com_runs = []
averages_meta_runs = []

# Initialise empty lists for the average distance of the communicative and meta-communicative words to their
# initialisation state
com_distance_start_end_average = []
meta_distance_start_end_average = []

# Initialise empty lists for the communicative and meta-communicative words
com_words = []
meta_words = []

# Iterate over the total number of runs to access every independent simulation run
for run in range(results.iloc[-1]["Simulation_run"] + 1):

    # Define the number of rounds used in the simulations
    n_rounds = results.iloc[0]["N_rounds"]

    # Calculate the number of states used in the simulation (start, middle (multiple middle states are possible), end)
    # based on the number of rounds
    n_states = (n_rounds / 500) + 1

    # The number of states for the older results (before saving lexicons after every 500 rounds)
    # n_states = 2

    # Define the number of words used in the simulations
    n_words = results["N_words"].iloc[0]

    # Calculate the start and end position in the results dataframe for every simulation
    end = int(2 * n_words * n_states)
    start_position = run * end
    end_position = ((run + 1) * end) - 2

    # Define the end position if you want an intermediate result
    # end_position = start_position + 43

    # print(results.iloc[end_position])

    # Define the start and end lexicons
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

    # ===============================================================================================================

    # Initialise empty lists to store the centroids and 2D SDs
    centroid_list = []
    average_distance_list = []
    for word_index in range(n_words):
        # Define the exemplars of the word
        exemplars = lexicon_end[word_index][0]
        # Plot the end state for all simulations
        # plt.scatter(*zip(*exemplars))

        # Gather the centroids for the first agent only, for the current word, for the end position of the simulation
        # and for the current simulation run
        centroid = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                               (results["State"] == "End") & (results["Simulation_run"] == run), "Centroid"]
        centroid_list.append(centroid.tolist())

        # Do the same for the 2D SDs
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
    # print("List of centroids: ", centroid_list)

    average_distance_list = list(chain.from_iterable(average_distance_list))
    # print("Average SD: ", sum(average_distance_list)/len(average_distance_list))

    # Save the plot of the end state of the simulations of the first agent
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # # plt.show()
    # plt.savefig("/Users/jacqueline/Documents/Onderzoeksassistentsschap/Simulations/Wedel_start/Wedel_4000/20_runs/"
    #             "amb_false/" + str(run + 1) + ".pdf")
    # plt.clf()

    # Calculate the distances between the word centroids
    centroid_list = list(chain(*centroid_list))
    centroid_distances = [distance(*combo) for combo in combinations(centroid_list, 2)]

    # Calculate the average centroids distance and 2D SD per simulation run
    average_centroid_distances.append(sum(centroid_distances) / len(centroid_distances))
    average_sd.append(sum(average_distance_list) / len(average_distance_list))

    # ==================================================================================================================

    # Save the plot of the average centroids distance
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

    # Save the plot of the average SD for a two dimensional space
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

    # ==================================================================================================================

    # Calculate the distance to the middle of the space for the different types of words: communicative vs
    # meta-communicative (continuers)

    distances_com = []
    distances_meta = []
    for word_index in range(n_words):
        # Calculate the exemplar distances to the middle for the communicative words
        if lexicon_end[word_index][1] == "C":
            exemplars = lexicon_end[word_index][0]
            com_words.append(exemplars)
            for exemplar in exemplars:
                distance_exemplar = distance(exemplar, [50, 50])
                distances_com.append(distance_exemplar)
        else:
            # Calculate the exemplar distances to the middle for the meta-communicative words
            exemplars = lexicon_end[word_index][0]
            meta_words.append(exemplars)
            for exemplar in exemplars:
                distance_exemplar = distance(exemplar, [50, 50])
                distances_meta.append(distance_exemplar)

    # Calculate the average distance to the middle of the space for the communicative and meta-communicative words
    average_distance_com = sum(distances_com) / len(distances_com)
    average_distance_meta = sum(distances_meta) / len(distances_meta)

    print("Com average distance: ", average_distance_com)
    print("Meta average distance: ", average_distance_meta)

    # Calculate the average distance to the middle of the space over all runs for the communicative and
    # meta-communicative words
    averages_com_runs.append(average_distance_com)
    averages_meta_runs.append(average_distance_meta)

    # ==================================================================================================================

    # 2D SD measure across all simulations: SD of first run to every other run compared for the communicative and
    # metacommunactive words

    com_distance_start_end = []
    meta_distance_start_end = []

    for word_index in range(n_words):
        # Calculate the distance between the start and end state of the centroid of the selected communicative word
        if lexicon_end[word_index][1] == "C":
            centroid_start = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                         (results["State"] == "Start") & (results["Simulation_run"] == run), "Centroid"]
            centroid_end = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                       (results["State"] == "End") & (results["Simulation_run"] == run), "Centroid"]
            distance_centroid = distance(centroid_start[0], list(chain(*centroid_end)))
            com_distance_start_end.append(distance_centroid)
        # Calculate the distance between the start and end state of the centroid of the selected meta-communicative word
        else:
            centroid_start = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                         (results["State"] == "Start") & (results["Simulation_run"] == run), "Centroid"]
            centroid_end = results.loc[(results["Word"] == word_index) & (results["Agent"] == 1) &
                                       (results["State"] == "End") & (results["Simulation_run"] == run), "Centroid"]
            distance_centroid = distance(centroid_start[0], list(chain(*centroid_end)))
            meta_distance_start_end.append(distance_centroid)

    # Calculate the average distance of the start and end states of the centroids over all words
    com_distance_start_end_average.append(sum(com_distance_start_end) / len(com_distance_start_end))
    meta_distance_start_end_average.append(sum(meta_distance_start_end) / len(meta_distance_start_end))

# Calculate the average distance of the start and end states of the centroids for the communicative and
# metacommunicative words over all runs
average_com_runs = sum(averages_com_runs) / len(averages_com_runs)
average_meta_runs = sum(averages_meta_runs) / len(averages_meta_runs)
print("Com average distance over all runs: ", average_com_runs)
print("Meta average distance over all runs: ", average_meta_runs)

# Print the average distance (averaged over words) between centroids of the initialisation versus the end for every word
print("Com average distance per run: ", com_distance_start_end_average)
print("Meta average distance per run: ", meta_distance_start_end_average)

# ======================================================================================================================

# Plot the distinct words over all the simulation runs per word (one plot per word to see how they move in the
# space)

fig, (ax1, ax2) = plt.subplots(1, 2)

for word in com_words:
    ax1.scatter(*zip(*word))

for word in meta_words:
    ax2.scatter(*zip(*word))

# # Plot the in between states as well
# sliced_results = results[results["Agent"] == 1]
#
# for index, row in sliced_results.iterrows():
#     if row["Lexicon"][1] == "C":
#         exemplars = row["Exemplars"]
#         plt.scatter(*zip(*exemplars))
#     else:
#         exemplars = row["Exemplars"]
#         plt.scatter(*zip(*exemplars))
# plt.show()

# Plot all the words for all runs
ax1.set_xlim([0, 100])
ax1.set_ylim([0, 100])

ax2.set_xlim([0, 100])
ax2.set_ylim([0, 100])

plt.show()