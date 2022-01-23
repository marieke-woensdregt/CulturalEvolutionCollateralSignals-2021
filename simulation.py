from agent import Agent
from production import Production
from perception2 import Perception
import random
import numpy as np
import math
import pandas as pd
import pickle
import copy
import sys
import time


def simulation(n_rounds, n_words, n_dimensions, seed, n_exemplars, n_continuers, similarity_bias_word,
               similarity_bias_segment, noise, anti_ambiguity_bias, continuer_G, word_similarity_weight, segment_similarity_weight, wedel_start, n_run):
    """
    Run a simulation of n_rounds rounds with the specified parameters.
    :param n_rounds: int; the number of rounds of the simulation run
    :param n_words: int; the number of v_words contained in the lexicon
    :param n_dimensions: int; the number of dimensions of the exemplars
    :param seed: int; a seed used to generate the agents' lexicons
    :param n_exemplars: int; the number of exemplars of a word
    :param n_continuers: int; the number of continuer v_words in the lexicon
    :param similarity_bias_word: boolean; whether the word similarity bias should be applied to the signals
    :param similarity_bias_segment: boolean; whether the segment similarity bias should be applied to the signals
    :param noise: boolean; whether noise should be added to the signals
    :param anti_ambiguity_bias: boolean; whether the anti-ambiguity bias should be applied to storing signals
    :param continuer_G: int; the constant used to determine the strength of the noise bias
    # :param word_ratio: float; the relative contribution of the word similarity bias in case of continuer v_words
    :param word_similarity_weight: float; the relative contribution of the word-similarity bias in case of continuer v_words
    :param segment_similarity_weight: float; the relative contribution of the segment-similarity bias in case of continuer v_words
    :param wedel_start: boolean; whether the means for initialising the lexicon are based on the one used in Wedel's
    model
    :param n_run: int; the current run number
    :return: list; the lexicon consists of a list of v_words, for which each word consists of a list of exemplars, which
                   in turn is a list of the number of dimensions floats
             list; the second agent's lexicon consists of a list of v_words, for which each word consists of a list of
                   exemplars, which in turn is a list of the number of dimensions floats
             list; the indices of the continuers in the lexicon
             list; the indices of the continuers in the lexicon of the second agent
             dataframe; a pandas dataframe containing the starting conditions
    """

    # Initialise agents

    # If the seed is defined use that seed to initialise the agents
    if seed:
        seed_value = seed
    else:
        # Else generate a random seed to use for the simulation
        seed_value = random.randrange(sys.maxsize)

    lexicon_start, v_words, continuer_words, indices_continuer = Agent(n_words, n_dimensions, seed_value, n_exemplars,
                                                                       n_continuers, wedel_start, n_run).\
        generate_lexicon()
    lexicon2_start, v_words2, continuer_words2, indices_continuer2 = Agent(n_words, n_dimensions, seed_value,
                                                                           n_exemplars, n_continuers, wedel_start,
                                                                           n_run).generate_lexicon()
    # print("Lexicon start: ", lexicon_start)
    # print("Continuer_words: ", continuer_words)
    #
    # print("Lexicon start 2: ", lexicon2_start)
    # print("Continuer_words 2: ", continuer_words2)

    # Store the state of the lexicons at the beginning for both agents
    start = pd.DataFrame(columns=["Simulation_run", "Agent", "Word", "Centroid", "Average_distance", "Lexicon",
                                  "Continuer_indices", "Similarity_bias_word", "Similarity_bias_segment", "Noise",
                                  "Anti_ambiguity_bias", "N_words", "N_dimensions", "Seed", "N_exemplars",
                                  "N_continuers", "N_rounds", "State", "Exemplars", "Store", "Probability_storages"])
    start.loc[len(start)] = [None, 1, None, None, None, lexicon_start, indices_continuer, similarity_bias_word,
                             similarity_bias_segment, noise, anti_ambiguity_bias, n_words, n_dimensions, seed,
                             n_exemplars, n_continuers, n_rounds, "Start", None, None, None]
    start.loc[len(start)] = [None, 2, None, None, None, lexicon2_start, indices_continuer2, similarity_bias_word,
                             similarity_bias_segment, noise, anti_ambiguity_bias, n_words, n_dimensions, seed,
                             n_exemplars, n_continuers, n_rounds, "Start", None, None, None]

    # Make a copy of the lexicon for the agents to use in conversation
    lexicon = copy.deepcopy(lexicon_start)
    lexicon2 = copy.deepcopy(lexicon2_start)

    # Start the simulation: i counts the number of runs. One run consists of one production and perception step
    i = 0

    # Store_count(2) counts how often the signal is stored when perceived for both agents
    store_count = 0
    store_count_2 = 0

    # Probability_storage(2) stores the probabilities for which a signal can be stored for both agents
    probability_storages = []
    probability_storages2 = []

    while i < n_rounds:
        if i % 100 == 0:
            print("Round number: ", i)

        # Assign the roles to the agents so they change role every round
        if (i % 2) == 0:
            producer_lex = lexicon
            producer_v_words = v_words
            producer_continuer_words = continuer_words
            perceiver_lex = lexicon2
            perceiver_v_words = v_words2
            perceiver_continuer_words = continuer_words2
        else:
            producer_lex = lexicon2
            producer_v_words = v_words2
            producer_continuer_words = continuer_words2
            perceiver_lex = lexicon
            perceiver_v_words = v_words
            perceiver_continuer_words = continuer_words

        # print("Producer lex: ", producer_lex)
        # print("Perceiver lex: ", perceiver_lex)

        # One agent starts producing something: first the exemplars are selected for every word category
        targets, total_activations = Production(producer_lex, producer_v_words,
                                                producer_continuer_words, n_words, n_dimensions,
                                                n_exemplars, n_continuers, similarity_bias_word,
                                                similarity_bias_segment, noise, continuer_G, word_similarity_weight, segment_similarity_weight).select_exemplar()
        # print("Chosen target exemplars: ", targets)

        # Then the biases are added to the selected exemplars
        target_exemplars = Production(producer_lex, producer_v_words, producer_continuer_words, n_words, n_dimensions,
                                      n_exemplars, n_continuers, similarity_bias_word, similarity_bias_segment, noise,
                                      continuer_G, word_similarity_weight, segment_similarity_weight).add_biases(targets, total_activations)
        # print("Bias added to targets: ", target_exemplars)

        # The other agent perceives the produced signals

        # First shuffle the signals so they are not always presented in the same word order
        random.shuffle(target_exemplars)

        for signal in target_exemplars:
            # First the similarity of the signal to every word category is calculated and a best fitting word category
            # is chosen accordingly
            index_max_sim, total_similarities = Perception(perceiver_lex, perceiver_v_words, perceiver_continuer_words,
                                                           n_words, n_dimensions, n_exemplars, n_continuers,
                                                           anti_ambiguity_bias).similarity(signal)
            # print("Word category signal: ", index_max_sim)
            # print("Total similarities: ", total_similarities)

            # Then the anti-ambiguity bias is added and the signal is stored (or not) depending on its probability of
            # being stored. This probability is based on how well the signal fits within the chosen word category
            # relative to the other word categories

            # The signal is stored in the lexicon of the agent being the perceiver this round
            if (i % 2) == 0:
                if anti_ambiguity_bias:
                    lexicon2, store, probability_storage = Perception(perceiver_lex, perceiver_v_words,
                                                                      perceiver_continuer_words, n_words, n_dimensions,
                                                                      n_exemplars, n_continuers, anti_ambiguity_bias). \
                        add_anti_ambiguity_bias(index_max_sim, total_similarities, signal)

                    probability_storages2.append(probability_storage)

                    if store is True:
                        store_count_2 += 1

                # If there's no anti-ambiguity bias, the signal is stored within the best fitting word category
                # whatsoever
                else:
                    lexicon2[index_max_sim][0].insert(0, signal)
                # Only the first 100 exemplars of a word are used
                lexicon2[index_max_sim][0] = lexicon2[index_max_sim][0][:100]
                # print("Stored signal: ", signal)
                # print("Lexicon word: ", lexicon2[index_max_sim])
            else:
                if anti_ambiguity_bias:
                    lexicon, store, probability_storage = Perception(perceiver_lex, perceiver_v_words,
                                                                     perceiver_continuer_words, n_words, n_dimensions,
                                                                     n_exemplars, n_continuers, anti_ambiguity_bias). \
                        add_anti_ambiguity_bias(index_max_sim, total_similarities, signal)

                    if store is True:
                        store_count += 1

                    probability_storages.append(probability_storage)

                # If there's no anti-ambiguity bias, the signal is stored within the best fitting word category
                # whatsoever
                else:
                    lexicon[index_max_sim][0].insert(0, signal)

                # Only the first 100 exemplars of a word are used
                lexicon[index_max_sim][0] = lexicon[index_max_sim][0][:100]

                # print("Stored signal: ", signal)
                # print("Lexicon word: ", lexicon[index_max_sim])

        # After every 500 rounds, store the agents' lexicons
        if i % 500 == 0 and i > 0:
            # Make a copy of the lexicon, and probability storages to store the intermediate results
            lexicon_middle = copy.deepcopy(lexicon)
            lexicon2_middle = copy.deepcopy(lexicon2)

            probability_storages_middle = copy.deepcopy(probability_storages)
            probability_storages2_middle = copy.deepcopy(probability_storages2)

            start.loc[len(start)] = [None, 1, None, None, None, lexicon_middle, indices_continuer, similarity_bias_word,
                                     similarity_bias_segment, noise, anti_ambiguity_bias, n_words, n_dimensions, seed,
                                     n_exemplars, n_continuers, i, "Middle", None, store_count,
                                     probability_storages_middle]
            start.loc[len(start)] = [None, 2, None, None, None, lexicon2_middle, indices_continuer2,
                                     similarity_bias_word, similarity_bias_segment, noise, anti_ambiguity_bias, n_words,
                                     n_dimensions, seed, n_exemplars, n_continuers, i, "Middle", None, store_count_2,
                                     probability_storages2_middle]

        i += 1

    return lexicon, lexicon2, indices_continuer, indices_continuer2, start, store_count, store_count_2, \
           probability_storages, probability_storages2


def simulation_runs(n_runs, n_rounds, n_words, n_dimensions, seed=None, n_exemplars=100, n_continuers=1,
                    similarity_bias_word=True, similarity_bias_segment=True, noise=True, anti_ambiguity_bias=True,
                    continuer_G=1250, word_similarity_weight=0.9, segment_similarity_weight=0.0, wedel_start=True):
    """
    Run n_runs simulations with the specified parameters and pickle and store the results as a dataframe.
    :param n_runs: int; the number of simulations run
    :param n_rounds: int; the number of rounds of the simulation run
    :param n_words: int; the number of v_words contained in the lexicon
    :param n_dimensions: int; the number of dimensions of the exemplars
    :param seed: int; a seed used to generate the agents' lexicons
    :param n_exemplars: int; the number of exemplars of a word
    :param n_continuers: int; the number of continuer v_words in the lexicon
    :param similarity_bias_word: boolean; whether the word similarity bias should be applied to the signals
    :param similarity_bias_segment: boolean; whether the segment similarity bias should be applied to the signals
    :param noise: boolean; whether noise should be added to the signals
    :param anti_ambiguity_bias: boolean; whether the anti-ambiguity bias should be applied to storing signals
    :param continuer_G: int; the constant used to determine the strength of the noise bias
    # :param word_ratio: float; the relative contribution of the word similarity bias in case of continuer v_words
    :param word_similarity_weight: float; the relative contribution of the word-similarity bias in case of continuer v_words
    :param segment_similarity_weight: float; the relative contribution of the segment-similarity bias in case of continuer v_words
    :param wedel_start: boolean; whether the means for initialising the lexicon are based on the one used in Wedel's
    model
    """

    t0 = time.time()

    # Turn off the warning
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Initialise the dataframe containing the results
    results = pd.DataFrame(columns=["Simulation_run", "Agent", "Word", "Centroid", "Average_distance",
                                    "Lexicon", "Continuer_indices", "Similarity_bias_word",
                                    "Similarity_bias_segment", "Noise", "Anti_ambiguity_bias", "N_words",
                                    "N_dimensions", "Seed", "N_exemplars", "N_continuers", "N_rounds", "State",
                                    "Exemplars", "Store", "Probability_storages"])

    # Run the simulations
    for n_run in range(n_runs):

        print('')
        print("Run number: ", n_run)

        lexicon, lexicon2, indices_continuer, indices_continuer2, start, store_count, store_count_2, \
        probability_storages, probability_storages2 = simulation(n_rounds, n_words, n_dimensions, seed, n_exemplars,
                                                                 n_continuers, similarity_bias_word,
                                                                 similarity_bias_segment, noise, anti_ambiguity_bias,
                                                                 continuer_G, word_similarity_weight, segment_similarity_weight, wedel_start, n_run)

        # Calculate some measures for all the rows of the start dataframe (containing the starting condition of a
        # simulation run)
        for row in range(start.shape[0]):

            for word_index in range(n_words):

                # Select the exemplars
                exemplars = start["Lexicon"][row][word_index][0]

                # Calculate the centroid of the word category
                centroid = np.mean(exemplars, axis=0)

                # Calculate the distance of the exemplars towards the centroid as a dispersion of the data for the start
                # condition for both agents
                total_distance = 0
                n = 1
                for exemplar in exemplars:
                    x = exemplar[0]
                    y = exemplar[1]
                    distance = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
                    total_distance += math.sqrt(distance)
                    n += 1
                average_distance = total_distance / n

                # Store the start/in between conditions for both agents
                start.at[row, "Simulation_run"] = n_run
                start.at[row, "Word"] = word_index
                start.at[row, "Centroid"] = centroid
                start.at[row, "Average_distance"] = average_distance
                start.at[row, "Exemplars"] = np.array([exemplars], dtype=object)

                # Add the starting conditions to the results
                results = results.append(start.iloc[row])

        # Store the end state results
        for word_index in range(n_words):

            # Get the exemplars of the specified word at the end state
            exemplars = lexicon[word_index][0]
            exemplars2 = lexicon2[word_index][0]

            # Calculate the mean of the word exemplars
            centroid = np.mean(exemplars, axis=0)

            # Calculate the distance of the exemplars towards the mean as a dispersion of the data
            total_distance = 0
            n = 1
            for exemplar in exemplars:
                x = exemplar[0]
                y = exemplar[1]
                distance = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
                total_distance += math.sqrt(distance)
                n += 1
            average_distance = total_distance / n

            # Store the results in the results dataframe
            results.loc[len(results)] = [n_run, 1, word_index, centroid, average_distance, lexicon, indices_continuer,
                                         similarity_bias_word, similarity_bias_segment, noise, anti_ambiguity_bias,
                                         n_words, n_dimensions, seed, n_exemplars, n_continuers, n_rounds, "End",
                                         exemplars, store_count, probability_storages]

            # Calculate the distance of the exemplars towards the mean as a dispersion of the data for the second agent
            centroid = np.mean(exemplars2, axis=0)

            total_distance = 0
            n = 1
            for exemplar in exemplars2:
                x = exemplar[0]
                y = exemplar[1]
                distance = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
                total_distance += math.sqrt(distance)
                n += 1
            average_distance = total_distance / n

            # Store the results for the second agent
            results.loc[len(results)] = [n_run, 2, word_index, centroid, average_distance, lexicon2, indices_continuer2,
                                         similarity_bias_word, similarity_bias_segment, noise, anti_ambiguity_bias,
                                         n_words, n_dimensions, seed, n_exemplars, n_continuers, n_rounds, "End",
                                         exemplars2, store_count_2, probability_storages2]

    results["Continuer_G"] = continuer_G
    # results["Word_ratio"] = word_ratio
    results["Word_similarity_weight"] = word_similarity_weight
    results["Segment_similarity_weight"] = segment_similarity_weight

    # Pickle the results
    filename = "results_" + str(n_runs) + "_" + str(n_rounds) + "_" + str(anti_ambiguity_bias) + "_" + \
               str(n_continuers) + "_" + str(continuer_G) + "_" + str(word_similarity_weight) + "_" + str(segment_similarity_weight) + "_" + str(n_words) + "_" + str(wedel_start) + ".p"
    outfile = open(filename, "wb")
    pickle.dump(results, outfile)
    outfile.close()


    t1 = time.time()

    simulation_time = t1-t0
    print("Simulation runs took "+str(round((simulation_time/60), 2))+" min. in total")
