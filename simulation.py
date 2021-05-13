from agent import Agent
from production import Production
from perception import Perception
import random
import numpy as np
import math
import pandas as pd
import pickle


def simulation(n_rounds, n_words, n_dimensions, seed, n_exemplars=100, n_continuers=0, similarity_bias_word=True,
               similarity_bias_segment=True, noise=True, anti_ambiguity_bias=True):
    # Initialise agents
    lexicon, com_words, meta_com_words, indices_meta = Agent(n_words, n_dimensions, seed, n_exemplars, n_continuers). \
        generate_lexicon()
    lexicon2, com_words2, meta_com_words2, indices_meta2 = Agent(n_words, n_dimensions, seed, n_exemplars,
                                                                 n_continuers).generate_lexicon()

    # Store the state of the lexicons at the beginning for both agents
    start = pd.DataFrame(columns=["Simulation_run", "Agent", "Word", "Centroid", "Average_distance", "Lexicon",
                                  "Continuer_indices", "Similarity_bias_word", "Similarity_bias_segment", "Noise",
                                  "Anti_ambiguity_bias", "N_words", "N_dimensions", "Seed", "N_exemplars",
                                  "N_continuers", "N_rounds", "State"])
    start.loc[len(start)] = [None, 1, None, None, None, lexicon, indices_meta, similarity_bias_word,
                             similarity_bias_segment, noise, anti_ambiguity_bias, n_words, n_dimensions, seed,
                             n_exemplars, n_continuers, n_rounds, "Start"]
    start.loc[len(start)] = [None, 2, None, None, None, lexicon2, indices_meta2, similarity_bias_word,
                             similarity_bias_segment, noise, anti_ambiguity_bias, n_words, n_dimensions, seed,
                             n_exemplars, n_continuers, n_rounds, "Start"]

    # Start the simulation: i counts the number of runs. One run consists of one production and perception step
    i = 0
    while i < n_rounds:
        print("Run number: ", i)

        # Assign the roles to the agents
        if (i % 2) == 0:
            producer_lex = lexicon
            producer_com_words = com_words
            producer_meta_com_words = meta_com_words
            perceiver_lex = lexicon2
            perceiver_com_words = com_words2
            perceiver_meta_com_words = meta_com_words2
        else:
            producer_lex = lexicon2
            producer_com_words = com_words2
            producer_meta_com_words = meta_com_words2
            perceiver_lex = lexicon
            perceiver_com_words = com_words
            perceiver_meta_com_words = meta_com_words

        # print("Producer lex: ", producer_lex)
        # print("Perceiver lex: ", perceiver_lex)

        # One agent starts producing something: first the exemplars are selected for every word category
        targets, total_activations = Production(producer_lex, producer_com_words,
                                                producer_meta_com_words, n_words, n_dimensions,
                                                n_exemplars, n_continuers, similarity_bias_word,
                                                similarity_bias_segment, noise).select_exemplar()
        # print("Chosen targets: ", targets)

        # Then the biases are added to the selected exemplars
        target_exemplars = Production(producer_lex, producer_com_words, producer_meta_com_words, n_words, n_dimensions,
                                      n_exemplars, n_continuers, similarity_bias_word, similarity_bias_segment, noise) \
            .add_biases(targets, total_activations)
        # print("Bias added to targets: ", target_exemplars)

        # The other agent perceives the produced signals

        # First shuffle the signals so they are not always presented in the same word order
        random.shuffle(target_exemplars)

        for signal in target_exemplars:
            # First the similarity of the signal to every word category is calculated and a best fitting word category
            # is chosen accordingly
            index_max_sim, total_similarities = Perception(perceiver_lex, perceiver_com_words, perceiver_meta_com_words,
                                                           n_words, n_dimensions, n_exemplars, n_continuers,
                                                           anti_ambiguity_bias).similarity(signal)
            # print("Word category signal: ", index_max_sim)
            # print("Total similarities: ", total_similarities)

            # Then the anti-ambiguity bias is added and the signal is stored (or not) depending on its probability of
            # being stored. This probability is based on how well the signal fits within the chosen word category
            # relative to the other word categories
            if (i % 2) == 0:
                if anti_ambiguity_bias:
                    lexicon2 = Perception(perceiver_lex, perceiver_com_words, perceiver_meta_com_words, n_words,
                                          n_dimensions, n_exemplars, n_continuers, anti_ambiguity_bias). \
                        add_anti_ambiguity_bias(index_max_sim, total_similarities, signal)

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
                    lexicon = Perception(perceiver_lex, perceiver_com_words, perceiver_meta_com_words, n_words,
                                         n_dimensions, n_exemplars, n_continuers, anti_ambiguity_bias). \
                        add_anti_ambiguity_bias(index_max_sim, total_similarities, signal)
                # If there's no anti-ambiguity bias, the signal is stored within the best fitting word category
                # whatsoever
                else:
                    lexicon[index_max_sim][0].insert(0, signal)

                # Only the first 100 exemplars of a word are used
                lexicon[index_max_sim][0] = lexicon[index_max_sim][0][:100]

                # print("Stored signal: ", signal)
                # print("Lexicon word: ", lexicon[index_max_sim])
            # print("LEXICON word 1: ", lexicon[0])
            # print("LEXICON 2 word 1: ", lexicon2[0])
        i += 1

    return lexicon, lexicon2, indices_meta, indices_meta2, start


def simulation_runs(n_runs, n_rounds, n_words, n_dimensions, seed, n_exemplars=100, n_continuers=0,
                    similarity_bias_word=True, similarity_bias_segment=True, noise=True, anti_ambiguity_bias=True):

    # Turn off the warning
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    results = pd.DataFrame(columns=["Simulation_run", "Agent", "Word", "Centroid", "Average_distance",
                                    "Lexicon", "Continuer_indices", "Similarity_bias_word",
                                    "Similarity_bias_segment", "Noise", "Anti_ambiguity_bias", "N_words",
                                    "N_dimensions", "Seed", "N_exemplars", "N_continuers", "N_rounds", "State"])

    for n_run in range(n_runs):
        lexicon, lexicon2, indices_meta, indices_meta2, start = simulation(n_rounds, n_words, n_dimensions, seed,
                                                                           n_exemplars, n_continuers,
                                                                           similarity_bias_word,
                                                                           similarity_bias_segment, noise,
                                                                           anti_ambiguity_bias)

        for word_index in range(n_words):

            # Calculate the distance of the exemplars towards the mean as a dispersion of the data for the start
            # condition for both agents
            exemplars = start["Lexicon"][0][word_index][0]
            exemplars2 = start["Lexicon"][1][word_index][0]

            centroid = np.mean(exemplars, axis=0)

            total_distance = 0
            n = 1
            for exemplar in exemplars:
                x = exemplar[0]
                y = exemplar[1]
                distance = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
                total_distance += math.sqrt(distance)
                n += 1
            average_distance = total_distance / n

            centroid2 = np.mean(exemplars2, axis=0)

            total_distance = 0
            n = 1
            for exemplar in exemplars2:
                x = exemplar[0]
                y = exemplar[1]
                distance = ((x - centroid2[0]) ** 2) + ((y - centroid2[1]) ** 2)
                total_distance += math.sqrt(distance)
                n += 1
            average_distance2 = total_distance / n

            # Store the start condition for both agents
            start["Simulation_run"] = n_run
            start["Word"] = word_index
            start["Centroid"] = [centroid, centroid2]
            start["Average_distance"] = [average_distance, average_distance2]
            results = results.append(start)

            # Get the exemplars of the specified word
            exemplars = lexicon[word_index][0]
            exemplars2 = lexicon2[word_index][0]

            # Calculate the distance of the exemplars towards the mean as a dispersion of the data
            centroid = np.mean(exemplars, axis=0)

            total_distance = 0
            n = 1
            for exemplar in exemplars:
                x = exemplar[0]
                y = exemplar[1]
                distance = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
                total_distance += math.sqrt(distance)
                n += 1
            average_distance = total_distance / n

            results.loc[len(results)] = [n_run, 1, word_index, centroid, average_distance, lexicon, indices_meta,
                                         similarity_bias_word, similarity_bias_segment, noise, anti_ambiguity_bias,
                                         n_words, n_dimensions, seed, n_exemplars, n_continuers, n_rounds, "End"]

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

            # Store the results
            results.loc[len(results)] = [n_run, 2, word_index, centroid, average_distance, lexicon2, indices_meta2,
                                         similarity_bias_word, similarity_bias_segment, noise, anti_ambiguity_bias,
                                         n_words, n_dimensions, seed, n_exemplars, n_continuers, n_rounds, "End"]

    # Pickle the results
    filename = "results_" + str(n_continuers) + ".p"
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()

    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # plt.show()

    # Plot the end state of the words for the second agent
    # for word_index in range(n_words):
    #     exemplars = lexicon2[word_index][0]
    #     plt.scatter(*zip(*exemplars))
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # plt.show()
    #
    # print("Continuer word indices: ", indices_meta)
    # print("Continuer word indices 2: ", indices_meta2)
