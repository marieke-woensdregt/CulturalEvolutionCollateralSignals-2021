from agent import Agent
from production import Production
from perception import Perception
import matplotlib.pyplot as plt
import random


def simulation(n_runs, n_words, n_dimensions, seed, n_exemplars=100, n_continuers=0, similarity_bias_word=True,
               similarity_bias_segment=True, noise=True, anti_ambiguity_bias=True):
    # Initialise agents
    lexicon, com_words, meta_com_words = Agent(n_words, n_dimensions, seed).generate_lexicon()
    lexicon2, com_words2, meta_com_words2 = Agent(n_words, n_dimensions, seed).generate_lexicon()

    # Start the simulation: i counts the number of runs. One run consists of one production and perception step
    i = 0
    while i < n_runs:
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

        # One agent starts producing something: first the exemplars are selected for every word category
        targets, activation_exemplars, total_activations = Production(producer_lex, producer_com_words,
                                                                      producer_meta_com_words, n_words, n_dimensions,
                                                                      n_exemplars, n_continuers, similarity_bias_word,
                                                                      similarity_bias_segment, noise).select_exemplar()
        #print("Chosen targets: ", targets)

        # Then the biases are added to the selected exemplars
        target_exemplars = Production(producer_lex, producer_com_words, producer_meta_com_words, n_words, n_dimensions,
                                      n_exemplars, n_continuers, similarity_bias_word, similarity_bias_segment, noise) \
            .add_biases(targets, activation_exemplars, total_activations)
        #print("Bias added to targets: ", target_exemplars)

        # The other agent perceives the produced signals

        # First shuffle the signals so they are not always presented in the same word order
        random.shuffle(target_exemplars)

        for signal in target_exemplars:
            # First the similarity of the signal to every word category is calculated and a best fitting word category
            # is chosen accordingly
            index_max_sim, total_similarities = Perception(perceiver_lex, perceiver_com_words, perceiver_meta_com_words,
                                                           n_words, n_dimensions, n_exemplars, n_continuers,
                                                           anti_ambiguity_bias).similarity(signal)

            # Then the anti-ambiguity bias is added and the signal is stored (or not) depending on its probability of
            # being stored. This probability is based on how well the signal fits within the chosen word category
            # relative to the other word categories
            if (i % 2) == 0:
                if anti_ambiguity_bias:
                    lexicon2 = Perception(perceiver_lex, perceiver_com_words, perceiver_meta_com_words, n_words,
                                          n_dimensions, n_exemplars, n_continuers, anti_ambiguity_bias).\
                        add_anti_ambiguity_bias(index_max_sim, total_similarities, signal)

                # If there's no anti-ambiguity bias, the signal is stored within the best fitting word category
                # whatsoever
                else:
                    lexicon2[index_max_sim][0].insert(0, signal)
                # Only the first 100 exemplars of a word are used
                lexicon2[index_max_sim][0] = lexicon2[index_max_sim][0][:100]
                #print("Stored signal: ", signal)
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

                #print("Stored signal: ", signal)
            # print(lexicon)
        i += 1

    # Plot the end state of the words
    for word_index in range(n_words):
        exemplars = lexicon[word_index][0]
        plt.scatter(*zip(*exemplars))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

    # Plot the end state of the words
    for word_index in range(n_words):
        exemplars = lexicon2[word_index][0]
        plt.scatter(*zip(*exemplars))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()
