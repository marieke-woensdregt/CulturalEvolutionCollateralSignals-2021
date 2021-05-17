import random
import numpy as np
import math


class Perception:

    def __init__(self, lexicon, com_words, meta_com_words, n_words, n_dimensions, n_exemplars, n_continuers,
                 anti_ambiguity_bias):

        self.lexicon = lexicon
        self.com_words = com_words
        self.meta_com_words = meta_com_words
        self.n_words = n_words
        self.n_dimensions = n_dimensions
        self.n_exemplars = n_exemplars
        self.n_continuers = n_continuers
        self.anti_ambiguity_bias = anti_ambiguity_bias

    def similarity(self, signal, k=0.2):

        # Calculate the activations first
        targets = []
        total_activations = []
        for word_index in range(self.n_words):

            # print("EXEMPLARS BEGINNING: ", exemplars)
            activation_exemplars = []
            for j in range(len(self.lexicon[word_index][0])):
                j += 1
                activation = math.exp(-0.02*j)
                # activation = 1 / (0.2 * j)
                activation_exemplars.append(activation)

            total_activations.append(activation_exemplars)

        # Calculate the similarity of the signal to all the exemplars of all the words
        similarities = []

        # Iterate over all the words
        for word_index in range(self.n_words):
            # exemplar_sim = []

            # Iterate over all the dimensions (to access the segments)

            sum_similarity = 0
            for dimension in range(self.n_dimensions):
                index = 0
                sum = 0
                # sum2 = 0

                # Calculate the similarity of a segment of the signal compared to all the exemplars in a word
                for exemplar in self.lexicon[word_index][0]:
                    # print("Exemplar dimension: ", exemplar[dimension])
                    # sum += exemplar[dimension] * total_activations[word_index][index] * math.exp((-k) * abs(signal[dimension] - exemplar[dimension]))
                    sum += total_activations[word_index][index] * math.exp((-k) * abs(signal[dimension] - exemplar[dimension]))
                    # sum2 += total_activations[word_index][index] * math.exp((-k) * abs(signal[dimension] - exemplar[dimension]))
                    index += 1

                # Take the sum of the distances of all the dimensions
                sum_similarity += sum

            # Store the distances for every exemplar of every word
            similarities.append(sum_similarity)
        # print("Similarities: ", similarities)

        # Get the word with the highest similarity to the signal, so the lowest distance
        # similarities = [1/distance for distance in similarities]
        max_similarity = max(similarities)
        index_max_sim = similarities.index(max_similarity)

        # print("SIGNAL: ", signal)
        # print("SIMILARITIES: ", similarities)
        # print("SIGNAL MOST SIMILAR TO WORD: ", index_max_sim)

        return index_max_sim, similarities

    def add_anti_ambiguity_bias(self, index_max_sim, total_similarities, signal):

        # The probability whether the signal is stored is calculated by its similarity to the word category divided to
        # the sum of its similarity to all word categories

        # print(1/total_similarities[index_max_sim])
        # print("Similarity best fit: ", total_similarities[index_max_sim])
        # print("Total similarity: ", sum(total_similarities))
        # probability_storage = ((1/total_similarities[index_max_sim])) / sum(total_similarities)
        probability_storage = (total_similarities[index_max_sim]) / sum(total_similarities)
        # print("Probability of being stored: ", probability_storage)

        # Determine whether the signal is stored based on the probability calculated
        store = random.choices([True, False], weights=[probability_storage, 1 - probability_storage], k=1)
        # store = random.choices([False, True], weights=[probability_storage, 1 - probability_storage], k=1)

        # If the signal is stored, the signal is stored in the lexicon as the first segment of the previously determined
        # word category
        if store[0]:
            self.lexicon[index_max_sim][0].insert(0, signal)

        # print(self.lexicon[index_max_sim])

        return self.lexicon
