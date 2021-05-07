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
        word_index = 0

        # Iterate over all the words
        for word_index in range(self.n_words):
            exemplar_sim = []

            # Iterate over all the dimensions (to access the segments)
            for dimension in range(self.n_dimensions):
                index = 0
                sum = 0
                sum2 = 0

                # Calculate the similarity of a segment of the signal compared to all the exemplars in a word
                for exemplar in self.lexicon[word_index][0]:
                    sum += exemplar[dimension] * total_activations[word_index][index] * math.exp(
                        (-k) * abs(signal[dimension] - exemplar[dimension]))
                    sum2 += total_activations[word_index][index] * math.exp(
                        (-k) * abs(signal[dimension] - exemplar[dimension]))
                    index += 1
                exemplar_sim.append(sum)

            word_index += 1

            # Store the similarities for every exemplar of every word
            similarities.append(exemplar_sim)

            # Take the sum of the similarities within a word category
            total_similarities = []
            for word_cat in similarities:
                sum = np.sum(word_cat)
                total_similarities.append(sum)
        # print(similarities)
        # print(total_similarities)

        # Get the word with the highest similarity to the signal
        max_similarity = max(total_similarities)
        index_max_sim = total_similarities.index(max_similarity)
        # print("SIGNAL MOST SIMILAR TO WORD: ", index_max_sim)

        return index_max_sim, total_similarities

    def add_anti_ambiguity_bias(self, index_max_sim, total_similarities, signal):

        # The probability whether the signal is stored is calculated by its similarity to the word category divided to
        # the sum of its similarity to all word categories

        # print(1/total_similarities[index_max_sim])
        # print(sum(total_similarities))
        probability_storage = (total_similarities[index_max_sim]) / sum(total_similarities)
        # print(probability_storage)

        # Determine whether the signal is stored based on the probability calculated
        store = random.choices([True, False], weights=[probability_storage, 1 - probability_storage], k=1)

        # If the signal is stored, the signal is stored in the lexicon as the first segment of the previously determined
        # word category
        if store[0]:
            self.lexicon[index_max_sim][0].insert(0, signal)

        # print(self.lexicon[index_max_sim])

        return self.lexicon
