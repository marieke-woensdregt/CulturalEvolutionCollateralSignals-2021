import random
import math


class Perception:

    def __init__(self, lexicon, v_words, continuer_words, n_words, n_dimensions, n_exemplars, n_continuers,
                 anti_ambiguity_bias, activation_constant):
        """
        The initialisation of the perception class.
        :param lexicon: list; a list of v_words, for which each word consists of a list of exemplars, which in turn is a
        list of the number of dimensions floats
        :param v_words: list; a list of the regular vocabulary v_words, for which each word consists of a list of
        exemplars, which in turn is a list of the number of dimensions floats
        :param continuer_words: list; a list of the continuer v_words, for which each word consists of a list of
        exemplars, which in turn is a list of the number of dimensions floats
        :param n_words: int; the number of v_words contained in the agent's lexicon
        :param n_dimensions: int; the number of dimensions of the exemplars
        :param n_exemplars: int; the number of exemplars per word
        :param n_continuers: int; the number of continuer v_words
        :param anti_ambiguity_bias: boolean; whether an anti-ambiguity bias is present
        :param activation_constant: float; the constant used to calculate the activation level
        """

        self.lexicon = lexicon
        self.v_words = v_words
        self.continuer_words = continuer_words
        self.n_words = n_words
        self.n_dimensions = n_dimensions
        self.n_exemplars = n_exemplars
        self.n_continuers = n_continuers
        self.anti_ambiguity_bias = anti_ambiguity_bias
        self.activation_constant = activation_constant

    def similarity(self, signal, k=0.2):
        """
        Calculate the similarity between a given signal and the word categories.
        :param signal: list; the signal produced by the producer, consisting of the number of dimensions floats
        :param k: float; a constant to calculate the similarity
        :return: int; the index of the word category which was the best fit for the signal
                 list; a list of the similarities of the signal to each word category
        """

        # Calculate the activations first based on their index (per word category)
        total_activations = []
        for word_index in range(self.n_words):

            # print("EXEMPLARS BEGINNING: ", exemplars)
            activation_exemplars = []
            for j in range(len(self.lexicon[word_index][0])):
                j += 1
                activation = math.exp(-self.activation_constant*j)
                activation_exemplars.append(activation)
            total_activations.append(activation_exemplars)

        # Calculate the similarity of the signal to all the exemplars of all the v_words
        similarities = []

        # Iterate over all the v_words
        for word_index in range(self.n_words):
            # Iterate over all the dimensions (to access the segments)
            sum_similarity = 0
            for dimension in range(self.n_dimensions):
                index = 0
                sum = 0
                # Calculate the similarity of a segment of the signal compared to all the exemplars in a word
                for exemplar in self.lexicon[word_index][0]:
                    sum += total_activations[word_index][index] * math.exp((-k) * abs(signal[dimension] -
                                                                                      exemplar[dimension]))
                    index += 1
                # Take the sum of the similarities of all the dimensions
                sum_similarity += sum

            # Store the similarities for every exemplar of every word
            similarities.append(sum_similarity)
        # print("Similarities: ", similarities)

        # Get the word with the highest similarity to the signal
        max_similarity = max(similarities)
        index_max_sim = similarities.index(max_similarity)

        # print("Signal: ", signal)
        # print("Similarities: ", similarities)
        # print("Signal most similar to word: ", index_max_sim)

        return index_max_sim, similarities

    def add_anti_ambiguity_bias(self, index_max_sim, total_similarities, signal):
        """
        Adds an anti-ambiguity bias to the perceived signal; whether to store the signal or not.
        :param index_max_sim: int; the index of the word category which was the best fit for the signal
        :param total_similarities: list; a list of the similarities of the signal to each word category
        :param signal: list; the signal produced by the producer, consisting of the number of dimensions floats
        :return: list; a list of v_words, for which each word consists of a list of exemplars, which in turn is a
        list of the number of dimensions floats
        """

        # The probability whether the signal is stored is calculated by its similarity to the word category divided by
        # the sum of its similarity to all word categories

        probability_storage = (total_similarities[index_max_sim]) / sum(total_similarities)
        # print("Probability of being stored: ", probability_storage)

        # Determine whether the signal is stored based on the probability calculated
        store = random.choices([True, False], weights=[probability_storage, 1 - probability_storage], k=1)

        # If the signal is stored, the signal is stored in the lexicon as the first segment of the previously determined
        # word category
        if store[0]:
            self.lexicon[index_max_sim][0].insert(0, signal)
            # print("The following signal is stored: ", signal)

        # print(self.lexicon[index_max_sim])

        return self.lexicon, store[0], probability_storage
