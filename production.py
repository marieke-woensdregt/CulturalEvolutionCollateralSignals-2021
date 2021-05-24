import random
import numpy as np
import math


class Production:

    def __init__(self, lexicon, com_words, meta_com_words, n_words, n_dimensions, n_exemplars, n_continuers,
                 similarity_bias_word, similarity_bias_segment, noise, activation_constant):
        """
        The initialisation of the production class.
        :param lexicon: list; a list of words, for which each word consists of a list of exemplars, which in turn is a
        list of the number of dimensions floats
        :param com_words: list; a list of the communicative words, for which each word consists of a list of exemplars,
        which in turn is a list of the number of dimensions floats
        :param meta_com_words: list; a list of the metacommunicative words, for which each word consists of a list of
        exemplars, which in turn is a list of the number of dimensions floats
        :param n_words: int; the number of words contained in the agent's lexicon
        :param n_dimensions: int; the number of dimensions of the exemplars
        :param n_exemplars: int; the number of exemplars per word
        :param n_continuers: int; the number of continuer (metacommunicative) words
        :param similarity_bias_word: boolean; whether the word similarity bias is present
        :param similarity_bias_segment: boolean; whether the segment similarity bias is present
        :param noise: boolean; whether noise is added to the signals
        :param activation_constant: float; the constant used to calculate the activation level
        """

        self.lexicon = lexicon
        self.com_words = com_words
        self.meta_com_words = meta_com_words
        self.n_words = n_words
        self.n_dimensions = n_dimensions
        self.n_exemplars = n_exemplars
        self.n_continuers = n_continuers
        self.similarity_bias_word = similarity_bias_word
        self.similarity_bias_segment = similarity_bias_segment
        self.noise = noise
        self.activation_constant = activation_constant

    def select_exemplar(self):
        """
        Select the exemplars for every word category based on their activation level.
        :return: list; a list containing the selected exemplars for every word category
                 list; a list containing the activations of the exemplars for every word category
        """

        # Select a target exemplar for every word category
        targets = []
        total_activations = []
        for word_index in range(self.n_words):

            # First calculate the activation of every exemplar and store in a list
            # print("Exemplars beginning: ", exemplars)
            activation_exemplars = []
            for j in range(len(self.lexicon[word_index][0])):
                j += 1
                activation = math.exp(-self.activation_constant*j)
                activation_exemplars.append(activation)

            # print("Activation exemplars: ", activation_exemplars)

            # Store all the activations for all exemplars and all words
            total_activations.append(activation_exemplars)

            # Calculate the sum of the activation of the exemplars of the word
            activation_word = sum(activation_exemplars)
            # print("Activation word: ", activation_word)

            # Calculate the probabilities of the exemplar being chosen to produce for that word category. The
            # probability is calculated by dividing the exemplar activation by the total activation of the word category
            exemplar_probs = [activation / activation_word for activation in activation_exemplars]
            # print("Exemplars probability: ", exemplar_probs)
            # print("Max probability: ", max(exemplar_probs))

            # Choose an exemplar to produce based on their probabilities
            target = random.choices(self.lexicon[word_index][0], weights=exemplar_probs, k=1)
            # print("Index chosen exemplar: ", self.lexicon[word_index][0].index(target[0]))
            # print("Chosen exemplar: ", target)

            # Store the exemplars for every word category to be produced
            targets.append(target[0])
            # print(targets)

        return targets, total_activations

    def add_biases(self, targets, total_activations, k=0.2):
        """
        Add the selected biases to the chosen exemplars.
        :param targets: list; the target exemplars to be produced.
        :param total_activations: list; a list containing the activations of the exemplars for every word category
        :param k: float; a constant for calculating the similarity
        :return: list; a list of the target exemplars with the selected biases added
        """

        # Add the biases to every target (one target per word) before production
        word_index = 0
        target_exemplars = []
        for target in targets:

            exemplars = self.lexicon[word_index][0]

            # print("Before similarity biases: ", target)
            # If both similarity biases are added to the target, they are combined
            if self.similarity_bias_word and self.similarity_bias_segment:
                word_bias = self.similarity_word(target, exemplars, total_activations[word_index], k)
                segment_bias = self.similarity_segment(target, total_activations, k)

                # print("Word bias: ", word_bias)
                # print("Segment bias: ", segment_bias)

                # The ratio of the word similarity to the segment similarity is 0.9:1 if we deal with communicative
                # words
                if self.lexicon[word_index][1] == "C":
                    total_bias = [(0.9 * bias_word) + bias_segment for bias_word, bias_segment in zip(word_bias,
                                                                                                      segment_bias)]
                    target = [bias / 1.9 for bias in total_bias]

                # The ratio of the word similarity to the segment similarity is 1:0.5 for metacommunicative words
                # (continuers)
                if self.lexicon[word_index][1] == "M":
                    total_bias = [bias_word + (0.0 * bias_segment) for bias_word, bias_segment in
                                  zip(word_bias, segment_bias)]
                    target = [bias / 1.0 for bias in total_bias]


                # print("After similarity biases: ", target)

            # If only one of the similarity biases is added
            elif self.similarity_bias_word:
                target = self.similarity_word(target, exemplars, total_activations[word_index], k)

            elif self.similarity_bias_segment:
                target = self.similarity_segment(target, total_activations, k)

            # If noise is added, it should be added to the target exemplar after the other biases have been added if
            # applicable
            if self.noise:
                # print("Before noise bias: ", target)
                if self.lexicon[word_index][1] == "C":
                    target = self.add_noise(target)
                # print("After noise bias: ", target)

                # The bias is weakened through a lower G value when the exemplar is from a continuer word
                else:
                    target = self.add_noise(target, G=1000)

            # Store all the targets after the biases have been added
            target_exemplars.append(target)
            word_index += 1

        # print("Targets before biases:", targets)
        # print("Targets after biases:", target_exemplars)

        return target_exemplars

    def similarity_word(self, target, exemplars, activation_exemplars, k):
        """
        Calculate the word similarity bias.
        :param target: list; the target to which the bias is added, consisting of a number of dimensions
        :param exemplars: list; the exemplars of the word category to which the target belongs
        :param activation_exemplars: list; the activations of the exemplars of the word category to which the target
        belongs
        :param k: float; a constant used to calculate the similarity
        :return: list; the target exemplar with the added word similarity bias
        """

        # For every segment (dimension) the similarity is calculated and the similarity bias is added
        bias_exemplar = []
        for dimension in range(self.n_dimensions):
            index = 0
            sum = 0
            sum2 = 0

            # This is done for every exemplar in the word (see the equation)
            for exemplar in exemplars:
                sum += exemplar[dimension] * activation_exemplars[index] * math.exp(
                    (-k) * abs(target[dimension] - exemplar[dimension]))
                sum2 += activation_exemplars[index] * math.exp((-k) * abs(target[dimension] - exemplar[dimension]))
                index += 1

            # Finally the target with the similarity bias is stored for the two dimensions
            bias_exemplar.append(sum / sum2)

        # print("Target with word similarity bias: ", bias_exemplar)

        return bias_exemplar

    def similarity_segment(self, target, total_activations, k):
        """
        Calculate the segment similarity bias.
        :param target: list; the target to which the bias is added, consisting of a number of dimensions
        :param total_activations: list; a list containing the activations of the exemplars for every word category
        :param k: float; a constant used to calculate the similarity
        :return: list; the target exemplar with the added word similarity bias
        """

        # For every segment (dimension) of all exemplars (of all words) the similarity is calculated and a bias is added
        # to the target
        bias_exemplar = []

        # Every segment (dimension) is compared for its similarity
        for dimension in range(self.n_dimensions):
            word_index = 0
            sum = 0
            sum2 = 0

            # This is done for all the words in the lexicon
            for word in self.lexicon:

                # The exemplar index
                index = 0

                # Every exemplar is compared to the target exemplar
                for exemplar in word[0]:

                    # According to the equation, the similarity of the target to every exemplar is calculated and added
                    # as a bias
                    sum += exemplar[dimension] * total_activations[word_index][index] * math.exp(
                        (-k) * abs(target[dimension] - exemplar[dimension]))
                    sum2 += total_activations[word_index][index] * math.exp(
                        (-k) * abs(target[dimension] - exemplar[dimension]))
                    index += 1
                word_index += 1

            # The target including the word similarity bias is stored for both dimensions
            bias_exemplar.append(sum / sum2)

        # print("Target with segment similarity bias added: ", bias_exemplar)

        return bias_exemplar

    def add_noise(self, target, G=5000):
        """
        Add noise to the target exemplar.
        :param target: list; the target to which the bias is added, consisting of a number of dimensions
        :param G: int; a constant determining the strength of the noise bias
        :return: list; the target exemplar with the added noise
        """

        target_noise = []

        # For every segment in the target exemplar noise is added
        for segment in target:
            N = 100

            # First the bias is calculated
            bias = ((segment - (N / 2)) ** 2) / G

            # The bias is substracted or added based on the target segment value
            if segment > (N / 2):
                new_target = segment - bias
            else:
                new_target = segment + bias

            # A new target is sampled from a normal distribution with sd 3 and the mean of the target segment with the
            # added bias
            added_noise = np.random.normal(new_target, 3, 1)
            # print("New target with noise: ", new_target)

            # The segments with their added noise are stored in a list (as we have multiple dimensions)
            target_noise.append(added_noise)
        # print("Target before noise: ", target)
        # print("Target after noise added: ", target_noise)

        return target_noise
