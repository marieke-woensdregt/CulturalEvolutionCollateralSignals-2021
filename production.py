import random
import numpy as np
import math


class Production:

    def __init__(self, lexicon, com_words, meta_com_words, n_words, n_dimensions, n_exemplars=100, n_continuers=0,
                 similarity_bias_word=True, similarity_bias_segment=True, noise=True):

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

    def select_exemplar(self):

        # Select a target exemplar for every word category
        targets = []
        total_activations = []
        for word_index in range(self.n_words):

            # Only store exemplars until position 100
            exemplars = self.lexicon[word_index][0][:101]

            # First calculate the activation of every exemplar
            print("EXEMPLARS BEGINNING: ", exemplars)
            activation_exemplars = []
            j = 1
            for exemplar in exemplars:
                # activation = math.exp(0.2*j)
                activation = 1 / (0.2 * j)
                activation_exemplars.append(activation)
                j += 1

            # Store all the activations for all exemplars and all words
            total_activations.append(activation_exemplars)
            # print("Activation exemplars: ", activation_exemplars)

            # Calculate the sum of the activation of the exemplars of the word
            activation_word = sum(activation_exemplars)
            # print("Activation word: ", activation_word)

            # Calculate the probabilities of the exemplar being chosen to produce for that word category
            exemplar_probs = [activation / activation_word for activation in activation_exemplars]
            # print("Exemplars probability: ", exemplar_probs)
            # max_prob = max(exemplar_probs)
            # print("Max probability: ", max_prob)

            # Choose an exemplar to produce based on their probabilities
            target = random.choices(exemplars, weights=exemplar_probs, k=1)
            print("CHOSEN EXEMPLARS: ", target)

            # Store the exemplars for every word to be produced
            targets.append(target[0])
            # print(targets)

        return targets, activation_exemplars, total_activations

    def add_biases(self, targets, activation_exemplars, total_activations, k=0.2):

        # Add the biases to every target (one target per word) before production
        word_index = 0
        target_exemplars = []
        for target in targets:

            # Select only the first 100 exemplars per word
            exemplars = self.lexicon[word_index][0][:101]

            # If both similarity biases are added to the target, they are combined
            if self.similarity_bias_word and self.similarity_bias_segment:
                word_bias = self.similarity_word(target, exemplars, activation_exemplars, k)
                segment_bias = self.similarity_segment(target, total_activations, k)

                # The ratio of the word similarity to the segment similarity is 9/10
                total_bias = [(9 * a) + b for a, b in zip(word_bias, segment_bias)]
                target = [bias / 10 for bias in total_bias]

            elif self.similarity_bias_word:
                target = self.similarity_word(target, exemplars, activation_exemplars, k)

            elif self.similarity_bias_segment:
                target = self.similarity_segment(target, total_activations, k)

            # If noise is added, it should be added to the target exemplar after the other biases have been added
            if self.noise:
                target = self.add_noise(target)

            # Store all the targets after the biases have been added
            target_exemplars.append(target)
            word_index += 1

        print("BEFORE BIASES:", targets)
        print("WITH BIASES:", target_exemplars)

        return target_exemplars

    def similarity_word(self, target, exemplars, activation_exemplars, k):

        # For every segment (dimension) the similarity is calculated and the similarity bias is added
        bias_exemplar = []
        for dimension in range(self.n_dimensions):
            index = 0
            sum = 0
            sum2 = 0

            # This is done for every exemplar in the word (see the equation)
            for exemplar in exemplars:
                sum += exemplar[dimension] * activation_exemplars[index] * math.exp(
                    -k * abs(target[dimension] - exemplar[dimension]))
                sum2 += activation_exemplars[index] * math.exp(-k * abs(target[dimension] - exemplar[dimension]))
                index += 1

            # Finally the target with the similarity bias is stored for the two dimensions
            bias_exemplar.append(sum / sum2)

        print("SIMILARITY WORD ADDED: ", bias_exemplar)

        return bias_exemplar

    def similarity_segment(self, target, total_activations, k):

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
                index = 0
                # print("word_index:", word_index)

                # Every exemplar is compared to the target exemplar
                for exemplar in word[0]:
                    # print("Ex dimension: ", exemplar[dimension])
                    # print("activations: ", total_activations[word_index][index])
                    # print("Math: ", math.exp(-k * abs(target[dimension]-exemplar[dimension])))

                    # According to the equation, the similarity of the target to every exemplar is calculated and added
                    # as a bias
                    sum += exemplar[dimension] * total_activations[word_index][index] * math.exp(
                        -k * abs(target[dimension] - exemplar[dimension]))
                    # print("sum: ", sum)
                    sum2 += total_activations[word_index][index] * math.exp(
                        -k * abs(target[dimension] - exemplar[dimension]))
                    index += 1
                word_index += 1

            # The target including the word similarity bias is stored for both dimensions
            bias_exemplar.append(sum / sum2)

        print("SIMILARITY SEGMENT ADDED: ", bias_exemplar)

        return bias_exemplar

    def add_noise(self, target, G=5000):

        target_noise = []

        # For every segment in the target exemplar noise is added
        for segment in target:
            N = 100

            # First the bias is calculated
            bias = ((segment - (N / 2)) ** 2) / G

            # A new target is sampled from a normal distribution with sd 3 and the mean of the target segment
            new_target = np.random.normal(segment, 3, 1)

            # The biases is substracted or added based on the target segment value
            if segment > N / 2:
                added_noise = new_target[0] - bias
            else:
                added_noise = new_target[0] + bias

            # The noise of the segments is stored (as we have multiple dimensions)
            target_noise.append(added_noise)
        print("NOISE ADDED: ", target_noise)

        return target_noise
