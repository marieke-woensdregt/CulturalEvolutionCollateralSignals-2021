import random
import numpy as np


class Agent:

    def __init__(self, n_words, n_dimensions, seed, n_exemplars=100, n_continuers=0):
        """
        Initialisation of the agent class.
        :param n_words: int; the number of word categories contained in the lexicons of the agents
        :param n_dimensions: int; the number of dimensions of the exemplars
        :param seed: int; the seed used to produce the means for sampling the word categories
        :param n_exemplars: int; the number of exemplars per word category
        :param n_continuers: int; the number of continuer v_words in the lexicon
        """

        self.n_words = n_words
        self.n_dimensions = n_dimensions
        self.seed = seed
        self.n_exemplars = n_exemplars
        self.n_continuers = n_continuers

    def generate_lexicon(self):
        """
        Generate a lexicon containing v_words, which in turn contains n_exemplar exemplars.
        :return: list; a list of v_words, for which each word consists of a list of exemplars, which in turn is a
                       list of the number of dimensions floats
                 list; a list containing the communicative v_words
                 list; a list containing the metacommunicative v_words (continuers)
                 list; a list containing the indices of the metacommunicative v_words in the lexicon
        """

        # If a seed was provided, set the seed
        if self.seed:
            random.seed(self.seed)

        # Create a lexicon consisting of n_words v_words each in turn consisting of n_exemplars exemplars
        lexicon = []

        # The means of the starting condition of the v_words used in the paper
        means = [[20, 80], [40, 40], [60, 60], [80, 20]]

        for w in range(self.n_words):
            word = []

            # We're using the starting condition used in the paper
            mean = means[w]

            cov = [[10, 0], [0, 10]]
            x, y = np.random.multivariate_normal(mean, cov, self.n_exemplars).T
            word.append(list(map(lambda x, y: [x, y], x, y)))

            # Initialise all v_words as regular vocabulary v_words ('V')
            lexicon.append([word[0], "V"])

        # Split the lexicon into meta communicative v_words (continuers) and communicative v_words if applicable
        indices_continuer = False
        if self.n_continuers:

            # If the number of continuer v_words is bigger than the number of v_words in the lexicon raise an error message
            if self.n_continuers > self.n_words:
                raise ValueError("The number of continuers must be lower than the number of v_words.")

            # The continuers are randomly chosen out of the lexicon
            indices_continuer = random.sample(range(self.n_words), k=self.n_continuers)

            continuer_words = []
            for index in indices_continuer:
                lexicon[index][1] = "C"

                # Create a separate lexicon with the meta communicative v_words
                continuer_words.append(lexicon[index])

            # The v_words that are not meta communicative v_words are communicative v_words
            v_words = [word for word in lexicon if word not in continuer_words]

            # print("Lexicon:", lexicon)

            # print("Meta lexicon:", continuer_words)
            # print("Com lexicon:", v_words)

        # If there are no continuers, the meta communicative v_words list is empty and all the v_words in the lexicon are
        # communicative v_words
        else:
            v_words = lexicon
            continuer_words = []

        return lexicon, v_words, continuer_words, indices_continuer
