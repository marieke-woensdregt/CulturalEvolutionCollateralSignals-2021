import random
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, n_words, n_dimensions, seed, n_exemplars=100, n_continuers=0):

        self.n_words = n_words
        self.n_dimensions = n_dimensions
        self.seed = seed
        self.n_exemplars = n_exemplars
        self.n_continuers = n_continuers

        # # Generate a lexicon as part of the initialisation
        # self.lexicon, self.com_words, self.meta_com_words = self.generate_lexicon()

    # Initialising lexicon
    def generate_lexicon(self):

        if self.seed:
            random.seed(self.seed)
        # Create a lexicon consisting of n_words words each in turn consisting of n_exemplars exemplars
        lexicon = []

        # The starting condition of the words used in the paper
        means = [[20,80], [40,40], [60,60], [80,20]]
        for w in range(self.n_words):
            word = []

            if self.seed:
                random.seed(self.seed+w)
            # Define the mean and the covariance to sample from a multivariate normal distribution to create clustered
            # exemplars for the words
            #mean = [random.randrange(10, 91) for i in range(self.n_dimensions)]

            # Instead, we're using the starting condition used in the paper
            mean = means[w]
            cov = [[10, 0], [0, 10]]
            x, y = np.random.multivariate_normal(mean, cov, self.n_exemplars).T
            word.append(list(map(lambda x, y: [x, y], x, y)))

            # Plot every word
            plt.scatter(x, y)

            # Initialiase all words as 'communicative words' ('C')
            lexicon.append([word[0], "C"])

        # print(lexicon)

        # Some plot settings
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

        # Split the lexicon into meta communicative words (continuers) and communicative words
        if self.n_continuers:
            if self.n_continuers > self.n_words:
                raise ValueError("The number of continuers must be lower than the number of words.")

            # The continuers are randomly chosen out of the lexicon
            indices_meta = random.sample(range(self.n_words), k=self.n_continuers)
            meta_com_words = []
            for index in indices_meta:
                lexicon[index][1] = "M"
                # Create a separate lexicon with the meta communicative words
                meta_com_words.append(lexicon[index])

            # The words that are not meta communicative words are communicative words
            com_words = [word for word in lexicon if word not in meta_com_words]

            # print("The word categories are split into communicative and metacommunicative words")
            # print("New lexicon:", lexicon)

            # print("Meta:", meta_com_words)
            # print("Com:", com_words)

        # If there are no continuers, the meta communicative words list is empty and all the words in the lexicon are
        # communicative words
        else:
            com_words = lexicon
            meta_com_words = []

        return lexicon, com_words, meta_com_words, indices_meta
