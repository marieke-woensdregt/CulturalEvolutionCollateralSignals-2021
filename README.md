# CulturalEvolutionCollateralSignals-2021

The implementation of the model consists of three files:
* *agent.py*: the implementation of the agents getting initialised with a lexicon.
* *production.py*: the implementation of the production.
* *perception.py*: the implementation of the perception.
* *perception2.py*: the most recent version of perception, in which we follow Wedel's implementation more closely. Specifically in how we calculate the similarity. 

The remaining files are used for running the simulation or analysis:
* *simulation.py*: to run n_runs simulations of n_rounds rounds. A round consists of one agent producing n_words signals and another agent perceiving the produced signals. 
* *analysis.py*: to analyse the results from the simulation. Produces several plots.

## How to run simulations
The easiest way to run a simulation is to run it from your command line. 
1. Go to the folder in which the implemention of the model is stored.
2. Run the following line: *python3 -c 'import simulation; simulation.simulation_runs(20, 10000, 5, 2, anti_ambiguity_bias=True, n_continuers=1, wedel_start=True)'*. The first four arguments are mandatory to specify and stand for the number of simulation runs, the number of rounds, the number of words (regular vocabulary words and continuer words summed) and the number of dimensions. Some other arguments can be specified when you do not want to run the default. These arguments include: 
    1. *seed=None*: if you want to specify a seed for initialising the agent's lexicon
    2. *n_exemplars=100*: set the number of exemplars per word category in the lexicon
    3. *n_continuers=0*: set the number of continuer words in the lexicon
    4. *similarity_bias_word=True*: whether to include the word similarity bias
    5. *similarity_bias_segment=True*: whether to include the segment similarity bias
    6. *noise=True*: whether to include the noise bias
    7. *anti_ambiguity_bias*=True: whether to include the anti-ambiguity bias
    8. *continuer_G=2500*: set the value of the noise bias constant G for continuer words
    9. *word_ratio=1.8*: set the ratio of the word similiarty bias for the continuer words (word:segment 1.8:0.1)
    10. *wedel_start=False*: whether to use the initialisation used in Wedel's (2012) paper
3. The simulation should start and show the current number of rounds
4. When the simulation has ended, the results file will be stored in the current directory. The filename is specified as follows: 
    *filename = "simulation_results/results_" + str(n_runs) + "_" + str(n_rounds) + "_" + str(anti_ambiguity_bias) + "_" + \
               str(n_continuers) + "_" + str(continuer_G) + "_" + str(word_similarity_weight) + "_" + str(segment_similarity_weight) + "_" + str(n_words) + "_" + str(wedel_start) + ".p"*
    
### Using Ponyland
When you want to make use of the computer cluster Ponyland to run simulations, there are a few things to take into account. First of all, I'd advise to make use of **GNU screens**. See the following link for instructions on how to use these: https://ponyland.science.ru.nl/doku.php?id=wiki:ponyland:gnu_screen. I used them so the simulations kept running after the connection to the server was broken due to inactivity. When running simulations in parallel, you can create multiple screens, as to run one simulation per screen.  

## How to analyse the results
Finally, when the simulations are done, the results can be analysed with the help of the file analysis.py. This will generate 3 or 4 different plots (4 when continuer words were present). The following plots are generated and saved:
1. *The exemplars of the word categories per simulation run*: when running multiple simulation runs of the same simulation, the exemplars of every word category are plotted per simulation run, resulting in a grid of n_runs plots. 
2. *The average centroid distance*: a histogram showing the average distance between the centroids of every word category per simulation run. 
3. *The average two dimensional SD*: the average distance between all the exemplars and the centroids of their corresponding categories per simulation run.
4. *The exemplars of all simulation runs plotted per word category*: this only applies when there is at least one continuer word. All exemplars for all simulation runs are plotted per word category. This now only plots 4 regular vocabulary words and 1 continuer word, this can be adjusted if needed.

Moreover, the following measures are calculated and printed:
1. *Average exclusion rate*: averaged over all simulation runs, how often signals are excluded (not stored in the lexicon) due to the anti-ambiguity bias.
2. *Average probability storage*:  averaged over all simulation runs, how often signals are stored in the lexicon.
3. *Regular vocabulary words average distance to middle*: the average distance of the regular vocabulary words to the middle of the space per run.
4. *Continuer average distance to middle*: the averagde distance of the continuer words to the middle of the space per run.
5. *Regular vocabulary word average distance over all runs*: the average distance of the start and end states of the centroids for the regular vocabulary words over all runs.
6. *Continuer average distance over all runs*: the average distance of the start and end states of the centroids for the continuer words over all runs.
6. *Regular vocabulary word average distance per run*: the average distance (averaged over regular vocabulary words) between centroids of the initialisation versus the end for every simulation run.
7. *Continuer average distance per run*: the average distance between the continuer centroids of the initialisation versus the end for every simulation run.
8. *Order of 500 rounds of most squareness*: a list containing for every simulation run which number of rounds got the most square configuration. The index represents an order of 500 rounds (e.g., an index of 19 means 10,000 rounds: 500*(19+1)).


In order to start the analysis, the following command can be run from the command line:
*python3 -c 'import analysis; analysis.analysis("simulation_results/", "plots/", "results_20_500_True_1_1250_0.9_0.0_10_False.p", intermediate=None, wedel_start=False)'*

The analysis function needs the following (mandatory) arguments:
1. *Folder*: the folder in which your results are saved and the analysis plots will be stored. 
2. *Results_file*: the name of the results file in the previously specified folder. 
3. *Intermediate=None*: an optional argument specifying whether you want to analyse an intermediate result (for instance at 4,000 rounds instead of the end number of rounds) 
4. *Wedel_start=True*: an optional argument specifying whether Wedel's (2012) initialisation was used. 


