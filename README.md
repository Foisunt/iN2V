# iN2V
iN2V: Bringing Transductive Node Embeddings to Inductive Graphs

published at ICML 2025, where you can see the poster and a short presentation: https://icml.cc/virtual/2025/poster/46107

Find the paper in
PMLR: https://proceedings.mlr.press/v267/lell25a.html
OpenReview: https://openreview.net/forum?id=BYakLzKJDz
or arXiv: https://arxiv.org/abs/2506.05039


We ran our experiments with 
Python 3.11.10
Cuda 12.4
torch 2.2.2
torch_geometric 2.5.3

For detailed requirements see the requirements.txt

The directories results and results_comb are empty right now, but when you run experiments results will be stored there.
The directory chkp is empty right now, but when you save embeddings (iN2V, FP, ...) they will be stored there and loaded for any GNN you train with them.
The directory Code contains the Code and datasets

The directory Expermients contains yaml files which define our experiments.
Each file contains a list for each hyperparameter, and the code will run all combinations (grid search) of the given hyperparameters in the file.
When the runs finished it will write "alldone:true" in the file that experiments are not run multiple times (if you want to rerun a specific experiment, delete this line from the settings).

The files starting with (datasetabbreviation)_(split) (where split is train|test, -> 10% train with 45% test ist 145) are the hyperparameter search for N2V.
Some names have changed in the paper vs code so here is a mapping with codename:papername
alpha:\lambda, embedding_dim:d, lr:learning rate, prob_replace:r, w_ms:\alpha, w_ndiv:\beta, fp:iter (of Feature Propagation)
Note that alpha, delay, fp do not modify the training loss -> all combinations of them can be searched together -> they are written in a list of list as the code unpacks the first-level lists to run all combinations
For example Cora_62.yml search the iN2V hyperparameters with the loss-based modifications.

The files starting with E_(datasetabbreviation)_(split) contain the best hyperparameter to calulate and save the extended iN2V/FP/... embeddings.

The files starting with F_(datasetabbreviation)_(split) contain the hyperparameter search for the GNNs. 
Some more names: drop_model:dropout rate, wd: weight decay
model_type specifies whether the model is lin (linear, no jumpin knowledge), or has jumping knowledge connections (jk: only layer outputs, jkb: including connection from input)
model is a concatenation of model_numlayers_hiddensize
datamode specifies which features to use: gra is the original graph features. otherwise, starting with emb_ and cat_ specifies wheter to use just the trained embeddings or concatenate them with the original graph features. After that comes 
fpbase:Feature Prop, fploss:Feature Prop+loss based N2V modification, fpprob: Feature Prop+sampling based N2V modification, transd: transductive N2V features, tr: (train only) inductive N2V, ba: (baseline) frozen N2V lamda=1, po: post-hoc, lo:post-hoc w losses, re: post-hoc w sampling

The files starting with R1_ or R2_ are experiment runs for the ablation plots.




To run an experiment, activate your venv with the installed requirements, navigate into the Code directory then run

python main.py filename_in_experiments.yml

which will run all experiments specified in the file filename_in_experiments.yml from the Experiments directory using 1 process on cuda:0 gpu
If you want to change the number of processes or where they run, you can say

python main.py 2 cpu filename_in_experiments.yml 
which runs 2 process on cpu

or 
python main.py 3 0.1.1 filename_in_experiments.yml 
which will run 3 processes on one on cuda:0 and two on cuda:1




Steps needed to reproduce results:

- Create dataset splits ("create_split" method in "create_dataset_splits.ipynb")

- (optional) search hyperparameters for the N2V embeddings
e.g. python main.py 2 0.0 Cora_145.yml

- (optional) extract the best hyperparameters from the previous search and write the according E_... file, e.g. using the Code in the second half of Tables_Experiments

- save the N2V embeddings with best hyperparamters (run E_ ... experiments), 

- (optional) creat F_ files to run final GNN experiments (Code in Tables_Experiments)

- train GNNs on the trained embeddings (by running the F_... experiments)

- print results, e.g. using the different print table functions in Tables_Experiments

