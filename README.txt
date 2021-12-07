#!/usr/bin/env python3

util functions:
1) utils_lda.py
2) cv_clf.py
3) lda_rf_sgd.py
4) args.py


1. Change the parameters in run_rbf.sh and run_rft.sh, and use the command line
conda activate t1.7
tmux new -s rbf
conda activate t1.7
sh run_rbf.sh  # sh run_rff.sh


2. lda_rf_summary.py
3. result_assemble.py


a. RBF kernel gamma=1.0 # feature tranformation
b. RandomFourier kernel gamma='auto' # gamma=1/X.shape[1].
c. RandomKernel

