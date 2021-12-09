#!/bin/bash

transformation='rbf'
actfun='None'

doc_list=("data/CNS" "data/Colon" "data/Leukemia")
# doc_list=("data/Accent" "data/Audit" "data/ARCX" "data/ARCZ" "data/LSVT" "data/PGD" "data/Colon" "data/Leukemia" "data/CNS")
# for p in 200 2000 20000 200000
for doc_root in ${doc_list[@]}; do
    echo "python lda_rf_sgd.py --doc_root $doc_root begin"
    python lda_rf_sgd.py --doc_root $doc_root --transformation $transformation --actfun $actfun
    echo "python lda_rf_sgd.py --doc_root $doc_root finish"
done

