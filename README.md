# hydrocephalus
Automated ventricular system segmentation in paediatric patients treated for hydrocephalus.


## Repository

This repository contains:
* data: sample data with CT scans
* splits: split generated for our experiments and sample split for cross validation
* src: hydrocephalus library which we used to run experiments
* notebooks: Jupyter notebooks containing workflow


## Steps to reproduce

1. Run `Split.ipynb` to create splits of training dataset for Cross-Validation.
2. Run `Train.ipynb` for each split and then with `CV_SPLIT_NUM` > `N_FOLDS` i.e. 11 to train on the whole dataset. Before evaluating you'll need to copy models from `result` to `models` directory.
3. Run: `Evaluate-Examination-Level.ipynb` to get Cross-Validation results.
4. Run: `Evaluate-Testset.ipynb` to get test set results (please use different set than used for training)
5. Run: `Results.ipynb` to calculate aggregated results.
