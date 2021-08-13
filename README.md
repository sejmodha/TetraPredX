    ####################################################################
    ####################################################################
    ##                                                                ##
    ##       _____    _             ____               ___  __        ##
    ##      |_   _|__| |_ _ __ __ _|  _ \ _ __ ___  __| \ \/ /        ##
    ##        | |/ _ \ __| '__/ _` | |_) | '__/ _ \/ _` |\  /         ##
    ##        | |  __/ |_| | | (_| |  __/| | |  __/ (_| |/  \         ## 
    ##        |_|\___|\__|_|  \__,_|_|   |_|  \___|\__,_/_/\_\        ##
    ##                                                                ##
    ##                                                                ## 
    ####################################################################
    ####################################################################

# TetraPredX

Microbial sequence predictor using short DNA features.

This tool requires the following Python modules are installed.

- Python 3.6 or higher
- BioPython
- joblib
- sklearn
- seaborn
- pathos 

Note: PyPI will take care of this automatically.

OR

Use the `seqpred.yml` file to create a new environment with all dependencies.
```
# create conda environment
conda env create -f seqpred.yml

# activate new environment
conda activate seqpred

# clone the data repository
git clone https://github.com/sejmodha/SequencePredictor.git

cd SequencePredictor

# run predictions
python predict.py -i test_zetavirus.fa -o test_out

# train new models
python train.py -i input_csv_with_features_and_label -o output_prefix

```

