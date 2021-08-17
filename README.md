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

TetraPredX can be used to predict the origin of unknown sequences assembled from metagenomic or metatranscriptomics datasets. It can also be used in combination with [UnXplore](https://github.com/sejmodha/UnXplore "UnXplore") framework.

### Dependencies:

This tool requires the following Python modules are installed.

- Python 3.6 or higher
- BioPython
- joblib
- sklearn
- seaborn
- pathos

Note: PyPI will take care of this automatically.

OR

Use the `tetrapredx.yml` file to create a new environment with all dependencies.

### Installation and Usage:

```
# clone the data repository
git clone https://github.com/sejmodha/TetraPredX.git

# create conda environment
conda env create -f tetrapredx.yml

# activate new environment
conda activate tetrapredx

cd TetraPredX

# run predictions
python predict.py -i test_zetavirus.fa -o test_out

```

TetraPredX also supports training new models and using them for predictions.
A pseudo example shows major steps required to train and save new models.

**Note**: This process may require substantial computing power and could take a long time depending on the input data size.

#### Step 1
Extract features and their frequencies using `FeatureExtractor.py` script.

```
# generate feature output output output file
python FeatureExtractor.py -i mysequences.fasta -o output_prefix
```
Output file generated from [Step 1](#Step-1) can be used as input for the next step.

#### Step 2
Train and save the models using `train.py`.

```
# train new models
python train.py -i input_csv_with_features_and_label -o output_prefix
```
`FeatureExtractor.py` and `TrainModels.py` contain a range of functions that can be used by importing them as standard python modules.
e.g.,

```
import FeatureExtractor as ft

# generate a feature table
df = ft.get_feature_table(....)
```

#### Further details on functions:

Help on module FeatureExtractor:

```
NAME
    FeatureExtractor - Created on Thu 30 Apr 11:48:35 BST 2020

DESCRIPTION
    @author: sejmodha

FUNCTIONS
    batch_iterator(iterator, batch_size)
        Return lists of length batch_size.

        This can be used on any iterator, for example to batch up
        SeqRecord objects from Bio.SeqIO.parse(...), or to batch
        Alignment objects from Bio.AlignIO.parse(...), or simply
        lines from a file handle.

        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.

        Taken from: https://biopython.org/wiki/Split_large_file

    extract_feat(infasta, tax_label, kmer, cpu, chunk)
        Extract k-mer features from a given FASTA file.

        Returns a dataframe with indexes, features and
        sequences labels (when known).

    generate_list_for_record(record, k)
        Generate a list of seq and revcomp seq kmers.

    generate_primer_ngrams(k, n)
        Generate n-grams of words.

    generate_primers(length)
        Generate primers.

    get_feature_table(infasta, out, tax_label, kmer, cpu, chunk)
        Convert feature table to a .csv file.

    get_kmers(dna, k)
        Extract k-mers of defined size k. Returns a list  of kmers.

    is_fasta(filename)
        Check the validity of FASTA file.

    main()
        Run the module as a script.

    set_vars()
        Set variables for the module.
```

Help on module TrainModels:

```
NAME
    TrainModels - Created on Wed 24 Mar 15:37:43 GMT 2021.

DESCRIPTION
    @author: sejmodha

FUNCTIONS
    get_best_model(X, y, cpu)
        Run GriSearchCV to identity the best model parameters.

    get_train_test(input_df, label_col, test_size, k, n_features)
        Generate train/test set for each class.

    main()
        Run the module as a script.

    set_vars()
        Set var_list required for the module.

    train_models_rfc(data_dict, out, path, cpu, cv)
        Run the Random forest classifier and saves models.

```
