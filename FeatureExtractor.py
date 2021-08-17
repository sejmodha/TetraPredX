#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 30 Apr 11:48:35 BST 2020

@author: sejmodha
"""

import argparse
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from pathos.pools import ProcessPool
import functools
from itertools import combinations


def set_vars():
    """Set variables for the module."""
    parser = argparse.ArgumentParser(description='This script calculates tetranucleotide frequence for a given fasta file')
    parser.add_argument('-i', '--inputFASTA', help='Input a valid FASTA file', required=True)
    parser.add_argument('-o', '--outputPrefix', help='Specify output file name prefix', required=True)
    parser.add_argument('-l', '--label', help='Input a valid taxonomic superkingdom label e.g. Bacteria', required=False, default='unknown', type=str)
    parser.add_argument('-k', '--kmer', help='Input a kmer length', required=False, default=4, type=int)
    parser.add_argument('-t', '--threads', help='Input number of threads to use', required=False, default=2, type=int)
    parser.add_argument('-b', '--batchsize', help='Input a batchsize to split input file', required=False, default=100, type=int)
    parser.add_argument('-path', '--modelpath', help='Provide locations to model files', required=False, default='models')

    args = parser.parse_args()

    infasta = args.inputFASTA
    out = args.outputPrefix
    tax_label = args.label
    kmer = args.kmer
    cpu = args.threads
    chunk = args.batchsize
    path = args.modelpath

    print('\nInput parameters are set as following:')
    print(f'Input file: {infasta}')
    print(f'Output prefix: {out}')
    # print(f'Taxonomy label: {tax_label}')
    print(f'kmer: {kmer}')
    print(f'CPU: {cpu}')
    print(f'Chunk size: {chunk}\n')

    return (infasta, out, tax_label, kmer, cpu, chunk, path)


def is_fasta(filename):
    """Check the validity of FASTA file."""
    with open(filename, 'r') as handle:
        fasta = SeqIO.parse(handle, 'fasta')
        return any(fasta)


def generate_primers(length):
    """Generate primers."""
    if length == 1:
        return ['A', 'T', 'G', 'C']
    else:
        result = []
        for seq in generate_primers(length - 1):
            for base in ['A', 'T', 'G', 'C']:
                result.append(seq + base)
        return result


def generate_primer_ngrams(k, n):
    """Generate n-grams of words."""
    if k == 1:
        return ['A', 'T', 'G', 'C']
    else:
        result = []
        for seq in generate_primers(k - 1):
            for base in ['A', 'T', 'G', 'C']:
                result.append(seq + base)
        return [' '.join(tup) for tup in list(combinations(result, n))]


def get_kmers(dna, k):
    """Extract k-mers of defined size k. Returns a list  of kmers."""
    kmers = []
    # print(len(generate_primers(4)))
    for i in range(len(dna) - k + 1):
        kmers.append(dna[i:i+k].lower())
    return kmers


def generate_list_for_record(record, k):
    """Generate a list of seq and revcomp seq kmers."""
    record_list = []
    record_list.append(record.description)
    seq_kmers = get_kmers(str(record.seq), k)
    revcomp_kmers = get_kmers(str(record.seq.reverse_complement()), k)
    record_list.append(' '.join(seq_kmers)+' '+' '.join(revcomp_kmers))
    return record_list


# following function was taken from: https://biopython.org/wiki/Split_large_file
def batch_iterator(iterator, batch_size):
    """Return lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch


def extract_feat(infasta, tax_label, kmer, cpu, chunk):
    """Extract k-mer features from a given FASTA file.

    Returns a dataframe with indexes, features and
    sequences labels (when known).
    """
    record_iter = SeqIO.parse(open(infasta), "fasta")
    slice = chunk
    df = pd.DataFrame()
    # print('\nGenerating feature set from FASTA file.\n')
    for i, batch in enumerate(batch_iterator(record_iter, slice)):

        # print(f'\nprocessing batch {i}\n')
        pool = ProcessPool(nodes=cpu)
        # creating a bag of words model here
        tv = TfidfVectorizer(norm='l1', use_idf=False, smooth_idf=False)
        tv.fit_transform(generate_primers(kmer))

        seq_list = pool.map(functools.partial(generate_list_for_record, k=kmer), (rec for rec in batch))
        cols = ['id', 'sentence']
        tmp_df = pd.DataFrame(seq_list, columns=cols)
        # print(f'transforming the counts using TfTdifvectorizer\nCurrent Time is {datetime.now()}')
        seq_kmers = tv.transform(tmp_df['sentence'])

        # print(seq_kmers)
        # seq_kmers = functools.reduce(operator.iconcat, seq_kmers, [])
        seq_kmers_df = pd.DataFrame(seq_kmers.todense(), columns=tv.get_feature_names(), index=tmp_df.id)
        df = df.append(seq_kmers_df)
    pool.close()
    #  print('Adding taxonomy label')
    df['class'] = tax_label
    # print(df.shape)

    return df


def get_feature_table(infasta, out, tax_label, kmer, cpu, chunk):
    """Convert feature table to a .csv file."""
    if is_fasta(infasta):
        pass
    else:
        print('Input a valid fasta file using parameter -i ')

    df = extract_feat(infasta, tax_label, kmer, cpu, chunk)
    # print(df)
    # Save df to a file
    # df.to_csv(out+'_feat.csv.gz', compression='gzip', index=True, header=True)
    return df


def main():
    """Run the module as a script."""
    infasta, out, tax_label, kmer, cpu, chunk, path = set_vars()
    # To get feature table in a dataframe and .csv
    get_feature_table(infasta, out, tax_label, kmer, cpu, chunk)


if __name__ == '__main__':
    main()
