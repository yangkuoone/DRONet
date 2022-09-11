# -*- coding: utf-8 -*-

import datetime
import getpass
import json
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from embed_train import embedding_training, load_embedding, read_node_labels, split_train_test_graph
from evaluation import LinkPrediction, NodeClassification
from scipy.spatial import distance
from tqdm import tqdm
result=open("/home/fsy/fsy/paper/DRONet_sum/new_BioNEV/mydata/result.txt",'w',encoding='utf8')
bp='/home/fsy/fsy/paper/DRONet_sum/new_BioNEV/'


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', default=bp+'mydata/LTR/graph+drug.edgelist',
                        help='Input graph file. Only accepted edgelist format.')
    parser.add_argument('--output',default=bp+'result/DeepWalk/graph+drug_100_embeddings.txt',
                        help='Output graph embedding file')


    parser.add_argument('--task', choices=[
        'none',
        'link-prediction',
        'node-classification'], default='none',
                        help='Choose to evaluate the embedding quality based on a specific prediction task. '
                             'None represents no evaluation, and only run for training embedding.')
    parser.add_argument('--testingratio', default=0.2, type=float,
                        help='Testing set ratio for prediction tasks.'
                             'In link prediction, it splits all the known edges; '
                             'in node classification, it splits all the labeled nodes.')
    parser.add_argument('--number-walks', default=64, type=int,
                        help='Number of random walks to start at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--walk-length', default=128, type=int,
                        help='Length of the random walk started at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--workers', default=16, type=int,
                        help='Number of parallel processes. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--dimensions', default=100, type=int,
                        help='the dimensions of embedding for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of word2vec model. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--epochs', default=500, type=int,
                        help='The training epochs of LINE, SDNE and GAE')
    parser.add_argument('--p', default=1.0, type=float,
                        help='p is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk explores.')
    parser.add_argument('--q', default=1.0, type=float,
                        help='q is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk leaves the neighborhood of starting node.')


    parser.add_argument('--method',default='DeepWalk', choices=[
        'Laplacian',
        'GF',
        'SVD',
        'HOPE',
        'GraRep',
        'DeepWalk',
        'node2vec',
        'struc2vec',
        'LINE',
        'SDNE',
        'GAE'
    ], help='The embedding learning method')


    parser.add_argument('--label-file', default='',
                        help='The label file for node classification')
    parser.add_argument('--negative-ratio', default=7, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--directed', type=bool, default=True,
                        help='Treat graph as directed')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='coefficient for L2 regularization for Graph Factorization.')
    parser.add_argument('--kstep', default=5, type=int,
                        help='Use k-step transition probability matrix for GraRep.')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=0.3, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=0, type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    parser.add_argument('--OPT1', default=True, type=bool,
                        help='optimization 1 for struc2vec')
    parser.add_argument('--OPT2', default=True, type=bool,
                        help='optimization 2 for struc2vec')
    parser.add_argument('--OPT3', default=True, type=bool,
                        help='optimization 3 for struc2vec')
    parser.add_argument('--until-layer', type=int, default=6,
                        help='Calculation until the layer. A hyper-parameter for struc2vec.')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--hidden', default=32, type=int, help='Number of units in hidden layer.')
    parser.add_argument('--gae_model_selection', default='gcn_ae', type=str,
                        help='gae model selection: gcn_ae or gcn_vae')


    parser.add_argument('--eval-result-file',default='', help='save evaluation performance')



    parser.add_argument('--seed',default=0, type=int,  help='seed value')
    args = parser.parse_args()

    return args

def main(args):
    print('#' * 70)
    print('Embedding Method: %s, Evaluation Task: %s' % (args.method, args.task))
    print('#' * 70)
    train_graph_filename = args.input
    time1 = time.time()
    embedding_training(args, train_graph_filename)
    embed_train_time = time.time() - time1
    print('Embedding Learning Time: %.2f s' % embed_train_time)


def more_main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    main(parse_args())


if __name__ == "__main__":
    more_main()
    result.close()
