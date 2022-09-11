import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from .RegressionTree import RegressionTree
import pandas as pd
import pickle


def dcg(scores):
    """
        Returns the DCG value of the list of scores.        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([
        (np.power(2, scores[i]) - 1) / np.log2(i + 2)
        for i in range(len(scores))
    ])


def dcg_k(scores, k):
    """
        Returns the DCG value of the list of scores and truncates to k values.        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([
        (np.power(2, scores[i]) - 1) / np.log2(i + 2)
        for i in range(len(scores[:k]))
    ])


def ideal_dcg(scores):
    """
        Returns the Ideal DCG value of the list of scores.      Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg(scores)


def ideal_dcg_k(scores, k):
    """
        Returns the Ideal DCG value of the list of scores and truncates to k values.       Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


def single_dcg(scores, i, j):
    """
        Returns the DCG value at a single point.        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        i : int
            This points to the ith value in scores
        j : int
            This sets the ith value in scores to be the jth rank

        Returns
        -------
        Single_DCG: int
            This is the value of the DCG at a single point
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j + 2)


def compute_lambda(args):
    """
        Returns the lambda and w values for a given query.       Parameters
        ----------
        args : zipped value of true_scores, predicted_scores, good_ij_pairs, idcg, query_key
            Contains a list of the true labels of documents, list of the predicted labels of documents,
            i and j pairs where true_score[i] > true_score[j], idcg values, and query keys.

        Returns
        -------
        lambdas : numpy array
            This contains the calculated lambda values
        w : numpy array
            This contains the computed w values
        query_key : int
            This is the query id these values refer to
    """

    true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args
    num_docs = len(true_scores)
    sorted_indexes = np.argsort(predicted_scores)[::-1]
    rev_indexes = np.argsort(sorted_indexes)
    true_scores = true_scores[sorted_indexes]
    predicted_scores = predicted_scores[sorted_indexes]

    lambdas = np.zeros(num_docs)
    w = np.zeros(num_docs)

    single_dcgs = {}
    for i, j in good_ij_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in good_ij_pairs:
        z_ndcg = abs(single_dcgs[(i, j)] - single_dcgs[(i, i)] + single_dcgs[(j, i)] - single_dcgs[(j, j)]) / idcg
        rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        rho_complement = 1.0 - rho
        lambda_val = z_ndcg * rho
        lambdas[i] += lambda_val
        lambdas[j] -= lambda_val

        w_val = rho * rho_complement * z_ndcg
        w[i] += w_val
        w[j] += w_val

    return lambdas[rev_indexes], w[rev_indexes], query_key


def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.     Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training Fold1

        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training Fold1 that are associated of those keys.
    """
    query_indexes = {}
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


def get_pairs(scores):
    """
        Returns pairs of indexes where the first value in the pair has a higher score than the second value in the pair   Parameters
        ----------
        scores : list of int
            Contain a list of numbers

        Returns
        -------
        query_pair : list of pairs
            This contains a list of pairs of indexes in scores.
    """

    query_pair = []
    for query_scores in scores:
        temp = sorted(query_scores, reverse=True)
        pairs = []
        for i in range(len(temp)):
            for j in range(len(temp)):
                if temp[i] > temp[j]:
                    pairs.append((i, j))
        query_pair.append(pairs)
    return query_pair


class LambdaMART:

    def __init__(self, training_data=None, number_of_trees=30, learning_rate=0.0005, tree_type='sklearn'):
        """
        This is the constructor for the LambdaMART object.
        Parameters
        ----------
        training_data : list of int
            Contain a list of numbers
        number_of_trees : int (default: 5)
            Number of trees LambdaMART goes through
        learning_rate : float (default: 0.1)
            Rate at which we update our prediction with each tree
        tree_type : string (default: "sklearn")
            Either "sklearn" for using Sklearn implementation of the tree of "original"
            for using our implementation
        """

        if tree_type != 'sklearn' and tree_type != 'original':
            raise ValueError('The "tree_type" must be "sklearn" or "original"')
        self.training_data = training_data
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.trees = []
        self.tree_type = tree_type

    def fit(self):
        """
        Fits the model on the training Fold1.
        """

        predicted_scores = np.zeros(len(self.training_data))
        query_indexes = group_queries(self.training_data, 1)
        query_keys = query_indexes.keys()
        true_scores = [self.training_data[query_indexes[query], 0] for query in query_keys]
        good_ij_pairs = get_pairs(true_scores)
        tree_data = pd.DataFrame(self.training_data[:, 2:7])
        labels = self.training_data[:, 0]

        # ideal dcg calculation
        idcg = [ideal_dcg(scores) for scores in true_scores]

        for k in range(self.number_of_trees):
            # print('Tree %d' % (k))
            lambdas = np.zeros(len(predicted_scores))
            w = np.zeros(len(predicted_scores))
            pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]

            pool = Pool()
            for lambda_val, w_val, query_key in pool.map(compute_lambda,
                                                         zip(true_scores, pred_scores, good_ij_pairs, idcg, query_keys),
                                                         chunksize=1):
                indexes = query_indexes[query_key]
                lambdas[indexes] = lambda_val
                w[indexes] = w_val
            pool.close()

            if self.tree_type == 'sklearn':
                # Sklearn implementation of the tree
                tree = DecisionTreeRegressor(max_depth=50)
                tree.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(tree)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
            elif self.tree_type == 'original':
                # Our implementation of the tree
                tree = RegressionTree(tree_data, lambdas, max_depth=10, ideal_ls=0.001)
                tree.fit()
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate

    def predict(self, data):
        """
        Predicts the scores for the test dataset.
        Parameters
        ----------
        data : Numpy array of documents
            Numpy array of documents with each document's format is [query index, feature vector]

        Returns
        -------
        predicted_scores : Numpy array of scores
            This contains an array or the predicted scores for the documents.
        """
        data = np.array(data)
        query_indexes = group_queries(data, 0)
        predicted_scores = np.zeros(len(data))
        for query in query_indexes:  # 
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 1:])
            predicted_scores[query_indexes[query]] = results
        return predicted_scores

    def validate(self, data, k):
        """
        Predicts the scores for the test dataset and calculates the NDCG value.
        Parameters
        ----------
        data : Numpy array of documents
            Numpy array of documents with each document's format is [relevance score, query index, feature vector]
        k : int
            this is used to compute the NDCG@k

        Returns
        -------
        average_ndcg : float
            This is the average NDCG value of all the queries
        predicted_scores : Numpy array of scores
            This contains an array or the predicted scores for the documents.
        """
        data = np.array(data)
        query_indexes = group_queries(data, 1)
        average_ndcg1 = []
        average_ndcg3 = []
        average_ndcg5 = []
        average_ndcg10 = []
        average_ndcg30 = []
        result=open("./resultdata/LambdaMart/example_lambdaMart_ran_1v5.txt",'w',encoding='utf8')

        mymetric=[]
        predicted_scores = np.zeros(len(data))
        hit10 = 0
        hit3 = 0
        hit1 = 0
        mrr = 0
        for query in query_indexes:
            h1 = 0
            h3 = 0
            h10 = 0
            mr = 0
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
            index_score = {}
            for q, s in zip(query_indexes[query], results):
                index_score[q] = s

            rank = sorted(index_score.items(), key=lambda item: item[1], reverse=True)

            example = list(dict(rank).keys())[:10]
            lst = []
            for e in example:
                lst.append(str(e))
            result.write(str(int(query)) + "\t" + ",".join(lst) + "\n")

            predicted_sorted_indexes = np.argsort(results)[::-1]

            t_results = data[query_indexes[query], 0]
            index_label = {}

            for q, i in zip(query_indexes[query], t_results):
                if i > 0:
                    index_label[q] = i
            for v in (index_label):
                ylist = list(dict(rank).keys())
                mr += 1 / (int(ylist.index(v)) + 1)
                if v in ylist[0:1]:
                    h1 += 1
                if v in ylist[:3]:
                    h3 += 1
                if v in ylist[:10]:
                    h10 += 1
            num = len(index_label)
            hit10 += h10 / num
            hit3 += h3 / num
            hit1 += h1 / num
            mrr += mr / num
            t_results = t_results[predicted_sorted_indexes]
            predicted_scores[query_indexes[query]] = results
            dcg_val1 = dcg_k(t_results, 1)
            idcg_val1 = ideal_dcg_k(t_results, 1)
            if dcg_val1 == 0:
                ndcg_val1 = 0.0
            else:
                ndcg_val1 = (dcg_val1 / idcg_val1)
            average_ndcg1.append(ndcg_val1)

            dcg_val3 = dcg_k(t_results, 3)
            idcg_val3 = ideal_dcg_k(t_results, 3)
            if dcg_val3 == 0:
                ndcg_val3 = 0.0
            else:
                ndcg_val3 = (dcg_val3 / idcg_val3)
            average_ndcg3.append(ndcg_val3)

            dcg_val5 = dcg_k(t_results, 5)
            idcg_val5 = ideal_dcg_k(t_results, 5)
            if dcg_val5 == 0:
                ndcg_val5 = 0.0
            else:
                ndcg_val5 = (dcg_val5 / idcg_val5)
            average_ndcg5.append(ndcg_val5)

            dcg_val10 = dcg_k(t_results, 10)
            idcg_val10 = ideal_dcg_k(t_results, 10)
            if dcg_val10 == 0:
                ndcg_val10 = 0.0
            else:
                ndcg_val10 = (dcg_val10 / idcg_val10)
            average_ndcg10.append(ndcg_val10)


            dcg_val30 = dcg_k(t_results, 30)
            idcg_val30 = ideal_dcg_k(t_results, 30)
            if dcg_val30 == 0:
                ndcg_val30 = 0.0
            else:
                ndcg_val30 = (dcg_val30 / idcg_val30)
            average_ndcg30.append(ndcg_val30)


        sum = len(query_indexes)
        print("total test:", sum)
        print("hit@10:", hit10 / sum)
        print("hit@3:", hit3 / sum)
        print("hit@1:", hit1 / sum)
        print("mrr:", mrr / sum)
        mymetric=[hit10 / sum,hit3 / sum,hit1 / sum,mrr / sum]
        result.close()
        return np.mean(average_ndcg10),mymetric ,predicted_scores

        # return [np.mean(average_ndcg1),np.mean(average_ndcg3),np.mean(average_ndcg5),np.mean(average_ndcg10),np.mean(average_ndcg30)], 0, predicted_scores


def save(self, fname):
    """
    Saves the model into a ".lmart" file with the name given as a parameter.
    Parameters
    ----------
    fname : string
        Filename of the file you want to save

    """
    pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)


def load(self, fname):
    """
    Loads the model from the ".lmart" file given as a parameter.
    Parameters
    ----------
    fname : string
        Filename of the file you want to load

    """
    model = pickle.load(open(fname, "rb"))
    self.training_data = model.training_data
    self.number_of_trees = model.number_of_trees
    self.tree_type = model.tree_type
    self.learning_rate = model.learning_rate
    self.trees = model.trees
