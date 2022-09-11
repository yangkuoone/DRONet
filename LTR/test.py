from models.LambdaMart.lambdamart import LambdaMART
from models.LambRank.lambdarank import LambdaRank
from models.RankNet.RankNet import *
from models.RankNet.utils import str2bool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd

def get_data(file_loc):
    f = open(file_loc, 'r')
    data = []
    for line in f:
        new_arr = []
        arr = line.split(' #')[0].split()
        score = arr[0]
        q_id = arr[1].split(':')[1]
        new_arr.append(int(score))
        new_arr.append(int(q_id))
        arr = arr[2:102]
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)

        
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
    parser.add_argument('--input_train', default='./mydata/train_1v1.txt',help='A training dataset of learning to rank.')
    parser.add_argument('--input_test', default='./mydata/test_1v5.txt',help='A test dataset of learning to rank.')
    parser.add_argument('--output',default='./resultdata/RankNet/example_RankNet_1v1.txt',help='Output file')                
    parser.add_argument('--method', required=True, choices=['LambdaMART','LambdaRank','RankNet'], help='The learning to rank method')
    #LambdaMart
    parser.add_argument('--number', default=30, type=int, help='The number of regression tree')
    parser.add_argument('--lr_LM', default=0.0005, type=float, help='learning rate of LambdaMART')
    #LambRank
    parser.add_argument('--h1_units', default=512, type=int, help='The first hidden layer')
    parser.add_argument('--h2_units', default=256, type=int, help='The second hidden layer')
    parser.add_argument('--epochs', default=100, type=int,help='The training epochs of LambdaRank')
    parser.add_argument('--lr_LR', default=0.0005, type=float,help='learning rate of LambdaRank')
    parser.add_argument('--k', default=10, type=int, help='used to compute the NDCG@k')
    #RankNet
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr_RN", type=float, default=0.0005,help='learning rate of RankNet')
    parser.add_argument("--optim", dest="optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--leaky_relu", dest="leaky_relu", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--ndcg_gain_in_train", dest="ndcg_gain_in_train",
        type=str, default="exp2", choices=["exp2", "identity"]
    )
    parser.add_argument("--small_dataset", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--double_precision", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--standardize", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="/tmp/ranking_output/")
    parser.add_argument(
        "--train_algo", dest="train_algo", default=SUM_SESSION,
        choices=[SUM_SESSION, ACC_GRADIENT, BASELINE],
        help=(
                "{}: Loss func sum on the session level,".format(SUM_SESSION) +
                "{}: compute gradient on session level, ".format(ACC_GRADIENT) +
                "{}: Loss func some on pairs".format(BASELINE)
        )
    )
    args = parser.parse_args()
    return args

def main(args):
  #training_data = get_data('./mydata/train_1v1_ran.txt')
  training_data = args.input_train
  #test_data = get_data('./mydata/test_1v5_ran.txt' )
  test_data = args.input_test
  if args.method == 'LambdaMART':
      model = LambdaMART(training_data, args.number, args.lr_LM, 'sklearn')
      model.fit()
      average_ndcg, mymetric, predicted_scores = model.validate(test_data, 10)
      print(args.number, average_ndcg, mymetric)
  elif args.method == 'LambdaRank':
      n_feature = training_data.shape[1] - 2
      model = LambdaRank(training_data, n_feature, args.h1_units, args.h2_units, args.epochs, args.lr_LR)
      model.fit()
      result=open("./resultdata/LambRank/example_lambdaRank_1v1.txt", 'w', encoding='utf8')
      ndcg, mymetric = model.validate(result, test_data, args.k)
      print("*******************testing*******************")
      print('Average NDCG : {}'.format(ndcg))
  elif args.method == 'RankNet':
      train_rank_net(
          args.start_epoch, args.additional_epoch, args.lr_RN, args.optim,
          args.train_algo,
          args.double_precision, args.standardize,
          args.small_dataset, args.debug,
          output_dir=args.output_dir,
      )


if __name__ == '__main__':
  main(parse_args())



