import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from load_data import Preprocess, Preprocess_elo, DATA
from run import train, test
from utils import try_makedirs, load_model, setSeeds
from sklearn.model_selection import ShuffleSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# assert torch.cuda.is_available(), "No Cuda available, AssertionError"


def train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid, n_fold):
    # ================================== model initialization ==================================

    model = load_model(params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")

    # ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        # Train Model
        train_loss, train_accuracy, train_auc = train(
            model, params, optimizer, train_q_data, train_qa_data, train_pid,  label='Train')
        # Validation step
        valid_loss, valid_accuracy, valid_auc = test(
            model,  params, optimizer, valid_q_data, valid_qa_data, valid_pid, label='Valid')

        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy,
              "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save,  file_name)+ f'{str(n_fold)}_fold' + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name)+ f'{str(n_fold)}_fold' +'_' + str(idx+1)
                       )
        if idx-best_epoch > 40:
            break   

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid,  best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    checkpoint = torch.load(os.path.join(
        'model', params.model, params.save, file_name) + '_'+str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_auc = test(
        model, params, None, test_q_data, test_qa_data, test_pid, label='Test')
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)

    # Now Delete all the models
    path = os.path.join('model', params.model, params.save,  file_name) + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    # Basic Parameters
    parser.add_argument('--max_iter', type=int, default=300,
                        help='number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='default seed')

    # Common parameters
    parser.add_argument('--optim', type=str, default='adam',
                        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')

    # AKT Specific Parameter
    parser.add_argument('--seqlen', type=int, default=200, help='default sequence length')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.05, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=1,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')
    parser.add_argument('--kq_same', type=int, default=1)

    # AKT-R Specific Parameter
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')

    # DKVMN Specific  Parameter
    parser.add_argument('--q_embed_dim', type=int, default=50,
                        help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256,
                        help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int,
                        default=50, help='memory size')
    parser.add_argument('--init_std', type=float, default=0.1,
                        help='weight initialization std')
    # DKT Specific Parameter
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lamda_r', type=float, default=0.1)
    parser.add_argument('--lamda_w1', type=float, default=0.1)
    parser.add_argument('--lamda_w2', type=float, default=0.1)

    # Datasets and Model
    parser.add_argument('--model', type=str, default='akt_pid')
    parser.add_argument('--data_dir', type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument('--asset_dir', type=str, default="asset")
    parser.add_argument('--train_file', type=str, default="train_all.csv")
    parser.add_argument('--test_file', type=str, default="test_data.csv")
    parser.add_argument('--project', type=str, default="AKT_elo")
    parser.add_argument('--fe_mode', type=str, default="elo", help="defualt or elo")

    parser.add_argument('--kfold', type=int, default=5)

    params = parser.parse_args()


    params.save = params.project
    params.load = params.project

    # preprocess
    setSeeds(params.seed)

    if params.fe_mode == 'elo':
        preprocess = Preprocess_elo(params)
    else:
        preprocess = Preprocess(params)    

    preprocess.load_train_data(params.train_file)
    train_data = preprocess.get_train_data()
    print("\n")
    print("Preprocessing is done.")
    print("\n")

    # setup
    params.n_question = preprocess.args.n_questions
    params.n_pid = preprocess.args.n_tag

    dat = DATA(n_question=params.n_question,
                   seqlen=params.seqlen)
    
    
    ###Train- Test
    if params.kfold>1:
        ss = ShuffleSplit(n_splits=params.kfold, test_size=0.3, random_state=params.seed)
        for n_fold, (train_set, vaild_set) in enumerate(ss.split(train_data)):
            train_q_data, train_qa_data, train_pid = dat.load_data(train_data[train_set])
            valid_q_data, valid_qa_data, valid_pid = dat.load_data(train_data[vaild_set])
            print("\n")
            print(f"{n_fold} fold start!", train_q_data.shape)
            print("train_q_data.shape", train_q_data.shape)
            print("train_qa_data.shape", train_qa_data.shape)
            print("valid_q_data.shape", valid_q_data.shape)  
            print("valid_qa_data.shape", valid_qa_data.shape)
            print("\n")

            best_epoch = train_one_dataset(
        params, params.project, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid, n_fold)

            print("\n")
            print(f"best epoch of {n_fold} fold", best_epoch)
            
 