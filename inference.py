import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
import math
from load_data import Preprocess, Preprocess_elo, DATA
from utils import load_model, setSeeds, model_isPid_type
from sklearn.model_selection import ShuffleSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transpose_data_model = {'akt'}

def test(net, params, q_data, qa_data, pid_data):
    # dataArray: [ array([[],[],..])] 
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  
    qa_data = qa_data.T 
    if pid_flag:
        pid_data = pid_data.T
    
    pred_list = []
        
    for idx in range(N):

        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   params.batch_size:(idx+1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx *
                             params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        # print 'seq_num', seq_num
        if model_type in transpose_data_model:
            # Shape (seqlen, batch_size)
            input_q = np.transpose(q_one_seq[:, :])
            # Shape (seqlen, batch_size)
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        
        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
                
        pred_list.append(pred)
      
    return pred_list

def inference(params, test_q_data,test_qa_data, test_pid):
    print("\n\nStart testing ......................\n ")
    
    model = load_model(params)
    test_qa = test_qa_data.copy()
    test_qa[test_qa_data<0]=0 # test idx
    if params.mode == 'ensemble':
        path = os.path.join('model', params.model,params.save) + '/*'
        count = 0
        for model_path in glob.glob(path):
            count += 1
            print("model path: ", model_path)
            print("count: ", count)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            pred_list = test(
                model, params, test_q_data, test_qa, test_pid)
            print("\ntest is done\t")

            all_qa = np.concatenate(test_qa_data, axis=0)
            all_pred = np.concatenate(pred_list, axis=0)
            if count ==1:
                preds=all_pred[all_qa<0]
            else:
                preds+=all_pred[all_qa<0]
        preds=preds/count
    else:
        checkpoint = torch.load(os.path.join('model', params.model, params.save) + params.mode)
        model.load_state_dict(checkpoint['model_state_dict'])
        pred_list = test(
                model, params, test_q_data, test_qa_data, test_pid)
        print("\ntest is done\t")

        all_qa = np.concatenate(test_qa_data, axis=0)
        all_pred = np.concatenate(pred_list, axis=0)
        preds=all_pred[all_qa<0]

    write_path = os.path.join(params.output_dir, f"{params.project}_{params.mode}.csv")
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))

if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    # Basic Parameters
    parser.add_argument('--seed', type=int, default=42, help='default seed')

    # Common parameters
    parser.add_argument('--batch_size', type=int,
                        default=1, help='the batch size')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    
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
    parser.add_argument('--output_dir', type=str, default="output")
    
    parser.add_argument('--test_file', type=str, default="test_data.csv")
    parser.add_argument('--project', type=str, default="AKT_elo")
    parser.add_argument('--fe_mode', type=str, default="elo", help="defualt or elo")

    parser.add_argument('--mode', type=str, default="ensemble", help="ensemble or model_ckpt name")
        
    params = parser.parse_args()

    params.save = params.project
    params.load = params.project

    # preprocess
    setSeeds(params.seed)

    if params.fe_mode == 'elo':
        preprocess = Preprocess_elo(params)
    else:
        preprocess = Preprocess(params)    

    preprocess.load_test_data(params.test_file)
    test_data = preprocess.get_test_data()
    print("\n")
    print("Preprocessing is done.")
    print("\n")

    # setup
    params.n_question = preprocess.args.n_questions
    params.n_pid = preprocess.args.n_tag

    dat = DATA(n_question=params.n_question,
                   seqlen=params.seqlen)
    
    test_q_data, test_qa_data, test_pid = dat.load_data(test_data)
    print("inference start!")
    print("test_q_data.shape", test_q_data.shape)
    print("test_qa_data.shape", test_qa_data.shape)
    ###Train- Test
    
    inference(params, test_q_data,test_qa_data, test_pid)

    
