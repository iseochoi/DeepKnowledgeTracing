import os
import torch
import random
import numpy as np

from akt import AKT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


def model_isPid_type(model_name):
    words = model_name.split('_')
    is_pid = True if 'pid' in words else False
    return is_pid, words[0]


def load_model(params):
    words = params.model.split('_')
    model_type = words[0]
    is_cid = words[1] == 'cid'
    if is_cid:
        params.n_pid = -1

    if model_type in {'akt'}:
        model = AKT(n_question=params.n_question, n_pid=params.n_pid, n_blocks=params.n_block, d_model=params.d_model,
                    dropout=params.dropout, kq_same=params.kq_same, model_type=model_type, l2=params.l2).to(device)
    else:
        model = None
    return model


def setSeeds(seed = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True