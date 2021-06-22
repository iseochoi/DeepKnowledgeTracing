import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm

from pathlib import Path
import re
import glob
import argparse

def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
    return theta + learning_rate_theta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
    )

def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
    return beta - learning_rate_beta(nb_previous_answers) * (
        is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
    )

def learning_rate_theta(nb_answers):
    return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

def learning_rate_beta(nb_answers):
    return 1 / (1 + 0.05 * nb_answers)

def probability_of_good_answer(theta, beta, left_asymptote):
    return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def estimate_parameters(answers_df, granularity_feature_name='assessmentItemID'):
    item_parameters = {
        granularity_feature_value: {"beta": 0, "nb_answers": 0}
        for granularity_feature_value in np.unique(answers_df[granularity_feature_name])
    }
    student_parameters = {
        student_id: {"theta": 0, "nb_answers": 0}
        for student_id in np.unique(answers_df.student_id)
    }

    print("Parameter estimation is starting...")

    for student_id, item_id, left_asymptote, answerCode in tqdm(
        zip(answers_df.student_id.values, answers_df[granularity_feature_name].values, answers_df.left_asymptote.values, answers_df.answerCode.values)
    ):
        theta = student_parameters[student_id]["theta"]
        beta = item_parameters[item_id]["beta"]

        item_parameters[item_id]["beta"] = get_new_beta(
            answerCode, beta, left_asymptote, theta, item_parameters[item_id]["nb_answers"],
        )
        student_parameters[student_id]["theta"] = get_new_theta(
            answerCode, beta, left_asymptote, theta, student_parameters[student_id]["nb_answers"],
        )
        
        item_parameters[item_id]["nb_answers"] += 1
        student_parameters[student_id]["nb_answers"] += 1

    print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
    return student_parameters, item_parameters


def update_parameters(answers_df, student_parameters, item_parameters, granularity_feature_name='assessmentItemID'):
    for student_id, item_id, left_asymptote, answerCode in tqdm(zip(
        answers_df.student_id.values, 
        answers_df[granularity_feature_name].values, 
        answers_df.left_asymptote.values, 
        answers_df.answerCode.values)
    ):
        if student_id not in student_parameters:
            student_parameters[student_id] = {'theta': 0, 'nb_answers': 0}
        if item_id not in item_parameters:
            item_parameters[item_id] = {'beta': 0, 'nb_answers': 0}
            
        theta = student_parameters[student_id]['theta']
        beta = item_parameters[item_id]['beta']

        student_parameters[student_id]['theta'] = get_new_theta(
            answerCode, beta, left_asymptote, theta, student_parameters[student_id]['nb_answers']
        )
        item_parameters[item_id]['beta'] = get_new_beta(
            answerCode, beta, left_asymptote, theta, item_parameters[item_id]['nb_answers']
        )
        
        student_parameters[student_id]['nb_answers'] += 1
        item_parameters[item_id]['nb_answers'] += 1

    print(f"Theta & beta estimations on {granularity_feature_name} are updated.")
        
    return student_parameters, item_parameters

def estimate_probas(test_df, student_parameters, item_parameters, granularity_feature_name='assessmentItemID'):
    probability_of_success_list = []
    
    for student_id, item_id, left_asymptote in tqdm(
        zip(test_df.student_id.values, test_df[granularity_feature_name].values, test_df.left_asymptote.values)
    ):
        theta = student_parameters[student_id]['theta'] if student_id in student_parameters else 0
        beta = item_parameters[item_id]['beta'] if item_id in item_parameters else 0

        probability_of_success_list.append(probability_of_good_answer(theta, beta, left_asymptote))

    return probability_of_success_list

def load_data(data_dir):
    df = pd.read_csv(
        filepath_or_buffer=data_dir,
        usecols=['userID', 'assessmentItemID', 'answerCode'],
        dtype = {'answerCode': 'int8'},
        )
    df.rename(columns={'userID': 'student_id'}, inplace=True)
    
    return df

def train_elo(train_df):
    training = train_df.copy()
    training = training[training.answerCode != -1]
    training['left_asymptote'] = 1/2

    print(f"Dataset of shape {training.shape}")
    print(f"Columns are {list(training.columns)}")
    
    student_parameters, item_parameters = estimate_parameters(training)
    
    return student_parameters, item_parameters

def update_test(test_df,student_parameters, item_parameters):
    test_copy = test_df.copy()
    test_copy = test_copy[test_copy['answerCode']!=-1]
    test_copy['left_asymptote']= 1/2
    student_parameters, item_parameters = update_parameters(test_copy, student_parameters, item_parameters)
    
    return student_parameters, item_parameters

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def inference(args, test_df,student_parameters, item_parameters):
    
    test_copy = test_df.copy()
    test_copy = test_copy[test_copy['answerCode']==-1]
    test_copy['left_asymptote']=1/2
    
    preds = estimate_probas(test_copy, student_parameters, item_parameters)

    save_dir = increment_path(os.path.join(args.output_dir, args.name), exist_ok=args.exist_ok)
    
    write_path = os.path.join(save_dir, 'test.csv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../input/data/train_dataset/train_data.csv')
    parser.add_argument('--test_dir', type=str, default='../input/data/train_dataset/test_data.csv')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--name', type=str, default='elo', help='result save at {output_dir}/{name}')

    parser.add_argument('--exist_ok', type=str, default='False')
    args = parser.parse_args()

    train_df = load_data(args.train_dir)
    student_parameters, item_parameters = train_elo(train_df)
    
    test_df = load_data(args.test_dir)
    student_parameters, item_parameters = update_test(test_df, student_parameters, item_parameters)

    inference(args, test_df,student_parameters, item_parameters)

