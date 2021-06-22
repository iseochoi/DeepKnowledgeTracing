import numpy as np
import math
import os
from datetime import datetime
import time
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
from elo import *
import pickle

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = ['assessmentItemID', 'KnowledgeTag']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            
            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(
        filepath_or_buffer=csv_file_path,
        usecols=['userID','Timestamp', 'assessmentItemID', 'answerCode', 'KnowledgeTag'])
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
                
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'answerCode', 'KnowledgeTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)

class Preprocess_elo:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = ['assessmentItemID', 'diffTag']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            
            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df, is_train): # elo rating

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        
        if is_train:
            elo_df=df.copy()
            elo_df.rename(columns={'userID': 'student_id'}, inplace=True)
            student_parameters, item_parameters = train_elo(elo_df)
            diff=pd.DataFrame(item_parameters).T

            diff_index = pd.qcut(diff['beta'], 100, labels=False).to_dict()
            df['diffTag'] = df.assessmentItemID.map(lambda x: diff_index.get(x,x))

            # save data
            label_path = os.path.join(self.args.asset_dir, 'diffTag.pickle')
            with open(label_path,'wb') as fw:
                pickle.dump(diff_index, fw)
            
        else:
            label_path = os.path.join(self.args.asset_dir, 'diffTag.pickle')
            with open(label_path, 'rb') as fr:
                diff_index = pickle.load(fr)
            df['diffTag'] = df.assessmentItemID.map(lambda x: diff_index.get(x,x))

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(
        filepath_or_buffer=csv_file_path,
        usecols=['userID','Timestamp', 'assessmentItemID', 'answerCode'])
        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
                
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'diffTag_classes.npy')))
        


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'answerCode', 'diffTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['assessmentItemID'].values,
                    r['diffTag'].values,
                    r['answerCode'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DATA(object):
    def __init__(self, n_question,  seqlen):
    
        self.seqlen = seqlen
        self.n_question = n_question
    
    def load_data(self, f_data):
        q_data = []
        qa_data = []
        p_data = []
        for idx, line in enumerate(f_data):
            Q = line[0]
            P = line[1]
            A = line[2]

            # start split the data
            n_split = 1
            # print('len(Q):',len(Q))
            if len(Q) > self.seqlen:
                n_split = math.floor(len(Q) / self.seqlen)
                if len(Q) % self.seqlen:
                    n_split = n_split + 1
            # print('n_split:',n_split)
            for k in range(n_split):
                question_sequence = []
                problem_sequence = []
                answer_sequence = []
                if k == n_split - 1:
                    endINdex = len(A)
                else:
                    endINdex = (k+1) * self.seqlen
                for i in range(k * self.seqlen, endINdex):
                    if Q[i]>0:
                        Xindex = int(Q[i]) + int(A[i]) * self.n_question
                        question_sequence.append(int(Q[i]))
                        problem_sequence.append(int(P[i]))
                        answer_sequence.append(Xindex)
                    else:
                        # print(Q[i])
                        pass
                q_data.append(question_sequence)
                qa_data.append(answer_sequence)
                p_data.append(problem_sequence)


        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return q_dataArray, qa_dataArray, p_dataArray
