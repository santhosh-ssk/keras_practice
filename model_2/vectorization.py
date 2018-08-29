from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import csv,sys
import numpy as np
from keras.preprocessing import sequence

class Text_Vectorize(object):

    def __init__(self):
        
        self.max_features=None
        self.text_vectorize=CountVectorizer()
        self.proccessor=self.text_vectorize.build_tokenizer()
        self.dataset=list()
        self.vocab=dict()
        self.max_len=0
        self.index2tag=dict()
        self.tag2index=dict()
        self.padded_dataset=np.zeros((0,))
    def text_tokenizer(self,text):
        return text.split()

    def load_dataset(self,file_path):
        #loads data from csv file
        with open(file_path,'r') as csv_file:
            reader=list(csv.reader(csv_file))
            for row in reader:
                if len(row)!=1:
                    continue
                self.dataset.append(row[0])
        print('Total Records:',len(self.dataset))

    def fit(self,data,max_feature,add_unknown):
        #prepares vocabulary from data
        self.max_features=max_feature
        print(self.max_features,max_feature,add_unknown)
        if max_feature!=None:
            self.text_vectorize=CountVectorizer(max_features=self.max_features,tokenizer=self.text_tokenizer)
        else:
            self.text_vectorize=CountVectorizer(tokenizer=self.text_tokenizer)
        self.text_vectorize.fit(data)
        self.vocab=self.text_vectorize.vocabulary_
        if add_unknown:
            print('added unknown')
            self.index2tag=[(index+1,tag) for tag,index in self.vocab.items()]
            self.index2tag.insert(0,(0,'UNK'))
            #self.index2tag.insert(1,(1,'<sol>'))
            #self.index2tag.insert(len(self.index2tag),(len(self.index2tag)-1,'<eol>'))
        else:
            self.index2tag=[(index,tag) for tag,index in self.vocab.items()]
            #self.index2tag.insert(0,(0,'<sol>'))
            #self.index2tag.insert(len(self.index2tag),(len(self.index2tag)-1,'<eol>'))
        self.tag2index=[(tag,index) for index,tag in self.index2tag]
        self.index2tag=dict(self.index2tag)
        self.tag2index=dict(self.tag2index) 
        print('Total Vocabulary:',len(self.index2tag))
        
    def transform(self,data,max_len=0):
        #transforms data to vectorized
        self.max_len=0
        self.padded_dataset=np.empty((len(data),),dtype=list)
        for i in range(len(data)):
            x=data[i].lower().strip()
            vectorized_text=list()
            tokenized_text=self.proccessor(x)
            """
            if i==0:
                print(tokenized_text)
            """
            if(len(tokenized_text)>self.max_len):
                self.max_len=len(tokenized_text)
            for token in tokenized_text:
                if token in self.tag2index:
                    vectorized_text.append([self.tag2index[token]])
                else:
                    #print(token)
                    vectorized_text.append([self.tag2index['UNK']])
            self.padded_dataset[i]=vectorized_text
        if max_len!=0:    
            self.max_len=max_len
        self.padded_dataset=sequence.pad_sequences(self.padded_dataset,self.max_len,padding='post').reshape((len(data),self.max_len))
        print('Total Vectorized records',self.padded_dataset.shape[0],self.padded_dataset.shape)
        print('Max Tokens in each records is:',self.max_len)
        #print(self.padded_dataset[0])
    
    def save_index_file(self,dir_path,input_flag=True):
        #saves vocabulary
        filename=''
        if input_flag:
            filename='ipt'
        else:
            filename='opt'
        with open(dir_path+'/'+filename+'_index2tag.csv','w') as data_file:
            writer=csv.writer(data_file)
            writer.writerows(self.index2tag.items())

        with open(dir_path+'/'+filename+'_tag2index.csv','w') as data_file:
            writer=csv.writer(data_file)
            writer.writerows(self.tag2index.items())
    
    def save_padded_dataset(self,file_name):
        with open(file_name,'w') as data_file:
            writer=csv.writer(data_file)
            writer.writerows(self.padded_dataset)
    
    def load_padded_dataset(self,file_name):
        self.padded_dataset=list()
        with open(file_name,'r') as data_file:
            reader=csv.reader(data_file)
            for row in reader:
                self.padded_dataset.append(row)
        self.padded_dataset=np.array(self.padded_dataset,dtype='float32')
        print(self.padded_dataset.shape)
        print('Total padded records:',self.padded_dataset.shape[0])
        #print(self.padded_dataset[0])

    def load_index_file(self,index2tag_file_path):
        with open(index2tag_file_path) as data_file:
            reader=csv.reader(data_file)
            for row in reader:
                if len(row)!=2:
                    continue
                index,tag = row
                index=int(index)
                self.index2tag[index]=tag
                self.tag2index[tag]=index
            print('Total index2tags:',len(self.index2tag))
    
        
if __name__=="__main__":
    args=sys.argv
    print(len(args))
    if len(args)==1:
        event=input('provide event name\n')
    else:
        event=args[1]
    if event=='prep':
        base_dir='/home/santhosh/resumes_folder/keras/model_2/data/'
        

        
        #preparing training input dataset
        print('preparing training input dataset')
        train_dataset_file=base_dir+'training_dataset_input.csv'
        save_padded_train_ipt_file=base_dir+'processed_data/padded_train_ipt.csv'
        train_dataset=Text_Vectorize()
        train_dataset.load_dataset(train_dataset_file)
        train_dataset.fit(train_dataset.dataset,7500-1,True)
        train_dataset.save_index_file(base_dir+'processed_data/')
        train_dataset.transform(train_dataset.dataset)
        train_dataset.save_padded_dataset(save_padded_train_ipt_file)

        #preparing training output dataset
        print('\n\npreparing training output dataset')
        train_dataset_opt_file=base_dir+'training_dataset_output.csv'
        save_padded_train_opt_file=base_dir+'processed_data/padded_train_opt.csv'
        train_opt_dataset=Text_Vectorize()
        train_opt_dataset.load_dataset(train_dataset_opt_file)
        train_opt_dataset.fit(train_opt_dataset.dataset,None,add_unknown=True)
        train_opt_dataset.save_index_file(base_dir+'processed_data/',input_flag=False)
        train_opt_dataset.transform(train_opt_dataset.dataset)
        train_opt_dataset.save_padded_dataset(save_padded_train_opt_file)
        
        
        #preparing testing input dataset
        print('\n\npreparing testing input dataset')
        test_ipt_dataset_file=base_dir+'testing_dataset_input.csv'
        save_test_ipt_padded_dataset_file=base_dir+'processed_data/padded_test_ipt.csv'
        load_index_file=base_dir+'processed_data/ipt_index2tag.csv'
        test_ipt_dataset=Text_Vectorize()
        test_ipt_dataset.load_index_file(load_index_file)
        test_ipt_dataset.load_dataset(test_ipt_dataset_file)
        test_ipt_dataset.transform(test_ipt_dataset.dataset,max_len=train_dataset.max_len)
        test_ipt_dataset.save_padded_dataset(save_test_ipt_padded_dataset_file)
        
        #preparing testing output dataset
        print('\n\npreparing testing output dataset')
        test_ipt_dataset_file=base_dir+'testing_dataset_output.csv'
        save_test_ipt_padded_dataset_file=base_dir+'processed_data/padded_test_opt.csv'
        load_index_file=base_dir+'processed_data/opt_index2tag.csv'
        test_ipt_dataset=Text_Vectorize()
        test_ipt_dataset.load_index_file(load_index_file)
        test_ipt_dataset.load_dataset(test_ipt_dataset_file)
        test_ipt_dataset.transform(test_ipt_dataset.dataset,train_opt_dataset.max_len)
        test_ipt_dataset.save_padded_dataset(save_test_ipt_padded_dataset_file)
        

