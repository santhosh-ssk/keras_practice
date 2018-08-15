import csv,collections
from sklearn.model_selection import ShuffleSplit

def train_test_split(data_dir,dsc_dir,test_split=0.2):
    print("\ntrain_test_split Initiated.....")
    ipt_opt=list()
    try:
        with open(data_dir,'r') as dataset_file:
           reader=csv.reader(dataset_file)
           for row in  reader:
               if len(row)!=2:
                   continue
               ipt_opt.append(row)
        print("Total Records :",len(ipt_opt))
        train_test_split=ShuffleSplit(n_splits=1,test_size=test_split,random_state=42)
        train_indexs,test_indexs=list(train_test_split.split(ipt_opt))[0]
        test_ipt_opt=list()
        train_ipt_opt=list()
        for index in test_indexs:
            test_ipt_opt.append(ipt_opt[index])
        for index in train_indexs:
            train_ipt_opt.append(ipt_opt[index])
        try:
            with open(dsc_dir+"/test_dataset.csv","w") as test_dataset:
                writer=csv.writer(test_dataset)
                writer.writerows(test_ipt_opt)
        except:
            print("error in writing test dataset")
            return 0
        try:
            with open(dsc_dir+"/train_dataset.csv","w") as train_dataset:
                writer=csv.writer(train_dataset)
                writer.writerows(train_ipt_opt)
        except:
            print("error in writing training dataset")
            return 0
        print("Total no of records in Training dataset:",len(train_ipt_opt))
        print("Total no of records in Testing dataset:",len(test_ipt_opt))
        print("Total no of records in Training,Testing dataset:",len(train_ipt_opt)+len(test_ipt_opt),"\n\n")
        return 1
    except:
        print("Error")
        return 0

def prepare_vocab(data_dir,dsc_dir,ipt_max_count,opt_max_count):
    print("Preparation of Vocabulary for Training Dataset Initiated.....")
    ipt_vocab_tokens=collections.Counter()
    opt_vocab_tokens=collections.Counter()
    ipt_flag=0
    opt_flag=0
    try:
        with open(data_dir,'r') as ipt_file:
            reader=csv.reader(ipt_file)
            for row in reader:
                if len(row)!=2:
                    continue
                input_text,output_text=row
                input_text=input_text.strip().lower()
                output_text=output_text.strip().lower()
                
                for token in input_text.split():
                    ipt_vocab_tokens[token]+=1

                for token in output_text.split():
                    opt_vocab_tokens[token]+=1
        if ipt_max_count==None:
                ipt_max_count=len(ipt_vocab_tokens)
        else:
            if ipt_max_count<len(ipt_vocab_tokens):
                ipt_max_count-=1
                ipt_flag=1
        
        if opt_max_count==None:
                opt_max_count=len(opt_vocab_tokens)
        else:
            if opt_max_count<len(opt_vocab_tokens):
                opt_max_count-=1
                opt_flag=1        
        ipt_vocab_tokens=[token for token,count in ipt_vocab_tokens.most_common(ipt_max_count)]
        opt_vocab_tokens=[token for token,count in opt_vocab_tokens.most_common(opt_max_count)]
        
    
        ipt_vocab_tokens.insert(0,"UNK")
        if opt_flag==1:
            opt_vocab_tokens.insert(0,"UNK")
        
        ipt_token2index=[[token,i] for i,token in enumerate(ipt_vocab_tokens)]
        ipt_index2token=[[i,token] for i,token in enumerate(ipt_vocab_tokens)]

        opt_token2index=[[token,i] for i,token in enumerate(opt_vocab_tokens)]
        opt_index2token=[[i,token] for i,token in enumerate(opt_vocab_tokens)]
        
        with open(dsc_dir+"/train_ipt_token2index","w") as train_ipt_token2index:
            writer=csv.writer(train_ipt_token2index)
            writer.writerows(ipt_token2index)
        
        with open(dsc_dir+"/train_ipt_index2token","w") as train_ipt_index2token:
            writer=csv.writer(train_ipt_index2token)
            writer.writerows(ipt_index2token)
        
        with open(dsc_dir+"/train_opt_token2index","w") as train_opt_token2index:
            writer=csv.writer(train_opt_token2index)
            writer.writerows(opt_token2index)
        
        with open(dsc_dir+"/train_opt_index2token","w") as train_opt_index2token:
            writer=csv.writer(train_opt_index2token)
            writer.writerows(opt_index2token)
        
        print("Total No of Input tokens in Training Vocabulary are ",len(ipt_vocab_tokens))
        print("Total No of Output tokens in Training Vocabulary are ",len(opt_vocab_tokens),"\n")
        
    except:
        print("Error")

if __name__=="__main__":
    base_path="/home/santhosh/resumes_folder/keras/Model_1/data/"
    base_path="/home/santhosh/resumes_folder/keras/Model_1/__emberd_data__/data/"
    data_dir=base_path+"dataset.csv"
    dsc_dir=base_path
    if(train_test_split(data_dir,dsc_dir)):
        data_dir=base_path+"train_dataset.csv"
        prepare_vocab(data_dir,dsc_dir,None,None)
    