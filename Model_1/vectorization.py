import csv
import numpy as np
def vectorization(dataset_path,ipt_vocab_path,opt_vocab_path,dsc_path,file_name):
    ipt_token2index=dict()
    opt_token2index=dict()

    try:
        with open(ipt_vocab_path,'r') as ipt_vocab_file:
            reader=csv.reader(ipt_vocab_file)
            for row in reader:
                if len(row)!=2:
                    continue
                key,val=row
                ipt_token2index[key]=val
        print("Total no of input vocab:",len(ipt_token2index))
    except:
        print("Error in Input Vocab File")
        return

    try:
        with open(opt_vocab_path,'r') as opt_vocab_file:
            reader=csv.reader(opt_vocab_file)
            for row in reader:
                if len(row)!=2:
                    continue
                key,val=row
                opt_token2index[key]=val
        opt_vocab_size=len(opt_token2index)
        print("Total no of Output vocab:",len(opt_token2index))
    except:
        print("Error in Output Vocab File")
        return
    
    try:
        with open(dataset_path,'r') as dataset_file:
            reader=csv.reader(dataset_file)
            ipt_opt=list()
            for row in reader:
                if len(row)!=2:
                    continue
                input_text,output_text=row
                input_vector=list()
                output_vector=np.zeros(opt_vocab_size)
                for token in input_text.split():
                    if token not in ipt_token2index:
                        input_vector.append(ipt_token2index["UNK"])
                    else:
                        input_vector.append(ipt_token2index[token])
                
                for token in output_text.split():
                    if token not in opt_token2index:
                        opt_index=opt_token2index["UNK"]
                    else:
                        opt_index=opt_token2index[token]
                    output_vector[int(opt_index)]=1
                ipt_opt.append([" ".join(input_vector)," ".join(map(str,output_vector))])
            with open(dsc_path+file_name+"_vextorized_ipt_opt.csv","w") as vector_file:
                writer=csv.writer(vector_file)
                writer.writerows(ipt_opt)
        print("Total no of records:",len(ipt_opt),"\n\n")
    
    except:
        print("Error in dataset File")
        return



    
if __name__ =="__main__":
    base_path="/home/santhosh/resumes_folder/keras/Model_1/data/"
    base_path="/home/santhosh/resumes_folder/keras/Model_1/__emberd_data__/data/"
    dataset_path=base_path+"train_dataset.csv"
    ipt_vocab_path=base_path+"train_ipt_token2index"
    opt_vocab_path=base_path+"train_opt_token2index"
    dsc_path=base_path
    file_name="training"
    vectorization(dataset_path,ipt_vocab_path,opt_vocab_path,dsc_path,file_name)
    
    dataset_path=base_path+"test_dataset.csv"
    ipt_vocab_path=base_path+"train_ipt_token2index"
    opt_vocab_path=base_path+"train_opt_token2index"
    dsc_path=base_path
    file_name="testing"
    vectorization(dataset_path,ipt_vocab_path,opt_vocab_path,dsc_path,file_name)
    