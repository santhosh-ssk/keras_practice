import csv
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
                output_vector=list()
                for token in input_text.split():
                    if token not in ipt_token2index:
                        input_vector.append(ipt_token2index["UNK"])
                    else:
                        input_vector.append(ipt_token2index[token])
                
                for token in output_text.split():
                    if token not in opt_token2index:
                        output_vector.append(opt_token2index["UNK"])
                    else:
                        output_vector.append(opt_token2index[token])
                ipt_opt.append([" ".join(input_vector)," ".join(output_vector)])
            with open(dsc_path+file_name+"_vextorized_ipt_opt.csv","w") as vector_file:
                writer=csv.writer(vector_file)
                writer.writerows(ipt_opt)
        print("Total no of records:",len(ipt_opt),"\n\n")
    
    except:
        print("Error in dataset File")
        return



    
if __name__ =="__main__":
    dataset_path="/home/santhosh/resumes_folder/keras/Model_1/data/train_dataset.csv"
    ipt_vocab_path="/home/santhosh/resumes_folder/keras/Model_1/data/train_ipt_token2index"
    opt_vocab_path="/home/santhosh/resumes_folder/keras/Model_1/data/train_opt_token2index"
    dsc_path="/home/santhosh/resumes_folder/keras/Model_1/data/"
    file_name="training"
    vectorization(dataset_path,ipt_vocab_path,opt_vocab_path,dsc_path,file_name)
    
    dataset_path="/home/santhosh/resumes_folder/keras/Model_1/data/test_dataset.csv"
    ipt_vocab_path="/home/santhosh/resumes_folder/keras/Model_1/data/train_ipt_token2index"
    opt_vocab_path="/home/santhosh/resumes_folder/keras/Model_1/data/train_opt_token2index"
    dsc_path="/home/santhosh/resumes_folder/keras/Model_1/data/"
    file_name="testing"
    vectorization(dataset_path,ipt_vocab_path,opt_vocab_path,dsc_path,file_name)
    