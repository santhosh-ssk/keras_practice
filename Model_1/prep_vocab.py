import csv,collections
def prepare_vocab(ipt_file_path,max_count,data_dir,file_name):
    print("Preparation of Vocabulary for file:",file_name,"Initiated.....")
    vocab_tokens=collections.Counter()
    
    try:
        with open(ipt_file_path,'r') as ipt_file:
            reader=csv.reader(ipt_file)
            for text in reader:
                if len(text)==0:
                    continue
                text="".join(text)
                #print(text.split(),"\n")
                text=text.strip().lower()
                for token in text.split():
                    vocab_tokens[token]+=1
        if max_count==None:
                max_count=len(vocab_tokens)
        vocab_tokens=[[token] for token,count in vocab_tokens.most_common(max_count)]
        #print(vocab_tokens)
        with open(data_dir+"/"+file_name+".csv","w") as vocab_file:
            writer=csv.writer(vocab_file)
            writer.writerows(vocab_tokens)
        print("Total No of tokens in Vocabulary are ",len(vocab_tokens),"\n")
    except:
        return "Error in opening a file"

if __name__=="__main__":
    ipt_file_path="/home/santhosh/resumes_folder/keras/Model 1/data/ipt_lines.csv"
    data_dir="/home/santhosh/resumes_folder/keras/Model 1/data"
    file_name="ipt_vocab"
    prepare_vocab(ipt_file_path,None,data_dir,file_name)
    
    ipt_file_path="/home/santhosh/resumes_folder/keras/Model 1/data/opt_lines.csv"
    data_dir="/home/santhosh/resumes_folder/keras/Model 1/data"
    file_name="opt_vocab"
    prepare_vocab(ipt_file_path,None,data_dir,file_name)
    
