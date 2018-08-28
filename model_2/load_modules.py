import csv

def load_vectorized_file(file_path):    
    try:
        input_data=list()
        input_max_len=0
        output_data=list()
        output_max_len=0
        with open(file_path,"r") as file_data:
            reader=csv.reader(file_data)
            for row in reader:
                if len(row)!=2:
                    continue
                input_text,output_text=row
                input_text=list(map(int,input_text.split()))
                output_text=list(map(float,output_text.split()))
                if len(input_text) > input_max_len:
                    input_max_len=len(input_text)
                
                if len(output_text) > output_max_len:
                    output_max_len=len(output_text)
                input_data.append(input_text)
                output_data.append(output_text)
        return input_data,len(input_data),input_max_len,output_data,len(output_data),output_max_len
    except:
        return None,None,None,None,None,None

def load_token2index(file_path):
    try:
        token2index=dict()
        with open(file_path,"r") as file_data:
            reader=csv.reader(file_data)
            for row in reader:
                if len(row)!=2:
                    continue
                key,val=row
                val=int(val)
                token2index[key]=val
            return token2index,len(token2index)
    except:
        return None,None

def load_index2token(file_path):
    try:
        index2token=dict()
        with open(file_path,"r") as file_data:
            reader=csv.reader(file_data)
            for row in reader:
                if len(row)!=2:
                    continue
                key,val=row
                key=int(key)
                index2token[key]=val
            return index2token,len(index2token)
    except:
        return None,None

def load_train_file(file_path):
    try:
        input_data=list()
        with open(file_path,"r") as file_data:
            reader=csv.reader(file_data)
            for row in reader:
                if len(row)!=2:
                    continue
                input_text,_=row
                input_data.append(input_text)
            return input_data,len(input_data) 
    except:
        return None,None
