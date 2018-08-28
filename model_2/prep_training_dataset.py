import collections,os,csv
from sklearn.model_selection import ShuffleSplit

def tokenizer(file_path,max_lines):
    input_lines=list()
    output_lines=list()
    try:
        with open(file_path,'r') as csv_file:
            reader=list(csv.reader(csv_file))
            if max_lines!=None:
                reader=reader[:max_lines]
            for input_text,output_text in reader:
                input_text =input_text.strip().lower()
                output_text=output_text.strip().lower()
                input_lines.append(input_text)
                if output_text=="1":
                    output_lines.append(input_text)
            input_lines="\n".join(input_lines)
            output_lines="sol "+"\n".join(output_lines)+" eol"
            return input_lines,output_lines
        return None,None    
    except:
        return None,None 

def prepare_dataset(src_path,dsc_dir, no_files_to_be_cov,max_lines):
    print("Started Preparing training set")
    files=os.listdir(src_path)
    files.sort()
    if no_files_to_be_cov!=None:
        files=files[:no_files_to_be_cov]
    count=0
    for file_name in files:
        file_path=src_path +'/'+ file_name
        try:
            ipt_resume,opt_summary=tokenizer(file_path,max_lines)
            with open(dsc_dir+'/dataset.csv','a') as dataset_file:
                writer=csv.writer(dataset_file)
                writer.writerow([ipt_resume,opt_summary])
            count+=1
        except:
            print("Error in processing ",file_name)
    print("Total no input files:",count,"used to generate training set")

def load_dataset(src_file_path):
    dataset=list()
    with open(src_file_path,'r') as dataset_file:
        reader=csv.reader(dataset_file)
        for row in reader:
            if len(row)!=2:
                continue
            dataset.append(row)
    return dataset,len(dataset)

def train_test_split(src_path,dsc_path,test_size=0.33):
    dataset,dataset_len=load_dataset(src_path)
    print("Total records:",dataset_len)
    train_test_suffle=ShuffleSplit(n_splits=1,test_size=test_size,random_state=42)
    train_index,test_index=list(train_test_suffle.split(dataset))[0]
    training_dataset_input=list()
    training_dataset_output=list()
    
    testing_dataset_input=list()
    testing_dataset_output=list()
    
    for index in train_index:
        training_dataset_input.append([dataset[index][0]])
        training_dataset_output.append([dataset[index][1]])

    for index in test_index:
        testing_dataset_input.append([dataset[index][0]])
        testing_dataset_output.append([dataset[index][1]])

    with open(dsc_path+'/training_dataset_input.csv','w') as datafile:
        writer=csv.writer(datafile)
        writer.writerows(training_dataset_input)

    with open(dsc_path+'/training_dataset_output.csv','w') as datafile:
        writer=csv.writer(datafile)
        writer.writerows(training_dataset_output)

    with open(dsc_path+'/testing_dataset_input.csv','w') as datafile:
        writer=csv.writer(datafile)
        writer.writerows(testing_dataset_input)
    
    with open(dsc_path+'/testing_dataset_output.csv','w') as datafile:
        writer=csv.writer(datafile)
        writer.writerows(testing_dataset_output)
    
    print("Total Training records:",len(train_index))
    print("Total Testing records:",len(test_index))


if __name__ == "__main__":
    src_path="/home/santhosh/resumes_folder/keras/model_2/__dataset__"
    dsc_path="/home/santhosh/resumes_folder/keras/model_2/data"
    prepare_dataset(src_path,dsc_path,None,50)    

    dataset_file='/home/santhosh/resumes_folder/keras/model_2/data/dataset.csv'
    train_test_split(dataset_file,dsc_path)

