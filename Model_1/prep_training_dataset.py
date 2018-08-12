import collections,os,csv

def tokenizer(file_path):
    input_lines=list()
    output_lines=list()
    try:
        with open(file_path,'r') as csv_file:
            reader=csv.reader(csv_file)
            for input_text,output_text in reader:
                input_text =input_text.strip().lower()
                output_text=output_text.strip().lower()
                input_lines.append(input_text)
                output_lines.append(output_text)

            return input_lines,output_lines
        return None,None    
    except:
        return None,None 

def prepare_dataset(dir_path, no_files_to_be_cov,data_dir,max_lines):
    print("Started Preparing training set")
    files=os.listdir(dir_path)
    files.sort()
    if no_files_to_be_cov!=None:
        files=files[:no_files_to_be_cov]
    count=0
    lines=list()
    for file_name in files:
        file_path=dir_path +'/'+ file_name
        try:
            ipt_lines,opt_lines=tokenizer(file_path)
            lines=list(zip(ipt_lines,opt_lines))
            if max_lines!=None:
                lines=lines[:max_lines]
            #print(lines)
            with open(data_dir+'/dataset.csv','a') as dataset_file:
                writer=csv.writer(dataset_file)
                writer.writerows(lines)
            count+=1
        except:
            print("Error in processing ",file_name)
    print("Total no input files:",count,"used to generate training set")


if __name__ == "__main__":
    dir_path="/home/santhosh/resumes_folder/keras/Model_1/dataset"
    data_path="/home/santhosh/resumes_folder/keras/Model_1/data"
    prepare_dataset(dir_path,None,data_path,40)    

