import collections,os,csv
from resume_tokenizer import tokenizer

def prepare_dataset(dir_path, no_files_to_be_cov,data_dir):

    print("Started Preparing training set")
    files=os.listdir(dir_path)
    files.sort()
    if no_files_to_be_cov!=None:
        files=files[:no_files_to_be_cov]
    
    input_lines=list()
    output_lines=list()
    for file_name in files:
        file_path=dir_path +'/'+ file_name
        try:
            ipt_lines,opt_lines=tokenizer(file_path)
            for line in ipt_lines:
                input_lines.append([line])
            for line in opt_lines:
                output_lines.append([line])
            
        except:
            print("Error in processing ",file_name)

    with open(data_dir+"/"+"ipt_lines.csv","w") as ipt_file:
        writer=csv.writer(ipt_file)
        writer.writerows(input_lines)
        
    with open(data_dir+"/"+"opt_lines.csv","w") as opt_file:
        writer=csv.writer(opt_file)
        writer.writerows(output_lines)
    
    print("Total no input files:",len(files),"used to generate training set")

if __name__ == "__main__":
    dir_path="/home/santhosh/resumes_folder/keras/Model 1/dataset"
    data_path="/home/santhosh/resumes_folder/keras/Model 1/data"
    prepare_dataset(dir_path,None,data_path)    

