import csv
import collections
def tokenizer(file_path):
    """
        resume_totenizer takes a csv file which contains rows of inputs and outputs 
        inputs:
            file_path:        source file in csv formate, which contains rows of inputs and outputs
    """
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

if  __name__ == "__main__":
    #test a sample file
    file_path="/home/santhosh/resumes_folder/keras/extract_summary_and_objective/resume_line_classification/data/Aarathi S.csv"
    input_lines,output_lines = tokenizer(file_path)
    print("Num of Input Lines:",len(input_lines))
    print("Num of Output Lines:",len(output_lines))
    