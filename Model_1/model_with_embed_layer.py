from __future__ import print_function
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Activation,Dense,LSTM,Dropout,RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score

import csv,os
from load_modules import load_vectorized_file,load_token2index,load_index2token,load_train_file
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def acc(y_true,y_pred):
    recall=tf.metrics.recall(y_true,y_pred)[0]
    prec=tf.metrics.precision(y_true,y_pred)[0]
    if (recall+prec!=0):
        f1=2*recall*prec/(recall+prec)
    else:
        f1=0
    return f1

EMBED_SIZE = 128
batch_size=128
epochs=100
hidden_size=128

#loading vectorized Training Dataset
print("\nLoading vectorized Training dataset")
training_vector_file="/home/santhosh/resumes_folder/keras/Model_1/data/training_vextorized_ipt_opt.csv"
training_ipt_dataset,training_ipt_len,training_ipt_max_len,training_opt_dataset,training_opt_len,training_opt_max_len=load_vectorized_file(training_vector_file)
print("Total No of Training Records:",training_ipt_len)

#loading train dataset data
print("\nloading train dataset data")
train_data_file="/home/santhosh/resumes_folder/keras/Model_1/data/train_dataset.csv"
train_data,train_data_len=load_train_file(train_data_file)
print("Total No of Training Records:",train_data_len)

print("\nLoading vectorized Testing dataset")
testing_vector_file="/home/santhosh/resumes_folder/keras/Model_1/data/testing_vextorized_ipt_opt.csv"
testing_ipt_dataset,testing_ipt_len,testing_ipt_max_len,testing_opt_dataset,testing_opt_len,testing_opt_max_len=load_vectorized_file(testing_vector_file)
print("Total No of Testing Records:",testing_ipt_len)


print("\nLoading  Training Input token2index")
train_ipt_token2index="/home/santhosh/resumes_folder/keras/Model_1/data/train_ipt_token2index"
train_ipt_token2index,train_ipt_token2index_size=load_token2index(train_ipt_token2index)
print("Total No of index2token:",train_ipt_token2index_size)

print("\nLoading  Training Input index2token")
train_ipt_index2token="/home/santhosh/resumes_folder/keras/Model_1/data/train_ipt_index2token"
train_ipt_index2token,train_ipt_index2token_size=load_index2token(train_ipt_index2token)
print("Total No of index2token:",train_ipt_index2token_size)

print("\nLoading  Training output token2index")
train_opt_token2index="/home/santhosh/resumes_folder/keras/Model_1/data/train_opt_token2index"
train_opt_token2index,train_opt_token2index_size=load_token2index(train_opt_token2index)
print("Total No of index2token:",train_opt_token2index_size)

print("\nLoading  Training output index2token")
train_opt_index2token="/home/santhosh/resumes_folder/keras/Model_1/data/train_opt_index2token"
train_opt_index2token,train_opt_index2token_size=load_index2token(train_opt_index2token)
print("Total No of index2token:",train_opt_index2token_size)

encoder_input_data=np.zeros((training_ipt_len,training_ipt_max_len),dtype='float32')
decoder_target_data=np.zeros((training_opt_len,1,train_opt_token2index_size),dtype='float32')


print("Dim of Training encoder shape",encoder_input_data.shape)
print("Dim of Training decoder shape",decoder_target_data.shape)

for i in range(len(training_ipt_dataset)):
    for k,data in enumerate(training_ipt_dataset[i]):
        encoder_input_data[i][k]=data

for i in range(len(training_ipt_dataset)):
    for k in range(len(training_opt_dataset[i])):
        decoder_target_data[i][0][k]=training_opt_dataset[i][k]

testing_input_data=np.zeros((testing_ipt_len,training_ipt_max_len),dtype="float32")
testing_output_data=np.zeros((testing_opt_len,1,training_opt_max_len),dtype='float32')

print("Dim of Testing encoder shape",testing_input_data.shape)
print("Dim of Testing decoder shape",testing_output_data.shape)

for i in range(testing_ipt_len):
    for k,data in enumerate(testing_ipt_dataset[i]):
        testing_input_data[i][k]=data
for i in range(testing_ipt_len):
    for k,data in enumerate(testing_opt_dataset[i]):
        testing_output_data[i][0][k]=data



model = Sequential()
model.add(Embedding(train_ipt_index2token_size, EMBED_SIZE,input_length=training_ipt_max_len))
model.add(Dropout(0.05))
model.add(LSTM(hidden_size, return_sequences=False,dropout=0.2,))
model.add(RepeatVector(1))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(train_opt_index2token_size)))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=[acc])
model.summary()

def test_sample_resume():
    #Test model:
    print("-"*50)
    index=int(np.random.randint(training_ipt_len/40*0.8)*40)
    print(index)
    test_input=""
    test_output=""
    for i in range(50):
        encoded_input_sequence=encoder_input_data[index: index + 1]
        output_sequence=model.predict(encoded_input_sequence, verbose=0)[0]
        output_sequence = train_opt_index2token[np.argmax(output_sequence)]
        encoded_input_sequence=encoded_input_sequence.reshape(tuple(encoded_input_sequence.shape[1:]))
        for j in encoded_input_sequence:
            if train_ipt_index2token[j]=="UNK":
                continue
            test_input+=train_ipt_index2token[j]+' '
        test_input+='\n'    
        if output_sequence=="1":
            output_sequence=''
            for j in encoded_input_sequence:
                if train_ipt_index2token[j]=="UNK":
                    continue
                output_sequence+=train_ipt_index2token[j]+' '
            output_sequence+='\n'
        else:
            output_sequence=''
        test_output+=output_sequence
        index+=1
        
    print("-"*50)
    print(test_input)
    print("---OUTPUT-----")
    print(test_output)
    print(" "*50+"-"*50)    

def estimate_score():
    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate(testing_input_data,testing_output_data,verbose=0)
    print("%s: %.2f%%\n\n" % (model.metrics_names[1], scores[1]*100))
    print("Testing Samples\n"+"-"*50)
    

iteration_file="/home/santhosh/resumes_folder/keras/Model_1/__emberd_data__/emberd_iteration.txt"
model_file="/home/santhosh/resumes_folder/keras/Model_1/__emberd_data__/model_data/"
checkpoint_file="/home/santhosh/resumes_folder/keras/Model_1/__emberd_data__/checkpoints/checkpoint.hdf5"
iteration=0

try:
    file=open(iteration_file,'r')
    last_line=file.read().split('\n')[-2]
    print('file_data,',last_line)
    iteration=int(last_line.split(':')[1])
    #print(iteration)
    file.close()
    # load weights
    print('loading the weights')
    file_path=model_file+str(iteration)+".h5"
    model=load_model(file_path)
    estimate_score()
    test_sample_resume()
    

except:
    print('no file exist')

#Test model:

# checkpoint
checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

while True:
    try:
        print('Iteration:',iteration+1)
        #training
        model.fit(encoder_input_data,decoder_target_data ,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,callbacks=callbacks_list)
        estimate_score()
        test_sample_resume()
        # Save model
        file=open(iteration_file,'a')
        file.write('iteration:'+str(iteration+1)+'\n')
        file.close()
        iteration+=1
        file_path=model_file+str(iteration)+".h5"
        model.save(file_path)
    
    except:
        print("error in iteration",iteration)
        file=open(iteration_file,'r')
        last_line=file.read().split('\n')[-2]
        print('file_data,',last_line)
        iteration=int(last_line.split(':')[1])
        #print(iteration)
        file.close()
        # load weights
        print('loading the weights')
        file_path=model_file+str(iteration)+".h5"
        model=load_model(file_path)
        estimate_score()
        