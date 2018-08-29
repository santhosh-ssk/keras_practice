from __future__ import print_function
import tensorflow as tf
from keras.models import Sequential,load_model,Model
from keras.layers import Activation,Dense,LSTM,Dropout,RepeatVector,Bidirectional,Input,Reshape
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score

import csv,os,sys,random
from vectorization import Text_Vectorize

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
base_path="/home/santhosh/resumes_folder/keras/model_2/data/"

#Loading padded Training input dataset
print("\nLoading padded Training input dataset")
training_ipt_vector_file=base_path+"processed_data/padded_train_ipt.csv"
training_ipt_index2tag_file=base_path+"processed_data/ipt_index2tag.csv"
training_ipt_vector=Text_Vectorize()
training_ipt_vector.load_padded_dataset(training_ipt_vector_file)
training_ipt_vector.load_index_file(training_ipt_index2tag_file)
#Loading padded Training output dataset
print("\nLoading padded Training output dataset")
training_opt_vector_file=base_path+"processed_data/padded_train_opt.csv"
training_opt_index2tag_file=base_path+"processed_data/opt_index2tag.csv"
training_opt_vector=Text_Vectorize()
training_opt_vector.load_padded_dataset(training_opt_vector_file)
training_opt_vector.load_index_file(training_opt_index2tag_file)


#Loading padded Testing intput dataset
print("\nLoading padded Testing intput dataset")
testing_ipt_vector_file=base_path+"processed_data/padded_test_ipt.csv"
testing_ipt_vector=Text_Vectorize()
testing_ipt_vector.load_padded_dataset(testing_ipt_vector_file)

#Loading padded Testing output dataset
print("\nLoading padded Testing output dataset")
testing_opt_vector_file=base_path+"processed_data/padded_test_opt.csv"
testing_opt_vector=Text_Vectorize()
testing_opt_vector.load_padded_dataset(testing_opt_vector_file)

#preparing encoder_input_data,decoder_input_data,decoder_target_data
print('preparing encoder_input_data,decoder_input_data,decoder_target_data')
encoder_input_data=training_ipt_vector.padded_dataset
decoder_input_data=training_opt_vector.padded_dataset
decoder_target_data=np.zeros(decoder_input_data.shape)
decoder_target_data[:,:-1]=decoder_input_data[:,1:]
decoder_target_data=to_categorical(decoder_target_data,num_classes=training_opt_vector.max_features)

#preparing encoder_test_input_data,decoder_test_input_data,decoder_test_target_data
print('preparing encoder_input_data,decoder_input_data,decoder_target_data')
encoder_test_input_data=testing_ipt_vector.padded_dataset
decoder_test_input_data=testing_opt_vector.padded_dataset
decoder_test_target_data=np.zeros(decoder_test_input_data.shape)
decoder_test_target_data[:,:-1]=decoder_test_input_data[:,1:]
decoder_test_target_data=to_categorical(decoder_test_target_data,num_classes=testing_opt_vector.max_features)


#define encoder_input model
encoder_inputs=Input(shape=(training_ipt_vector.max_len,))
embed_encoder_inputs=Embedding(training_ipt_vector.max_features,258)
embed_encoder_inputs=embed_encoder_inputs(encoder_inputs)
print(embed_encoder_inputs)
encoder=LSTM(hidden_size,return_state=True)
encoder_outputs, state_h, state_c = encoder(embed_encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

decoder_inputs = Input(shape=(training_opt_vector.max_len,))
embed_decoder=Embedding(training_opt_vector.max_features,128)
embed_decoder_inputs=embed_decoder(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embed_decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense = Dense(training_opt_vector.max_features, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


#Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=["accuracy"])
model.summary()


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    embed_decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
#print(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.summary()


def decode_sequence(input_seq):
    input_text=" ".join([training_ipt_vector.index2tag[int(tag)] for tag in input_seq])
    input_seq=input_seq.reshape((1,training_ipt_vector.max_len,))
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,training_opt_vector.max_len,))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = training_opt_vector.tag2index['<sol>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    word_count=0

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        #print(output_tokens.shape,np.argmax(output_tokens))
        # Sample a token
        #print(np.sum(output_tokens[0,1:-1, :],axis=1))
        sampled_token_index = np.argmax(np.sum(output_tokens[0,1:-1, :],axis=1))+1
        #print(sampled_token_index)
        sampled_word = training_opt_vector.index2tag[sampled_token_index]
        decoded_sentence += ' '+sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '<eol>' or
           word_count > training_opt_vector.max_len-2):
            stop_condition = True
        #print(target_seq)
        target_seq[0,0]=sampled_token_index
        # Update states
        states_value = [h, c]
        word_count+=1

    if('<eol>' in decoded_sentence):
        decoded_sentence=" ".join(decoded_sentence.split()[:-1])
    return [input_text,decoded_sentence]
    

def estimate_score():
    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate([encoder_test_input_data,decoder_test_input_data],decoder_test_target_data,verbose=0)
    print("%s: %.2f%%\n\n" % (model.metrics_names[1], scores[1]*100))
    print("Testing Samples\n"+"-"*50)
    return scores[1]*100


iteration_file="/home/santhosh/resumes_folder/keras/model_2/iteration.txt"
model_file="/home/santhosh/resumes_folder/keras/model_2/__model_data__/"
checkpoint_file=base_path+"checkpoints/checkpoint.hdf5"
log_file="output_logs/logs.txt"
iteration=0

try:
    file=open(iteration_file,'r')
    last_line=file.read().split('\n')[-2]
    print('file_data,',last_line)
    iteration=int(last_line.split(':')[1])
    #print(iteration)
    file.close()
    try:
        print('loading the weights')
        file_path=model_file+str(iteration)+".h5"
        model=load_model(file_path)
        estimate_score()
    except:
        print('no model file exist')        
except:
    print('no iteration file exist')

def test_sample_resume():
    index=random.randint(0,len(training_ipt_vector.padded_dataset))
    input_sample_resume,output_sample_summary=decode_sequence(encoder_input_data[index])
    print('-'*30,'Testing Sample resume','-'*30)
    print(input_sample_resume)
    print('-'*70)
    print('-'*30,'OUTPUT','-'*30)
    print(output_sample_summary)
    #write outputin a file
    with open(log_file,'a') as text_file:
        text_file.write('\nIteration :'+str(iteration+1)+' Output\n')
        text_file.write('Score: '+str(score)+'\n')
        text_file.write('-'*30+'Testing Sample resume'+'-'*30+'\n')
        text_file.write(input_sample_resume+'\n')
        text_file.write('-'*70+'\n'+'-'*30+'OUTPUT'+'-'*30+'\n')
        text_file.write(output_sample_summary+'\n'+'-'*70+'\n')


if __name__ =="__main__":
    arg=sys.argv[1]
    if arg=="train":
            while True:
                try:
                    try:
                        file=open(iteration_file,'r')
                        last_line=file.read().split('\n')[-2]
                        print('file_data,',last_line)
                        iteration=int(last_line.split(':')[1])
                        #print(iteration)
                        file.close()
                    except:
                        iteration=0        
                    print('Iteration:',iteration+1)
                    #training
                    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=5,
                    epochs=5,
                    validation_split=0.2)
                    score=estimate_score()
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
                    except:
                        break
                    
    elif arg=="test":
        estimate_score()
