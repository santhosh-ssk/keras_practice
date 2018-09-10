from __future__ import print_function
import tensorflow as tf
from keras.models import load_model,Model
from keras.layers import Input,Activation,Dense,GRU,Embedding,Bidirectional,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score

import csv,os,sys,random
from vectorization import Text_Vectorize

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def acc(y_true,y_pred):
    #print(y_true,y_pred)
    #recall=tf.metrics.recall(y_true,y_pred)[0]
    prec=tf.metrics.precision(y_true,y_pred)[0]
    """
    if (recall+prec!=0):
        f1=2*recall*prec/(recall+prec)
    else:
        f1=0
    return f1
    """
    return prec



base_dir="/home/santhosh/resumes_folder/keras/taxonomy_categorization_using_GRU_encoder_decoder/"
base_path=base_dir+"data/"
#Loading padded Training input dataset
print("\nLoading padded Training input dataset")
training_ipt_vector_file=base_path+"vectorized_data/padded_train_ipt.csv"
training_ipt_index2tag_file=base_path+"ipt_index2tag.csv"
training_ipt_vector=Text_Vectorize()
training_ipt_vector.load_padded_dataset(training_ipt_vector_file)
training_ipt_vector.load_index_file(training_ipt_index2tag_file)

#Loading padded Training output dataset
print("\nLoading padded Training output dataset")
training_opt_vector_file=base_path+"vectorized_data/padded_train_opt.csv"
training_opt_index2tag_file=base_path+"opt_index2tag.csv"
training_opt_vector=Text_Vectorize()
training_opt_vector.load_padded_dataset(training_opt_vector_file)
training_opt_vector.load_index_file(training_opt_index2tag_file)


#Loading padded Testing intput dataset
print("\nLoading padded Testing intput dataset")
testing_ipt_vector_file=base_path+"vectorized_data/padded_test_ipt.csv"
testing_ipt_vector=Text_Vectorize()
testing_ipt_vector.load_padded_dataset(testing_ipt_vector_file)

#Loading padded Testing output dataset
print("\nLoading padded Testing output dataset")
testing_opt_vector_file=base_path+"vectorized_data/padded_test_opt.csv"
testing_opt_vector=Text_Vectorize()
testing_opt_vector.load_padded_dataset(testing_opt_vector_file)


#preparing encoder_input_data,decoder_input_data
print('preparing encoder_input_data,decoder_input_data,decoder_target_data')
encoder_input_data=training_ipt_vector.padded_dataset
decoder_input_data=training_opt_vector.padded_dataset
decoder_target_data=np.zeros(decoder_input_data.shape)

decoder_target_data[:,:-1]=decoder_input_data[:,1:]
decoder_target_data=to_categorical(decoder_target_data,num_classes=training_opt_vector.max_features)

encoder_input_max_features=training_ipt_vector.max_features
decoder_input_max_features=training_opt_vector.max_features
print('Encoder input  shape: ',encoder_input_data.shape)
print('Decoder input  shape: ',decoder_input_data.shape)
print('Decoder target shape:',decoder_target_data.shape)

#preparing test encoder_input_data,decoder_input_data
print('\npreparing  test encoder_input_data,decoder_input_data,decoder_target_data')
test_encoder_input_data=testing_ipt_vector.padded_dataset
test_decoder_input_data=testing_opt_vector.padded_dataset
test_decoder_target_data=np.zeros(test_decoder_input_data.shape)

test_decoder_target_data[:,:-1]=test_decoder_input_data[:,1:]
test_decoder_target_data=to_categorical(test_decoder_target_data,num_classes=training_opt_vector.max_features)

print('test Encoder input  shape: ',test_encoder_input_data.shape)
print('test Decoder input  shape: ',test_decoder_input_data.shape)
print('test Decoder target shape:',test_decoder_target_data.shape)


latent_dim=128
batch_size=512

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
embed_encoded_inputs = Embedding(training_ipt_vector.max_features,latent_dim)(encoder_inputs)
encoder = Bidirectional(GRU(latent_dim, return_state=True))
encoder_outputs, forward_state_h,backward_state_h = encoder(embed_encoded_inputs)
state_h=Concatenate()([forward_state_h,backward_state_h])

# Set up the decoder, using state_h as initial state.
decoder_inputs = Input(shape=(None,))
embed_decoder_inputs=Embedding(training_opt_vector.max_features,latent_dim)(decoder_inputs)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_gru = GRU(latent_dim*2, return_sequences=True,return_state=True)
decoder_outputs,_ = decoder_gru(embed_decoder_inputs, initial_state=state_h)

decoder_dense = Dense(training_opt_vector.max_features, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

#inferences
#define encoder model
encoder_model=Model(encoder_inputs,state_h)
print('\n\nEncoder model')
encoder_model.summary()

#define decoder model
decoder_state_input=Input(shape=(latent_dim*2,))
embed_decoder_final_inputs=Embedding(training_opt_vector.max_features,latent_dim)(decoder_inputs)
decoder_outputs2,decoder_state_h=decoder_gru(embed_decoder_final_inputs,initial_state=[decoder_state_input])
decoder_outputs2=decoder_dense(decoder_outputs2)

# sampling model will take encoder states and decoder_input(seed initially) and output the predictions(taxonomy word index) We dont care about decoder_states2
decoder_model1 = Model(
    [decoder_inputs] + [decoder_state_input],
    [decoder_outputs2] +[decoder_state_h])
print('\n\nDecoder model')
decoder_model1.summary()

def decode_sequence(input_seq):
    input_sentence=" ".join([training_ipt_vector.index2tag[int(index)] for index in input_seq if int(index)!=0])
    #input_seq=input_seq.reshape((1,))
    #print('shape: ',input_seq.shape)
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = training_opt_vector.tag2index['<sol>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h = decoder_model1.predict(
        [target_seq] + [states_value])

        # Sample a token
        #print(np.argmax(np.sum(output_tokens[0,1:, :],axis=1))+1)
        #sampled_token_index = np.argmax(np.sum(output_tokens[0,1:, :],axis=1))+1
        #print(output_tokens.shape)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = training_opt_vector.index2tag[sampled_token_index]
        decoded_sentence += ' '+sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '<eol>' or
            len(decoded_sentence.split()) > training_opt_vector.max_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = h
    return input_sentence,decoded_sentence




def estimate_score():
    scores=model.evaluate([test_encoder_input_data,test_decoder_input_data],test_decoder_target_data,batch_size=256,verbose=1)
    print("\n%s: %.2f%%\n\n" % (model.metrics_names[1], scores[1]*100))

def test_samples(write_output=False,iteration=0):
    print('Testing from Training dataset\n')
    if write_output:
        with open(log_file,'a') as file_data:
            file_data.write('-'*50+"\n")
            file_data.write('\n\nIteration: '+str(iteration)+'\n')
            file_data.write('Testing from Training dataset\n')
    for i in range(5):
        index=random.randint(0,len(encoder_input_data))
        x,y=decode_sequence(encoder_input_data[index])
        y_true=" ".join([training_opt_vector.index2tag[i] for i in decoder_input_data[index] if i>2])
        print(x,"--->",y,'(',y_true,')')
        if write_output:
            with open(log_file,'a') as file_data:
                file_data.write(x+"--->"+y+'( '+y_true+' )'+'\n')
    print('Testing from Testing dataset\n')
    if write_output:
        with open(log_file,'a') as file_data:
            file_data.write('*'*25+"\n")
            file_data.write('Testing from Testing dataset\n')

    for i in range(5):
        index=random.randint(0,len(test_encoder_input_data))
        x,y=decode_sequence(test_encoder_input_data[index])
        y_true=" ".join([training_opt_vector.index2tag[i] for i in test_decoder_input_data[index] if i>2])
        print(x,"--->",y,'(',y_true,')')
        print(x,"--->",y,'(',y_true,')')
        if write_output:
            with open(log_file,'a') as file_data:
                file_data.write(x+"--->"+y+'( '+y_true+' )'+'\n')

iteration_file=base_dir+"iteration_bidir.txt"
model_file=base_dir+"__bidir_model__/"
checkpoint_file=base_dir+"__checkpoints__/bidir_model_checkpoint.hdf5"
log_file=base_dir+"output_logs/bidir_model_logs.txt"

checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

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
        model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
            verbose=1,
            validation_split=0.1,
            batch_size=400,
            epochs=5,
            callbacks=callbacks_list)
        estimate_score()
        test_samples(True,iteration+1)

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