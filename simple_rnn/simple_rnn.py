#	program to find next character given a sequence of 10 characters using 
#	simple rnn by Santhosh on 19-7-18

#preparation of dataset
file=open('alice.txt')
lines=list()
for line in file:
	line=line.strip().lower()
	line.encode("ascii","ignore")
	if len(line)==0:
		continue
	lines.append(line)
file.close()
text=" ".join(lines)

#generating a characters vocabulary,
chars=set([c for c in text]) 
nb_chars=len(chars)
char2index=	dict((c,i) for i,c in enumerate(chars))
index2chars=	dict((i,c) for i,c in enumerate(chars))
print(nb_chars)

SEQLEN = 10
STEP = 1
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
	input_chars.append(text[i:i + SEQLEN])
	label_chars.append(text[i + SEQLEN])

#preparing training data
import numpy as numpy
x=np.array((len(input_chars),SEQLEN,nb_chars),dtype=np.bool)
y=np.array((len(input_chars),nb_chars),dtype=np.bool)
for i,input_char in enumerate(input_chars):
	for j,ch in enumerate(input_char):
		x[i,j,char2index[ch]]=1
	y[i,char2index[label_chars[i]]]=1
