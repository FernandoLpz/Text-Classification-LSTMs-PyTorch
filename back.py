# def RNN():
#     inputs = Input(name='inputs', shape=[max_len])

		# max_words = vocab_len
		# 50 = hidden_size
		# max_len = sequence_len
		
#     layer = Embedding(max_words,50,input_length=max_len)(inputs)
#     layer = LSTM(64)(layer)
#     layer = Dense(257,name='FC1')(layer)
#     layer = Activation('relu')(layer)
#     layer = Dropout(0.5)(layer)
#     layer = Dense(1,name='out_layer')(layer)
#     layer = Activation('sigmoid')(layer)
#     model = Model(inputs=inputs,outputs=layer)
#     return model
    
# model = RNN()
# model.summary()
# model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

# model.fit(sequences_matrix,Y_train,batch_size=64,epochs=30,
#           validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
          
# test_sequences = tok.texts_to_sequences(X_test)
# test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# accr = model.evaluate(test_sequences_matrix,Y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))