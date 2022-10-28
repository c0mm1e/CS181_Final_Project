import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import random

# CNN model
model = Sequential()
# 1st Convolution Layer
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(196, 200, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# 2nd Convolution Layer
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# 3rd Convolution Layer
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# Flatten Convolution Layer to Fully Connected Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# Number of choices: 4
model.add(Dense(4, activation='softmax'))

opt = keras.optimizers.adam(lr=1e-4, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 270 = 30 * 9
current = 0
mini_group_size = 30
files_list = os.listdir("train_data")
files_num = len(files_list)
random.shuffle(files_list)
# Each loop, we train 30 files
while True:
    all_choices = [[] for _ in range(4)]

    # Load data and save it in the list corresponding to choice
    for file in files_list[current:current+mini_group_size]:
        data = list(np.load(os.path.join("train_data", file), allow_pickle=True))
        # data is a list of lists. Each list contains a single training example
        for pair in data:
            # The index of choice is the first element of the pair
            all_choices[np.argmax(pair[0])].append(pair)
    # Ensure that the length of each choice list is the same
    choices_dict = {"protect_base": all_choices[0],
                    "attack_unit": all_choices[1],
                    "attack_structure": all_choices[2],
                    "attack_start_point": all_choices[3]}
    lengths_list = []
    for choice in choices_dict:
        lengths_list.append(len(choices_dict[choice]))
    min_length = min(lengths_list)
    # New dataset
    for i in range(4):
        random.shuffle(all_choices[i])
        all_choices[i] = all_choices[i][:min_length]
    # Merge to get new train_data
    train_data = all_choices[0] + all_choices[1] + all_choices[2] + all_choices[3]
    # Shuffle again
    random.shuffle(train_data)

    test_size = mini_group_size // 2
    # Train
    x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 196, 200, 3)
    y_train = np.array([i[0] for i in train_data[:-test_size]])
    # Test
    x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 196, 200, 3)
    y_test = np.array([i[0] for i in train_data[-test_size:]])

    model.fit(x_train, y_train, batch_size=16, validation_data=(x_test, y_test),
                shuffle=True, verbose=1, epochs = 10)

    current += mini_group_size
    if current >= files_num:
        break
