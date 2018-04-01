import Predict
import data
import os
import numpy as np

data_dir = 'Datasets/image_set_train'
data_list = os.listdir(data_dir)
train_data, validation_data, max_len = data.read_data(data_dir, data_list.__len__() - 1, 70, verbose=1)

train_input, train_target, train_lens, num_train = train_data
validation_input, validation_target, validation_lens, num_validation = validation_data

train_results, train_confs = Predict.predict_images(train_input)
validation_results, validation_confs = Predict.predict_images(validation_input)

train_correct_or_wrong = np.array([int(train_target[i] == train_results[i]) for i in range(num_train)])
validation_correct_or_wrong = np.array([int(validation_target[i] == validation_results[i]) for i in range(num_validation)])

train_num_correct = np.sum(train_correct_or_wrong)
train_num_incorrect = num_train - train_num_correct

validation_num_correct = np.sum(validation_correct_or_wrong)
validation_num_incorrect = num_validation - validation_num_correct

train_mean_conf_correct = np.sum(np.multiply(train_confs, train_correct_or_wrong))/train_num_correct
train_mean_conf_incorrect = np.sum(np.multiply(train_confs, 1-train_correct_or_wrong))/train_num_incorrect

validation_mean_conf_correct = np.sum(np.multiply(validation_confs, validation_correct_or_wrong))/validation_num_correct
validation_mean_conf_incorrect = np.sum(np.multiply(validation_confs, 1-validation_correct_or_wrong))/validation_num_incorrect

print("FOR TRAIN DATA:\n")
print("Accuracy: " + str(int(np.sum(train_correct_or_wrong) * 100 / num_train)) + "%\n")
print("mean confidence for correct predictions: " + str(int(train_mean_conf_correct * 100)) + "%\n")
print("mean confidence for incorrect predictions: " + str(int(train_mean_conf_incorrect * 100)) + "%\n\n")
print("FOR VALIDATION DATA:\n")
print("Accuracy: " + str(int(np.sum(validation_correct_or_wrong) * 100 / num_validation)) + "%\n")
print("mean confidence for correct predictions: " + str(int(validation_mean_conf_correct * 100)) + "%\n")
print("mean confidence for incorrect predictions: " + str(int(validation_mean_conf_incorrect * 100)) + "%\n\n")