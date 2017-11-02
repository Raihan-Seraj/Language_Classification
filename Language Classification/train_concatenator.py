from utils import read_file

train_x = read_file('dataset/train_set_x.csv')
train_y = read_file('dataset/train_set_y.csv')

merged = open("dataset/merged_train.csv", "w")

for i, instance in enumerate(train_x):
    print(str(i) + "," + train_y[i] + "," + instance.lower(), file=merged)
