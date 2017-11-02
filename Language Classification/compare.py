from utils import read_file

train_x = read_file('dataset/train_set_x.csv')
kaggle_x = read_file('dataset/kaggle_set_x.csv')
predicted_y = read_file('dataset/predicted_kaggle_set_y.csv')
actual_y = read_file('dataset/train_set_y.csv')

print("Predicted,Actual,Charcters,Body")

for i, instance in enumerate(train_x):
    if predicted_y[i] is not None and predicted_y[i] != actual_y[i]:
        fmt = ('{:<3}{:<3}{:<42}{:<100}'.format(predicted_y[i], actual_y[i], str(kaggle_x[i]), str(train_x[i])))
        print(fmt)
