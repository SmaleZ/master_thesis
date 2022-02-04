import csv
import  matplotlib.pyplot as plt

loss_file = open("train_loss.csv")
loss_file_reader = csv.reader(loss_file)
data = list(loss_file_reader)
losses = list()
time_steps = list()
for i in range(len(data)):
    time_steps.append(i)
    losses.append(data[i][0])

plt.plot(time_steps, losses)
plt.show()