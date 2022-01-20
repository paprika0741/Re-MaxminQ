
import numpy as np
import matplotlib.pyplot as plt
# 输出是10，000 每次10个平均值，图像间隔100_000
def compute (fname ):
    sum = 0
    with open(fname,"r") as f:
        line = f.readlines()
    for j in range(100_000):
        i = line[j]
        l = i.split()
        if l==[]:
            break
        if l[0]=='=':
            break
        if(int(l[0]) == 999     ):
            sum += int(l[1])
    return sum

def compute_epoch_step(fname ):
    sum = [0]*1000
    with open(fname,"r") as f:
        line = f.readlines()
    for j in range(100_000):
        i = line[j]
        l = i.split()
        if l==[]:
            break
        if l[0]=='=':
            break
        sum[int(l[0])%1000] += float(l[2])
        
    return [i/100 for i   in sum]
fname = "run_100_MaxminQsigma_10_batch8_net2_set1.txt"
sum = compute_epoch_step(fname)
print(fname)
print(sum)


# 输出是10，000 每次10个平均值，图像间隔100_000
def plot_average_steps (filename_1, filename_2, filename_3,filename_4 ,color1, color2 ,color3,color4 ):
    plt.figure(figsize=(20,10))
    epoch =  np.arange(1000)
    epoch = [i+1 for i in epoch]
    
    steps = compute_epoch_step(filename_1)
    label = "Q-Learning"
    plt.plot(epoch, steps, color1, label =label  )

    steps = compute_epoch_step(filename_2)
    label = "Double Q-Learning"
    plt.plot(epoch, steps, color2, label =label  )
    plt.legend(fontsize="large",loc='upper left')

 
#     plt.savefig(label + '.png')
    
