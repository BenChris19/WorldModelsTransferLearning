import numpy as np
import matplotlib.pyplot as plt

rec_losses = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"loss"+".npy")
rec_losses_bi = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"lossBipedal"+".npy")
rec_losses_pre = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"lossBipedalPretrained"+".npy")
rec_losses_two = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"lossBipedalTwoLayers"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = rec_losses
x_1 = np.arange(1,10,9/len(y_1))

y_2 = rec_losses_bi
x_2 = np.arange(1,10,9/len(y_2))

y_3 = rec_losses_pre
x_3 = np.arange(1,10,9/len(y_3))

y_4 = rec_losses_two
x_4 = np.arange(1,10,9/len(y_4))

plt.title("Reconstruction Loss vs epochs CVAE")

plt.plot(x_1, y_1,label='Trained from scratch') 
plt.plot(x_2,y_2,label='Freezing Encoder')  
plt.plot(x_3, y_3,label='Pretrained model')
plt.plot(x_4,y_4,label='Freezing 2 Encoder layers')

plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#########################################
kl_losses = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"loss"+".npy")
kl_losses_bi = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"lossBipedal"+".npy")
kl_losses_pre = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"lossBipedalPretrained"+".npy")
kl_losses_two = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"lossBipedalTwoLayers"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = kl_losses
x_1 = np.arange(1,10,9/len(y_1))

y_2 = kl_losses_bi
x_2 = np.arange(1,10,9/len(y_2))

y_3 = kl_losses_pre
x_3 = np.arange(1,10,9/len(y_3))

y_4 = kl_losses_two
x_4 = np.arange(1,10,9/len(y_4))

plt.title("KL Loss vs epochs CVAE")

plt.plot(x_1, y_1,label='Trained from scratch') 
plt.plot(x_2,y_2,label='Freezing Encoder')  
plt.plot(x_3, y_3,label='Pretrained model')
plt.plot(x_4,y_4,label='Freezing 2 Encoder layers')

plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#########################################
mem_losses = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"mem"+"lossMemoryScratch"+".npy")
mem_losses_pretrained = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"mem"+"lossMemoryPretrained"+".npy")
mem_losses_FreezeLSTM = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"mem"+"lossMemoryFreezeLSTM"+".npy")
mem_losses_FreezeFC = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"mem"+"lossMemoryFreezeFC"+".npy")


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = mem_losses
x_1 = np.arange(1,4000,3999/len(y_1))

y_2 = mem_losses_pretrained
x_2 = np.arange(1,2000,1999/len(y_2))

y_3 = mem_losses_FreezeLSTM
x_3 = np.arange(1,2000,1999/len(y_3))

y_4 = mem_losses_FreezeFC
x_4 = np.arange(1,2000,1999/len(y_4))

plt.title("Loss vs Timesteps RNN-MDN")

plt.plot(x_1, y_1,label='Trained from scratch') 
plt.plot(x_2, y_2,label='Pretrained Model') 
plt.plot(x_3, y_3,label='Frozen LSTM') 
plt.plot(x_4, y_4,label='Frozen MDN') 
plt.xlabel('Timesteps')
plt.ylabel('Epochs')

plt.legend()
plt.show()
#########################
best_bipedal = []
avg_bipedal = []
worst_bipedal = []
for i in range(5):
    best_bipedal.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"bestBipedalFinal"+".npy"+str(i)+".npy"))
    avg_bipedal.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgBipedalFinal"+".npy"+str(i)+".npy"))
    worst_bipedal.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstBipedalFinal"+".npy"+str(i)+".npy"))

frozen2_car = np.reshape(best_bipedal, (-1, 20))
no_avg_car = np.reshape(avg_bipedal, (-1, 20))
full_frozen_car = np.reshape(worst_bipedal, (-1, 20))

bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)

plt.tight_layout()
plt.title("BipedalWalker-v2 performance across 20 generations \n Trained from scratch")
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.show()
#########################################

#########################################

best_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"bestCar"+".npy")
avg_car= np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgCar"+".npy")
worst_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstCar"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = best_car
x_1 = np.arange(1,11,10/len(y_1))

y_2 = avg_car
x_2 = np.arange(1,11,10/len(y_2))

y_3 = worst_car
x_3 = np.arange(1,11,10/len(y_3))


plt.title("CarRacing-v0 performance across 10 generations training from scratch")

plt.plot(x_1, y_1,label='Best individual') 
plt.plot(x_2, y_2,label='Population mean') 
plt.plot(x_3, y_3,label='Worst individual') 


plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.legend()
plt.show()

best_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenFullbestCar"+".npy")
avg_car= np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenavgFullCar"+".npy")
worst_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenworstFullCar"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = best_car
x_1 = np.arange(1,11,10/len(y_1))

y_2 = avg_car
x_2 = np.arange(1,11,10/len(y_2))

y_3 = worst_car
x_3 = np.arange(1,11,10/len(y_3))


plt.title("CarRacing-v0 performance across 10 generations training from frozen encoder")

plt.plot(x_1, y_1,label='Best individual') 
plt.plot(x_2, y_2,label='Population mean') 
plt.plot(x_3, y_3,label='Worst individual') 


plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.legend()
plt.show()

best_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2FullbestCar"+".npy")
avg_car= np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2avgFullCar"+".npy")
worst_car = np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2worstFullCar"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

y_1 = best_car
x_1 = np.arange(1,11,10/len(y_1))

y_2 = avg_car
x_2 = np.arange(1,11,10/len(y_2))

y_3 = worst_car
x_3 = np.arange(1,11,10/len(y_3))


plt.title("CarRacing-v0 performance across 10 generations training from freezing 2 encoders")

plt.plot(x_1, y_1,label='Best individual') 
plt.plot(x_2, y_2,label='Population mean') 
plt.plot(x_3, y_3,label='Worst individual') 


plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.legend()
plt.show()

###########################################################

###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"bestLunarFinal"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgLunarFinal"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstLunarFinal"+".npy"+str(i)+".npy"))

frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))

bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)

plt.tight_layout()
plt.title("LunarLander-v2 performance across 20 generations \n Trained from TL")
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.show()

###########################################################

###########################################################

frozen2_car = []
no_avg_car = []


for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstTransferavgLunarFinal"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgLunarFinal"+".npy"+str(i)+".npy"))

frozen2_bi = np.reshape(frozen2_car, (-1, 20))
no_bi = np.reshape(no_avg_car, (-1, 20))


bpl = plt.boxplot(frozen2_bi, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_bi, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')


x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Mean population scratch')
plt.plot([], c='#2C7BB6', label='Mean population TL')

plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title("LunarLander-v2 performance across 20 generations\n using and not using transfer learning")
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.show()


###########################################################

frozen2_bi = []
no_bi = []

for i in range(5):
    frozen2_bi.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgTransBipedalFinal"+".npy"+str(i)+".npy"))
    no_bi.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgBipedalFinal"+".npy"+str(i)+".npy"))

frozen2_bi = np.reshape(frozen2_bi, (-1, 20))
no_bi = np.reshape(no_bi, (-1, 20))


bpl = plt.boxplot(frozen2_bi, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_bi, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')


x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Mean population scratch')
plt.plot([], c='#2C7BB6', label='Mean population TL')

plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title("BipedalWalker-v2 performance using and not using Transfer Learning")
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.show()


###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"bestBipedalFinal"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgBipedalFinal"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstBipedalFinal"+".npy"+str(i)+".npy"))

frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))

bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)

plt.tight_layout()
plt.title("BipedalWalker-v2 performance across 20 generations \n Trained from TL")
plt.xlabel('Generations')
plt.ylabel('Reward')
plt.show()

######################
frozen2_bi=np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"timeTaken"+".npy")
time_frozen =np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"timeTakenWithTransfer"+".npy")

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


y_1 = frozen2_bi
x_1 = [i for i in range(1,11)]

y_2 = time_frozen
x_2 = [i for i in range(1,11)]



plt.title("Training Vision Model Time vs Epochs")

plt.plot(x_1, y_1, label='Time taken scratch') 
plt.plot(x_2, y_2, label='Time taken transfer') 



plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Epochs')
plt.ylabel('Hours')
plt.legend()
plt.show()

###########################################################

###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2BestCarNormal20"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2avgCarNormal20"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2worstCarNormal20"+".npy"+str(i)+".npy"))


frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))


bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best Individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst Individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title('CarRacing-v0 trained with 2 frozen layers across \n 20 generations')
plt.ylabel('Reward')
plt.xlabel('Generations')
plt.show()


###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenFullBestCarNormal20"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenFullavgCarNormal20"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenFullworstCarNormal20"+".npy"+str(i)+".npy"))


frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))


bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best Individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst Individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title('CarRacing-v0 trained with frozen encoder across \n 20 generations')
plt.ylabel('Reward')
plt.xlabel('Generations')
plt.show()

###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"BestCarNormal20"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgCarNormal20"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstCarNormal20"+".npy"+str(i)+".npy"))

frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))



bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='Best Individual')
plt.plot([], c='#2C7BB6', label='Mean population')
plt.plot([], c='#00FF00', label='Worst Individual')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title('CarRacing-v0 trained from scratch across \n 20 generations')
plt.ylabel('Reward')
plt.xlabel('Generations')
plt.show()

###########################################################

frozen2_car = []
no_avg_car = []
full_frozen_car = []

for i in range(5):
    frozen2_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"Frozen2avgCarNormal20"+".npy"+str(i)+".npy"))
    no_avg_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"avgCarNormal20"+".npy"+str(i)+".npy"))
    full_frozen_car.append(np.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"FrozenFullavgCarNormal20"+".npy"+str(i)+".npy"))


frozen2_car = np.reshape(frozen2_car, (-1, 20))
no_avg_car = np.reshape(no_avg_car, (-1, 20))
full_frozen_car = np.reshape(full_frozen_car, (-1, 20))



bpl = plt.boxplot(frozen2_car, positions=np.array(range(20))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(no_avg_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)
bps = plt.boxplot(full_frozen_car, positions=np.array(range(20))*2.0+0.4, sym='', widths=0.6)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, '#00FF00')

x_1 = [i+1 for i in range(20)]

plt.plot([], c='#D7191C', label='2 frozen layers')
plt.plot([], c='#2C7BB6', label='Scratch')
plt.plot([], c='#00FF00', label='Frozen encoder')
plt.legend()

plt.xticks(range(0, len(x_1) * 2, 2), x_1)
plt.tight_layout()
plt.title('CarRacing-v0 trained on different training techniques \n across 20 generations')
plt.ylabel('Reward')
plt.xlabel('Generations')
plt.show()