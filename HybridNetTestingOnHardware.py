import pennylane as qml
import numpy as np
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from MNIST01Database import MNIST01Database
import HoppingDatabaseClass
from torch.utils.data import DataLoader
import imageio

n_qubits = 4
n_layers = 3
epochs = 10 # Inputting just 1 for test
n_samples = 10000
batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Npartitions = 10000
Nkx = 11
Nky = 11
Nkz = 2
Root = "NN_data_equal_v1.dat"

# Hopping Database
# layers = [484, 20, 20, 10, n_qubits]

# Train_dataset = HoppingDatabaseClass.HoppingsDataset(Root, Npartitions, Nkx, Nky, Nkz, train=True)
# Test_dataset = HoppingDatabaseClass.HoppingsDataset(Root, Npartitions, Nkx, Nky, Nkz, train=False)

# # Data loader
# train_loader = DataLoader(dataset=Train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_loader = DataLoader(dataset=Test_dataset, batch_size=1, shuffle=True, num_workers=0)

# MNIST dataset
layers = [784, 20, 20, 10, n_qubits]
train_loader, test_loader = MNIST01Database(n_samples, batch_size)
# train_loader, test_loader = HoppingDatabaseClass.HoppingsDatabase(batch_size)

dev = qml.device("default.qubit", wires=n_qubits)#, shots=1024)
dev2 = qml.device('qiskit.ibmq', wires=4, shots=100, backend='ibmq_quito', ibmqx_token="67eaa0874f668c69f22934cfb46076197c615df4c0aeebccb87d022c9b6a25694dd027eec6adec23b880c0b41eebbe1f086cc899ab6f0413c1cb7533bd6d1a71")
weight_shapes = {"weights": (n_layers, n_qubits)}

@qml.qnode(dev, diff_method = 'best')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

@qml.qnode(dev2, diff_method = 'best')
def hardwareqnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

@qml.qnode(dev, diff_method = 'best')
def stateqnode(weights):
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.state()

class HybridNet(nn.Module):
    def __init__(self, architecture):
        super(HybridNet, self).__init__()
        self.l1 = nn.Linear(architecture[0], architecture[1])
        self.l2 = nn.Linear(architecture[1], architecture[2])
        self.l3 = nn.Linear(architecture[2], architecture[3])
        self.l4 = nn.Linear(architecture[3], architecture[4])
        self.ql = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.l5 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        out = F.leaky_relu(self.l1(x))
        out = F.leaky_relu(self.l2(out))
        out = F.leaky_relu(self.l3(out))
        out = self.l4(out)
        out = self.ql(out)
        out = F.leaky_relu(self.l5(out))
        return out
    
class HardwareNet(nn.Module):
    def __init__(self, architecture):
        super(HardwareNet, self).__init__()
        self.l1 = nn.Linear(architecture[0], architecture[1])
        self.l2 = nn.Linear(architecture[1], architecture[2])
        self.l3 = nn.Linear(architecture[2], architecture[3])
        self.l4 = nn.Linear(architecture[3], architecture[4])
        self.ql = qml.qnn.TorchLayer(hardwareqnode, weight_shapes)
        self.l5 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        out = F.leaky_relu(self.l1(x))
        out = F.leaky_relu(self.l2(out))
        out = F.leaky_relu(self.l3(out))
        out = self.l4(out)
        out = self.ql(out)
        out = F.leaky_relu(self.l5(out))
        return out
    
def plot_probs(i, coefficients):
    fig = plt.figure(i, figsize=(10,5))
    # ax = fig.add_axes([0.1,0.1,0.9,0.9])
    ax = fig.add_subplot(111)
    binary = [bin(x)[2:].zfill(4) for x in range(16)]
    ax.set_ylabel('Probability (%)')
    ax.set_xlabel('Binary Output')
    ax.set_yticks(np.arange(0, 51, 5))
    ax.set_ylim([0, 50])
    ax.set_title('Outcome Distribution')
    ax.bar(binary,coefficients, color='#660099')
    plt.savefig(f'c:/Users/oscar/Documents/Masters ML/Neural Net/Probabilities/MNIST4QubitOutcomeDistribution{i}.png', transparent=False, facecolor='white')
    plt.close()
    
model = HybridNet(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
weights_holder = np.ones((epochs*len(train_loader),int(n_qubits*n_layers)))
q_circ_input_holder = np.ones((epochs*len(train_loader),4))
q_circ_target_holder = np.ones((epochs*len(train_loader),1)) 
probability_holder = np.ones((epochs*len(train_loader),int(n_qubits**2)))
loss_list = np.ones((epochs*len(train_loader)))
frames = []
model.train()
for epoch in range(epochs):
    start_time = time.time()
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_time = time.time()
        data, target = data.to(device), target.to(device)
        # data = data.to(device)
        # target = target.view(100).type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        # Forward pass 
        output = model(torch.flatten(data, 1))
        # Calculating loss
        chosen_sample = np.random.randint(0,100)
        q_circ_input_holder[batch_idx + epoch*len(train_loader)] = model.l4(F.leaky_relu(model.l3(F.leaky_relu(model.l2(F.leaky_relu(model.l1(torch.flatten(data, 1)[chosen_sample]))))))).detach().cpu().numpy()
        q_circ_target_holder[batch_idx + epoch*len(train_loader)] = target[chosen_sample].detach().cpu().numpy()
        for name, param in model.named_parameters():
            if param.requires_grad and name == 'ql.weights': 
                weights_holder[batch_idx+epoch*len(train_loader)] = param.view(int(n_qubits*n_layers)).detach().cpu().numpy()
        #         coefficients = np.square(np.abs(stateqnode(param).detach().cpu().numpy()))*100
        #         plot_probs(batch_idx+epoch*len(train_loader), coefficients)
        #         image = imageio.v2.imread(f'c:/Users/oscar/Documents/Masters ML/Neural Net/Probabilities/MNIST4QubitOutcomeDistribution{batch_idx+epoch*len(train_loader)}.png')
        #         frames.append(image)
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        # print('batch took: ', time.time()-batch_time, ' seconds')
        # total_loss.append(loss.item())
        loss_list[batch_idx+epoch*len(train_loader)] = loss.item() #sum(total_loss)/len(total_loss)
    print('Training [{:.0f}%]\tLoss: {:.4f}\t Time taken {:.0f}s'.format(100. * (epoch + 1) / epochs, loss_list[(epoch+1)*len(train_loader)-1], time.time()-start_time))

torch.save(model.state_dict(), f'c:/Users/oscar/Documents/Masters ML/Neural Net/Module_Params/BiTeIParams.pt')
hardware_model = HardwareNet(layers).to(device)
hardware_model.load_state_dict(torch.load(f'c:/Users/oscar/Documents/Masters ML/Neural Net/Module_Params/BiTeIParams.pt'))

# imageio.mimsave(f'c:/Users/oscar/Documents/Masters ML/Neural Net/Probabilities/MNIST4QubitBasicEntanglementProbabilities.gif', frames, fps=60)
plt.figure(1)
plt.plot(weights_holder[:,0], color='#4285F4', label='R0', linewidth='2')
plt.plot(weights_holder[:,1], color='#24A853', label='R1', linewidth='2')
plt.plot(weights_holder[:,2], color='#FBBC05', label='R2', linewidth='2')
plt.plot(weights_holder[:,3], color='#EA4335', label='R3', linewidth='2')
plt.title('Evolution of QC Weights During Training', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(0, 6, 7, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend()
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_First_Layer_Weights.png')

plt.figure(2)
plt.plot(weights_holder[:,4], color='#4285F4', label='R4', linewidth='2')
plt.plot(weights_holder[:,5], color='#24A853', label='R5', linewidth='2')
plt.plot(weights_holder[:,6], color='#FBBC05', label='R6', linewidth='2')
plt.plot(weights_holder[:,7], color='#EA4335', label='R7', linewidth='2')
plt.title('Evolution of QC Weights During Training', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(0, 6, 7, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend()
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Second_Layer_Weights.png')

plt.figure(3)
plt.plot(weights_holder[:,8], color='#4285F4', label='R8', linewidth='2')
plt.plot(weights_holder[:,9], color='#24A853', label='R9', linewidth='2')
plt.plot(weights_holder[:,10], color='#FBBC05', label='R10', linewidth='2')
plt.plot(weights_holder[:,11], color='#EA4335', label='R11', linewidth='2')
plt.title('Evolution of QC Weights During Training', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(0, 6, 7, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend()
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Third_Layer_Weights.png')

from matplotlib.lines import Line2D
c_data = np.genfromtxt('c:/Users/oscar/Documents/Masters ML/Neural Net/MNIST_classical_loss_list.txt', )
plt.figure(4)
plt.plot(loss_list, color='#660099', alpha=1)
plt.plot(c_data, color='#660099', alpha=0.5)
plt.title("Loss of the Quantum NN During Training", fontsize=16)
plt.yticks(np.linspace(0, 0.8, 5, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Loss of Network', fontsize=14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend([Line2D([0], [0], color='#660099', lw=4, alpha=1),
                Line2D([0], [0], color='#660099', lw=4, alpha=0.5)], ['QNN', 'CNN'], loc='upper right')
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Loss_list_Quantum.png')

plt.figure(5)
for ID, sample in enumerate(q_circ_input_holder):
    if q_circ_target_holder[ID] == 0:
        triv = plt.scatter(ID, sample[0], color='red', s=3)
    if q_circ_target_holder[ID] == 1:
        top = plt.scatter(ID, sample[0], color='blue', s=3)
plt.title('Angle Embedding of the First Qubit', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(-0.5, 1.5, 5, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend([Line2D([0], [0], marker='o', color='blue', markersize=5,ls='none'), Line2D([0], [0], marker='o', color='red', markersize=5, ls='none')], ['Topological', 'Trivial'])
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Angle_Embedding_First_Qubit.png')

plt.figure(6)
for ID, sample in enumerate(q_circ_input_holder):
    if q_circ_target_holder[ID] == 0:
        triv = plt.scatter(ID, sample[1], color='red', s=3)
    if q_circ_target_holder[ID] == 1:
        top = plt.scatter(ID, sample[1], color='blue', s=3)
plt.title('Angle Embedding of the Second Qubit', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(-0.5, 1.5, 5, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend([Line2D([0], [0], marker='o', color='blue', markersize=5,ls='none'), Line2D([0], [0], marker='o', color='red', markersize=5, ls='none')], ['Topological', 'Trivial'])
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Angle_Embedding_Second_Qubit.png')

plt.figure(7)
for ID, sample in enumerate(q_circ_input_holder):
    if q_circ_target_holder[ID] == 0:
        triv = plt.scatter(ID, sample[2], color='red', s=3)
    if q_circ_target_holder[ID] == 1:
        top = plt.scatter(ID, sample[2], color='blue', s=3)
plt.title('Angle Embedding of the Third Qubit', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(-0.5, 1.5, 5, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend([Line2D([0], [0], marker='o', color='blue', markersize=5,ls='none'), Line2D([0], [0], marker='o', color='red', markersize=5, ls='none')], ['Topological', 'Trivial'])
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Angle_Embedding_Third_Qubit.png')

plt.figure(8)
for ID, sample in enumerate(q_circ_input_holder):
    if q_circ_target_holder[ID] == 0:
        triv = plt.scatter(ID, sample[3], color='red', s=3)
    if q_circ_target_holder[ID] == 1:
        top = plt.scatter(ID, sample[3], color='blue', s=3)
plt.title('Angle Embedding of the Fourth Qubit', fontsize=16)
plt.xlabel('Epoch', fontsize = 14)
plt.yticks(np.linspace(-0.5, 1.5, 5, endpoint=True), fontsize=12)
plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
plt.ylabel('Angle of Rotation (rads)', fontsize=14)
plt.legend([Line2D([0], [0], marker='o', color='blue', markersize=5,ls='none'), Line2D([0], [0], marker='o', color='red', markersize=5, ls='none')], ['Topological', 'Trivial'])
plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/MNIST_2_Angle_Embedding_Fourth_Qubit.png')
########################################################################
# plt.figure(2)
# plt.plot(weights_holder[:,8], color='#4285F4', label='R8', linewidth='2')
# plt.plot(weights_holder[:,9], color='#24A853', label='R9', linewidth='2')
# plt.plot(weights_holder[:,10], color='#FBBC05', label='R10', linewidth='2')
# plt.plot(weights_holder[:,11], color='#EA4335', label='R11', linewidth='2')
# plt.title('Evolution of QC Weights During Training', fontsize=16)
# plt.xlabel('Epoch', fontsize = 14)
# plt.yticks(np.linspace(0, 6, 7, endpoint=True), fontsize=12)
# plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
# plt.ylabel('Angle of Rotation (rads)', fontsize=14
# )
# plt.legend()
# plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/BiTeI_Third_Layer_Weights.png')

# from matplotlib.lines import Line2D
# plt.figure(4)
# plt.plot(loss_list, color='#660099', alpha=1)
# plt.plot(c_data, color='#660099', alpha=0.5)
# plt.title("Loss of the Quantum NN During Training", fontsize=16)
# plt.yticks(np.linspace(0, 0.8, 5, endpoint=True), fontsize=12)
# plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
# plt.ylabel('Loss of Network', fontsize=14)
# plt.xlabel('Epoch', fontsize = 14)
# plt.legend([Line2D([0], [0], color='#660099', lw=4, alpha=1),
#                 Line2D([0], [0], color='#660099', lw=4, alpha=0.5)], ['QNN', 'CNN'], loc='upper right')
# plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/Loss_list_BiTeI_Quantum.png')
# plt.show()

# plt.figure(5)
# for ID, sample in enumerate(q_circ_input_holder):
#     if q_circ_target_holder[ID] == 0:
#         triv = plt.scatter(ID, sample[3], color='red', s=3)
#     if q_circ_target_holder[ID] == 1:
#         top = plt.scatter(ID, sample[3], color='blue', s=3)
# plt.title('Angle Embedding of the Fourth Qubit', fontsize=16)
# plt.xlabel('Epoch', fontsize = 14)
# plt.yticks(np.linspace(-0.5, 1.5, 5, endpoint=True), fontsize=12)
# plt.xticks(ticks=np.linspace(0, 800, 6), labels=np.linspace(0,10,6, dtype=int), fontsize=12)
# plt.ylabel('Angle of Rotation (rads)', fontsize=14)
# plt.legend([Line2D([0], [0], marker='o', color='blue', markersize=5,ls='none'), Line2D([0], [0], marker='o', color='red', markersize=5, ls='none')], ['Topological', 'Trivial'])
# plt.savefig('c:/Users/oscar/Documents/Masters ML/Neural Net/Final_figs_I_swear/BiTeI_Angle_Embedding_Fourth_Qubit.png')
# plt.show()
########################################################################

model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # data = data.to(device)
        # target = target.view(1).type(torch.LongTensor).to(device)

        output = model(torch.flatten(data, 1))
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)

        total_loss.append(loss.item())
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.2f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100))
#     f.write('\n # Final loss: ' + str(sum(total_loss) / len(total_loss)) + '     Test Accuracy: ' + str(np.round(correct/len(test_loader)*100, decimals=2)) + 
#                              '% in ' + str(len(test_loader)) + ' test samples.')
# f.close()
# from datetime import datetime

# n_samples_show = 5
# count = 0
# correct = 0
# # fig, axes = plt.subplots(nrows=1, ncols=n_samples_show , figsize=(10, 3))
# f = open("HardwareTracker.txt", "a")
# hardware_model.eval()
# # f.write('\n \nTesting on IBM Quito, Trivial = 0, Topological = 1:')
# f.write('\n \nTesting on IBM Quito, 0 = 0, 1 = 1')
# f.write('\nLoss of trained network: ' + str(sum(total_loss) / len(total_loss)))
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(test_loader):
#         if count == n_samples_show:
#             break
#         # data = data.to(device)
#         # target = target.view(1).type(torch.LongTensor).to(device)
#         data, target = data.to(device), target.to(device)

#         output = hardware_model(torch.flatten(data, 1))
        
#         pred = output.argmax(dim=1, keepdim=True) 
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         now = datetime.now()
#         dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#         f.write("\nDate and time = " + str(dt_string))
#         f.write('\nTarget: ' + str(target.item()))
#         f.write('\nNetwork output: ' + str(output))
#         if pred.eq(target.view_as(pred)).sum().item() == 1:
#             f.write('\nNetwork predicted correct')
#         else: f.write('\nNetwork prediction incorrect')
#         # axes[count].imshow(data[0].cpu().numpy().squeeze(), cmap='gray')

#         # axes[count].set_xticks([])
#         # axes[count].set_yticks([])
#         # axes[count].set_title('Predicted {}'.format(pred.item()))
        
#         count += 1
#     f.write('\nRunning on ibm_quito number of test samples are 5, number of correct: ' + str(correct))
# f.close()

# from datetime import datetime
# correct = 0
# # fig, axes = plt.subplots(nrows=1, ncols=n_samples_show , figsize=(10, 3))
# f = open("HardwareTracker.txt", "a")
# hardware_model.eval()
# # f.write('\n \nTesting on IBM Quito, Trivial = 0, Topological = 1:')
# f.write('\n \n \n Running the selected data to test top triv and close transition ')
# f.write('\nTesting on IBM Quito, 0 = Trivial, 1 = Topological. ')
# f.write('\nLoss of trained network: ' + str(sum(total_loss) / len(total_loss)))

# from DataLoadFunc_v2 import load_data
# data_tensor, phases_classification, alpha_array = load_data('c:/Users/oscar/Documents/Masters ML/Neural Net/SelectedTestingData.txt', 9, 11, 11, 2)
# data_tensor = torch.from_numpy(data_tensor.astype(np.float32))
# data_tensor = data_tensor.view(9,484)
# data_tensor = data_tensor.to(device)
# phases_classification = torch.from_numpy(phases_classification.astype(np.float32)).to(device)
# phases_classification.to(device)
# with torch.no_grad():
#     for id, data in enumerate(data_tensor):
#         data = data.to(device)
#         phases_classification[id] = phases_classification[id].view(1).type(torch.LongTensor)
#         # data, phases_classification[id] = data.to(device), phases_classificastion[id].to(device)
#         output = hardware_model(data)
#         # pred = output.argmax(dim=1, keepdim=True) 
#         # correct += pred.eq(phases_classification[id].view_as(pred)).sum().item()
#         now = datetime.now()
#         dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#         f.write("\nDate and time = " + str(dt_string))
#         f.write('\nAlpha value: ' + str(alpha_array[id]))
#         f.write('\nTarget: ' + str(phases_classification[id].item()))
#         f.write('\nNetwork output: ' + str(output))
#         # if pred.eq(target.view_as(pred)).sum().item() == 1:
#         #     f.write('\nNetwork predicted correct')
#         # else: f.write('\nNetwork prediction incorrect')
#         # axes[count].imshow(data[0].cpu().numpy().squeeze(), cmap='gray')

#         # axes[count].set_xticks([])
#         # axes[count].set_yticks([])
#         # axes[count].set_title('Predicted {}'.format(pred.item()))
        
#     # f.write('\nRunning on ibm_quito number of selected test samples are 9, number of correct: ' + str(correct))
# f.close()
