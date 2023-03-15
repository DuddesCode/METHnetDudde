import sys
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from progress.bar import IncrementalBar

t1 = torch.ones(1, requires_grad=True)
t2 = torch.ones(1, requires_grad=False)
t3 = torch.cat((t1,t2))
print(t3)
print(t3.requires_grad)
sys.exit()

writer = SummaryWriter('runs/bullshit')
class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.fc = nn.Linear(9, 3)
        #torch.nn.init.ones_(self.fc)
        #with torch.no_grad():
        self.fc.weight = torch.nn.Parameter(torch.ones([3,9],dtype=torch.float,device='cpu'))
        self.fc.bias = torch.nn.Parameter(torch.ones([3],dtype=torch.float, device='cpu'))
        #self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.fc(x)
        #x = self.bn(x)
        return x

class B(nn.Module):
    def __init__(self):
        super(B, self).__init__() 
        self.shared_fc = nn.Linear(3 ,1)
        #with torch.no_grad():
        self.shared_fc.weight = torch.nn.Parameter(torch.ones([1,3],dtype=torch.float,device='cpu'))
        self.shared_fc.bias = torch.nn.Parameter(torch.ones([1],dtype=torch.float, device='cpu'))

    def forward(self, x):
        x = self.shared_fc(x)
        return x

class C(nn.Module):
    def __init__(self, a, b):
        super(C, self).__init__()
        self.all = [a, b]
        #self.all = [nn.Linear(9,1)]
        #self.all[0].weight = torch.nn.Parameter(torch.ones([1,9],dtype=torch.float,device='cpu'))
        self.net = nn.Sequential(*self.all)

    def forward(self, x):
        x = self.net(x)
        return x
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        #self.all = [nn.Linear(9, 3), nn.BatchNorm1d(3),nn.Linear(3, 1)]
        self.all = [nn.Linear(9, 3), nn.Linear(3, 1)]
        #with torch.no_grad():
        self.all[0].weight = torch.nn.Parameter(torch.ones([3,9],dtype=torch.float,device='cpu'))
        self.all[1].weight = torch.nn.Parameter(torch.ones([1,3],dtype=torch.float,device='cpu'))

        self.all[0].bias = torch.nn.Parameter(torch.ones([3],dtype=torch.float, device='cpu'))

        self.all[1].bias = torch.nn.Parameter(torch.ones([1],dtype=torch.float, device='cpu'))
        #self.all = [nn.Linear(9,1)]
        #self.all[0].weight = torch.nn.Parameter(torch.ones([1,9],dtype=torch.float,device='cpu'))
        self.net = nn.Sequential(*self.all)

    def forward(self, x):
        x = self.net(x)
        return x



net_A = A()
net_B = B()
net_C = C(net_A.fc, net_B.shared_fc)
net_D = D()
#net_A.train()
optim_A = torch.optim.Adam(net_A.parameters())
optim_B = torch.optim.Adam(net_B.parameters())
optim_C = torch.optim.Adam(net_C.parameters())
optim_D = torch.optim.Adam(net_D.parameters())

batch_size1 = 1024
batch_size2 = 4
target = torch.randn(batch_size1,1)

x_A = torch.rand(batch_size1, 9)
ra = torch.empty(0,3)

for i in range(0,int(batch_size1/batch_size2)):

    if i == 0:
        tmp = net_A(x_A[i*batch_size2:(i*batch_size2)+batch_size2])
    else:
        with torch.no_grad():
            tmp = net_A(x_A[i*batch_size2:(i*batch_size2)+batch_size2])
    ra = torch.cat((ra, tmp),0)

    #print(ra.grad)
#ra = net_A(x_A)
#print(ra.shape)
lossB = 0.0
#for i in range(0,1):
#    rb = net_B(ra[i*batch_size2:(i*batch_size2)+batch_size2])

#    lossB += F.mse_loss(rb, target[i*batch_size2:(i*batch_size2)+batch_size2])
#lossB = lossB /((batch_size1) / (batch_size2))
rb = net_B(ra)

lossB = F.mse_loss(rb, target)
#rc = net_C(x_A)
rd = net_D(x_A)
#print("Results")
#print(rd)
#print(rb)



old_c1 = net_C.net[0].weight.clone()
old_c2 = net_C.net[1].weight.clone()
old_a = net_A.fc.weight.clone()
old_b = net_B.shared_fc.weight.clone()

old_d1 = net_D.net[0].weight.clone()
old_d2 = net_D.net[1].weight.clone()
#print("Here")
#print(torch.sum(old_d1-old_c1))
#print(torch.sum(old_d2-old_c2))
#print("___________")
#print(ra)
#print(rb)
#print(rc)
#lossC = F.mse_loss(rc, target)
writer.add_graph(net_C, x_A) 
lossD = F.mse_loss(rd, target)
print("Losses")
print(lossB)
print(lossD)
print("_____")
#print(lossB.grad())
optim_C.zero_grad()
lossB.backward()
#lossC.backward()
optim_C.step()

optim_D.zero_grad()
lossD.backward()
optim_D.step()

#ra = net_A(x_A)
#rb = net_B(ra)

#rc = net_C(x_A)
#print(ra)
#print(rb)
#print(rc)
print("Network C")
print(torch.sum(net_C.net[0].weight-old_c1))
print(torch.sum(net_C.net[1].weight-old_c2))
print("Network D")
print(torch.sum(net_D.net[0].weight-old_d1))
print(torch.sum(net_D.net[1].weight-old_d2))
print("Network A/B")
print(torch.sum(net_A.fc.weight-old_a))
print(torch.sum(net_B.shared_fc.weight-old_b))


print("________________")
old_c1 = net_C.net[0].weight.clone()
old_c2 = net_C.net[1].weight.clone()
old_a = net_A.fc.weight.clone()
old_b = net_B.shared_fc.weight.clone()

old_d1 = net_D.net[0].weight.clone()
old_d2 = net_D.net[1].weight.clone()
print(torch.sum(old_d1-old_c1))
print(torch.sum(old_d2-old_c2))

print("_____________")
rc = net_C(x_A)
rd = net_D(x_A)
print(torch.sum(rc-rd))
writer.close()
sys.exit()

def train_loop_clam(epoch, model, clam, resnet,patients, optimizer, n_classes, loss_fn = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clam.train() 
    resnet.train()
    model.train()


    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)


    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')

    batch_indices = np.arange(0, len(patients))
    np.random.shuffle(batch_indices)


    verbose =False
    for c, batch_idx in enumerate(batch_indices):
        
        if c == 0 and verbose:
            break
        patient = patients[batch_idx]
        if verbose:
            print(patient.get_identifier())
        x =  patient.get_wsis()[0].get_all_tiles((256,256))
        x = np.transpose(x, (0, 3, 1, 2))
        d = dataset.Patient_Dataset(x)
        loader = DataLoader(dataset=d, batch_size=128, collate_fn=collate_features)
        if verbose:
            print("Data loaded")

        patient_features = torch.empty(0,1024,device='cuda')
        
        for count, (batch) in enumerate(loader):
            batch = batch.to(device)
            if count == 0:
                #batch must be shuffled TODO
                tmp = resnet(batch)
            else:
                with torch.no_grad():
                    tmp = resnet(batch)
            #print("Hello there")
            #print(tmp.shape)
            #print(batch.shape)
            #print(patient_features.shape)
            #print(torch.cuda.memory_stats(device='cuda'))
            #tmp = tmp.to('cpu')
            patient_features = torch.cat((patient_features, tmp), 0)
            #break

        if verbose:
            print("Features created")
        label = patient.get_diagnosis().get_label()
        label = torch.LongTensor([label]) 
        
        #patient_features, label = collate_MIL((patient_features, label))

        label = label.to(device)
        

        logits, Y_prob, Y_hat, _, instance_dict = clam(patient_features, label=label, instance_eval=True)
        if verbose:
            print("Slide predicted")
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        if True:
            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value


            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            total_loss = 0.7 * loss + 0.3 * instance_loss
        else:
            total_loss = loss

        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        
        #print(patient.get_diagnosis().get_label())
        #print(error)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        #break #TODO
        if verbose:
            print("Backpropagation completed")
    train_loss /= len(batch_indices)
    train_error /= len(batch_indices)
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(n_classes):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct{}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))

def validate_clam(cur, epoch, model,clam,resnet, patients, n_classes, early_stopping=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet.eval()
    clam.eval()    
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.

    inst_count = 0

    prob = np.zeros((len(patients), n_classes))
    labels = np.zeros(len(patients))
    sample_size = clam.k_sample

    batch_indices = np.arange(0, len(patients))
    verbose = False
    with torch.no_grad():
        for c, batch_idx in enumerate(batch_indices):
            patient = patients[batch_idx]
            if verbose:
                print(patient.get_identifier())
            #x =  patient.get_wsis()[0].get_random_grid_tiles(256,(256,256), patient.get_diagnosis().get_label())#TODO
            x = patient.get_wsis()[0].get_all_tiles((256,256))
            x = np.transpose(x, (0, 3, 1, 2))
            d = dataset.Patient_Dataset(x)
            loader = DataLoader(dataset=d, batch_size=128, collate_fn=collate_features)
            if verbose:
                print("Data loaded")
            patient_features = torch.empty(0,1024,device='cuda')
        
            for count, (batch) in enumerate(loader):
                batch = batch.to(device)

                tmp = resnet(batch)

                patient_features = torch.cat((patient_features, tmp), 0)

            if verbose:
                print("Features predicted")
            label = patient.get_diagnosis().get_label()
            label = torch.LongTensor([label]) 
            label = label.to(device)
    
            logits, Y_prob, Y_hat, _, instance_dict = clam(patient_features, label=label, instance_eval=True)#TODO
            if verbose:
                print("CLAM done")
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()
            if True:
                instance_loss = instance_dict['instance_loss']

                inst_count += 1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

        
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error
    
    val_error /= len(patients)
    val_loss /= len(patients)
    print("Validation error is {}".format(val_error))
    early_stopping(epoch, val_loss, model, ckpt_name="s_{}_checkpoint.pt".format(cur))
    
    if early_stopping.early_stop:
        return True
    
    return False