import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss

from models.resnet import Reduced_ResNet18
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random

class ProxyContrastiveReplay(ContinualLearner):
    """
        Proxy-based Contrastive Replay,
        Implements the strategy defined in
        "PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning"
        https://arxiv.org/abs/2304.04408

        This strategy has been designed and tested in the
        Online Setting (OnlineCLScenario). However, it
        can also be used in non-online scenarios
        """
    def __init__(self, model, opt, params):
        super(ProxyContrastiveReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

        self.soft_ = nn.Softmax(dim=1)
    
    

    def train_learner(self, x_train, y_train, task_number):        
        self.before_train(x_train, y_train)
        #print("y_trainnnnnnn", y_train.shape, type(y_train), y_train)
        #print("x_trainnnnnnn", x_train.shape, type(x_train))
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=False, num_workers=0,
                                       drop_last=True)
        
        unique_classes = set()
        
        count_ = np.sum(self.buffer.buffer_label.cpu().numpy() == 0)
        # Assuming each batch's labels are in the second element
        
        for _, labels in train_loader:
            unique_classes.update(labels.numpy())
        
        '''if count_ != self.buffer.buffer_label.shape[0]:
            unique_classes.update(self.buffer.buffer_label.cpu().numpy())
            #print("self.buffer.buffer_img.cpu().numpy().shape", self.buffer.buffer_img.cpu().numpy().shape)
            #print("self.buffer.buffer_img.permute(0, 3, 1, 2).cpu().numpy().shape", self.buffer.buffer_img.permute(0, 3, 1, 2).cpu().numpy().shape)
            train_dataset_buffer = dataset_transform(self.buffer.buffer_img.permute(0, 2, 3, 1).cpu().numpy(), 
                                                     self.buffer.buffer_label.cpu().numpy(), 
                                                     transform=transforms_match[self.data])
            #train_loader_buffer = data.DataLoader(train_dataset_buffer, batch_size=self.batch, shuffle=False, num_workers=0, drop_last=True)
        
            # Merge the two datasets
            merged_dataset = ConcatDataset([train_dataset, train_dataset_buffer])
            
            # Create a DataLoader for the merged dataset
            merged_loader = data.DataLoader(merged_dataset, batch_size=self.batch, shuffle=False, num_workers=0, drop_last=True)

        else:
            merged_dataset = train_dataset
            merged_loader = train_loader

        #print(f"Number of unique classes: {len(unique_classes)}", unique_classes)'''

        device = "cuda"
        Model_Carto = Reduced_ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        
        #mapping = {26:0, 86:1, 2:2, 55:3, 75:4, 93:5, 16:6, 73:7, 54:8, 95:9,
         #         53:10, 92:11, 78:12, 13:13, 7:14, 30:15, 22:16, 24:17, 33:18, 8:19,
          #        43:20, 62:21, 3:22, 71:23, 45:24, 48:25, 6:26, 99:27, 82:28, 76:29,
           #       60:30, 80:31, 90:32, 68:33, 51:34, 27:35, 18:36, 56:37, 63:38, 74:39,
            #      1:40, 61:41, 42:42, 41:43, 4:44, 15:45, 17:46, 40:47, 38:48, 5:49,
             #     91:50, 59:51, 0:52, 34:53, 28:54, 50:55, 11:56, 35:57, 23:58, 52:59,
              #    10:60, 31:61, 66:62, 57:63, 79:64, 85:65, 32:66, 84:67, 14:68, 89:69,
               #   19:70, 29:71, 49:72, 97:73, 98:74, 69:75, 20:76, 94:77, 72:78, 77:79,
                #  25:80, 37:81, 81:82, 46:83, 39:84, 65:85, 58:86, 12:87, 88:88, 70:89,
                 # 87:90, 36:91, 21:92, 83:93, 9:94, 96:95, 67:96, 64:97, 47:98, 44:99}

        mapping = {value: index for index, value in enumerate(unique_classes)}
        #print(mapping)
        
        # Training
        Carto = []
        for epoch_ in range(12):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                #print("targets", targets)
                
                
                targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
                #print("targets", targets)
                
                optimizer_.zero_grad()
                #print("inputs.shapeeeeeeeeeee", inputs.shape)
                outputs = Model_Carto(inputs)
                #print("outputs.shape", outputs.shape)
                soft_ = self.soft_(outputs)
                #print("soft_", soft_)
                confidence_batch = []
                #print("outputs", outputs)
        
                for i in range(targets.shape[0]):
                  confidence_batch.append(soft_[i,targets[i]].item())
                if (targets.shape[0] != self.batch):
                  for j in range(self.batch - targets.shape[0]):
                    confidence_batch.append(0)
                confidence_epoch.append(confidence_batch)
                #print(len(confidence_epoch[0]))
        
                loss = criterion_(outputs, targets)
                loss.backward()
                optimizer_.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        

            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total)
            conf_tensor = torch.tensor(confidence_epoch)
            conf_tensor = conf_tensor.reshape(conf_tensor.shape[0]*conf_tensor.shape[1])
            conf_tensor = conf_tensor[:total]
            #print(conf_tensor.shape)
            
            Carto.append(conf_tensor.numpy())
            scheduler_.step()

        Carto_tensor = torch.tensor(np.array(Carto))
        #print(Carto_tensor.shape)
        Confidence_mean = Carto_tensor.mean(dim=0)
        Variability = Carto_tensor.std(dim = 0)
        #print("Confidence_mean.shape", Confidence_mean.shape)
        #print("Variability.shape", Variability.shape)
        
        plt.scatter(Variability, Confidence_mean, s = 2)
        
        
        # Add Axes Labels
        
        plt.xlabel("Variability") 
        plt.ylabel("Confidence") 
        
        # Display
        
        plt.savefig('scatter_plot.png')


        # Number of top values you're interested in
        top_n = (self.params.mem_size//(task_number+1)) + 1
        
        # Find the indices that would sort the array
        sorted_indices_1 = np.argsort(Confidence_mean.numpy())
        sorted_indices_2 = np.argsort(Variability.numpy())



        # Take the last 'top_n' indices (i.e., the top values)
        top_indices_1 = sorted_indices_1[:int(0.9*top_n)]
        top_indices_2 = sorted_indices_1[-int(0.05*top_n):]
        top_indices_3 = sorted_indices_2[-(top_n - (int(0.9*top_n) + int(0.05*top_n))):]
        
        #print("top_indicesssss", top_indices.shape, top_indices, type(top_indices))

        top_indices_12 = np.concatenate((top_indices_3, top_indices_2))
        top_indices_123 = np.concatenate((top_indices_12, top_indices_1))
        
        # If you want these indices in ascending order, you can sort them
        #top_indices_sorted = np.sort(top_indices_123)
        top_indices_sorted = top_indices_123

        
        
        # Take the last 'top_n' indices (i.e., the top values)
        ##top_indices_1 = sorted_indices_1[:top_n]
        
        #top_indices_sorted = top_indices_1[::-1]
        ##top_indices_sorted = top_indices_1
        
        #print("top_indices_sorted", top_indices_sorted, top_indices_sorted.shape)
        print("top_indices_sorted.shape", top_indices_sorted.shape)
        
        subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
        #print("subset_dataaaaaaaa", subset_data)
        trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=self.batch, shuffle=False, num_workers=0)

        images_list = []
        labels_list = []
        
        for images, labels in trainloader_C:  # Assuming train_loader is your DataLoader
            images_list.append(images)
            labels_list.append(labels)
        
        all_images = torch.cat(images_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        #print(all_images.shape)  # This should print something like torch.Size([50000, 3, 32, 32]) depending on your DataLoader's batch size
        #print(all_labels.shape)  # This should print torch.Size([50000])



        #print("task_numberrrrrrrrrr", task_number)

        if task_number > 0:
    
            space = self.params.mem_size
            pointer = 0  # This will keep track of where to insert in M
            
            for j in range(task_number+1):  
                portion = space // (task_number + 1 - j)  # Use integer division for portion size
                
                # Fill the buffer
                for k in range(portion):
                    if task_number != j:
                        self.buffer.buffer_img[pointer] = self.buffer.buffer_img[j*self.params.mem_size//task_number + k]
                        self.buffer.buffer_label[pointer] = self.buffer.buffer_label[j*self.params.mem_size//task_number + k]
                        pointer += 1
                    else:
                        self.buffer.buffer_img[pointer] = all_images.to(device)[k]
                        self.buffer.buffer_label[pointer] = all_labels.to(device)[k]
                        pointer += 1
                    
                space -= portion


        # set up model
        self.model = self.model.train()
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))
                for j in range(self.mem_iters):
                    logits, feas= self.model.pcrForward(batch_x_combine)
                    novel_loss = 0*self.criterion(logits, batch_y_combine)
                    self.opt.zero_grad()


                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        # mem_x, mem_y = Rotation(mem_x, mem_y)
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])


                        mem_logits, mem_fea= self.model.pcrForward(mem_x_combine)

                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels]

                        combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                        combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                            combined_feas_aug)
                        combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                  combined_feas_aug_normalized.unsqueeze(1)],
                                                 dim=1)
                        PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                        novel_loss += PSC(features=cos_features, labels=combined_labels)


                    novel_loss.backward()
                    self.opt.step()
                # update mem
                if count_ == self.buffer.buffer_label.shape[0]:
                    self.buffer.update(batch_x, batch_y)

        #print("self.buffer.buffer_img", self.buffer.buffer_img.shape, type(self.buffer.buffer_img))
        #print("self.buffer.buffer_label", self.buffer.buffer_label.shape, type(self.buffer.buffer_label), self.buffer.buffer_label)

        if count_ == self.buffer.buffer_label.shape[0]:
            self.buffer.buffer_img = all_images.to(device)
            self.buffer.buffer_label = all_labels.to(device)

        #print("self.buffer.buffer_img", self.buffer.buffer_img.shape, type(self.buffer.buffer_img))
        #print("self.buffer.buffer_label", self.buffer.buffer_label.shape, type(self.buffer.buffer_label), self.buffer.buffer_label)
        
        unique_classes__ = set()
        unique_classes__.update(self.buffer.buffer_label.cpu().numpy())

        print("unique_classes__", unique_classes__)
        
        self.after_train()
