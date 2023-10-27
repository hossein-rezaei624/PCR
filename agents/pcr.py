import torch ## our final approach fiveth
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss

import torch.nn as nn
from models.resnet import ResNet18
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle

from collections import defaultdict
from torch.utils.data import Subset

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


    def distribute_samples(self, probabilities, M):
        # Normalize the probabilities
        total_probability = sum(probabilities.values())
        normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}
    
        # Calculate the number of samples for each class
        samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
        
        # Check if there's any discrepancy due to rounding and correct it
        discrepancy = M - sum(samples.values())
        
        for key in samples:
            if discrepancy == 0:
                break
            if discrepancy > 0:
                samples[key] += 1
                discrepancy -= 1
            else:
                samples[key] -= 1
                discrepancy += 1

        return samples


    def distribute_excess(self, lst):
        # Calculate the total excess value
        total_excess = sum(val - 500 for val in lst if val > 500)
    
        # Number of elements that are not greater than 500
        recipients = [i for i, val in enumerate(lst) if val < 500]
    
        num_recipients = len(recipients)
    
        # Calculate the average share and remainder
        avg_share, remainder = divmod(total_excess, num_recipients)
    
        lst = [val if val <= 500 else 500 for val in lst]
        
        # Distribute the average share
        for idx in recipients:
            lst[idx] += avg_share
        
        # Distribute the remainder
        for idx in recipients[:remainder]:
            lst[idx] += 1
        
        # Cap values greater than 500
        for i, val in enumerate(lst):
            if val > 500:
                return distribute_excess(lst)
                break
    
        return lst
    
    
    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        
        
        unique_classes = set()
        for _, labels, indices_1 in train_loader:
            unique_classes.update(labels.numpy())
        

        device = "cuda"
        Model_Carto = ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        

        mapping = {value: index for index, value in enumerate(unique_classes)}
        reverse_mapping = {index: value for value, index in mapping.items()}


        # Initializing the dictionaries        
        confidence_by_class = {class_id: {epoch: [] for epoch in range(6)} for class_id, __ in enumerate(unique_classes)}

        
        # Training
        Carto = torch.zeros((6, len(y_train)))
        for epoch_ in range(6):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            for batch_idx, (inputs, targets, indices_1) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)                
                targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
                
                optimizer_.zero_grad()
                outputs = Model_Carto(inputs)
                soft_ = self.soft_(outputs)
                confidence_batch = []
        
                # Accumulate confidences and counts
                for i in range(targets.shape[0]):
                    confidence_batch.append(soft_[i,targets[i]].item())
                    
                    # Update the dictionary with the confidence score for the current class for the current epoch
                    confidence_by_class[targets[i].item()][epoch_].append(soft_[i, targets[i]].item())

                loss = criterion_(outputs, targets)
                loss.backward()
                optimizer_.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
                conf_tensor = torch.tensor(confidence_batch)
                Carto[epoch_, indices_1] = conf_tensor
                
            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))

            scheduler_.step()

        mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}
        std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(6)])) for class_id, __ in enumerate(unique_classes)}
        

       
        
        # set up model
        self.model = self.model.train()
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y, indices_1 = batch_data
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
                self.buffer.update(batch_x, batch_y)


        list_of_indices = []
        counter__ = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                counter__ +=1
                list_of_indices.append(i)

        top_n = counter__



        updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}
        
        ##print("updated_std_of_means_by_class", updated_std_of_means_by_class)

        dist = self.distribute_samples(updated_std_of_means_by_class, top_n)

        
        num_per_class = top_n//len(unique_classes)
        counter_class = [0 for _ in range(len(unique_classes))]

        if len(y_train) == top_n:
            condition = [num_per_class for _ in range(len(unique_classes))]
            diff = top_n - num_per_class*len(unique_classes)
            for o in range(diff):
                condition[o] += 1
        else:
            condition = [value for k, value in dist.items()]


        check_bound = len(y_train)/len(unique_classes)
        ##print("check_bound", check_bound)
        ##print("condition", condition, sum(condition))
        for i in range(len(condition)):
            if condition[i] > check_bound:
                ##print("iiiiiiiii", i)
                condition = self.distribute_excess(condition)
                break

        
        #here

        class_indices = defaultdict(list)
        for idx, (_, label, __) in enumerate(train_dataset):
            class_indices[label.item()].append(idx)
            print("class_indices", class_indices)

        selected_indices = []

        for class_id, num_samples in enumerate(condition):
            class_samples = class_indices[reverse_mapping[class_id]]  # get indices for the class
            selected_for_class = random.sample(class_samples, num_samples)
            selected_indices.extend(selected_for_class)
            print("selected_indices", selected_indices)

        selected_dataset = Subset(train_dataset, selected_indices)
        trainloader_C = torch.utils.data.DataLoader(selected_dataset, batch_size=self.batch, shuffle=True, num_workers=0)

        images_list = []
        labels_list = []
        
        for images, labels, indices_1 in trainloader_C:  # Assuming train_loader is your DataLoader
            images_list.append(images)
            labels_list.append(labels)
        
        all_images = torch.cat(images_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        self.buffer.buffer_label[list_of_indices] = all_labels.to(device)
        self.buffer.buffer_img[list_of_indices] = all_images.to(device)
        
        self.after_train()
