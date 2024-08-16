import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from module.layers import OutputLayer
from torch.nn import init

class FCNN(nn.Module):
    def __init__(self, orig_extra_fea_len, extra_fea_len, task_types, n_h=2, **kwargs):
        super(FCNN, self).__init__()
        hidden_dim = extra_fea_len
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.loss_aggregation = kwargs.get('loss_aggregation', "sum")
        self.shared_fc = self._create_shared_fc(orig_extra_fea_len, hidden_dim, n_h)
        
        self.task_types = task_types
        self.n_tasks = len(task_types)
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc_outs = nn.ModuleList([OutputLayer(extra_fea_len, task_tp) for task_tp in self.task_types])
        
        # define log_vars for each task
        if self.loss_aggregation == "trainable_weight_sum":
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # He Initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He Initialization
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def _create_shared_fc(self, input_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, extra_fea, **kwargs):
        x = extra_fea.view(extra_fea.size(0), -1)
        shared_rep = self.shared_fc(x)
        shared_rep = self.dropout(shared_rep)
        task_outputs = []
        for i in range(self.n_tasks):
            task_outputs.append(self.fc_outs[i](shared_rep))
        return task_outputs