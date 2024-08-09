import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter

'''

Wrapper class for a feature extractor, representation manipulators and classifiers

'''

class FullModel(nn.Module):

    def __init__(self, feature_extractor, rep_manipulator, classifier):
        super(FullModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.rep_manipulator = rep_manipulator
        self.cls = classifier

    def forward(self, x):
        feats = self.feature_extractor(x)
        reps = self.rep_manipulator(feats)
        output = self.cls(reps)
        
        return output, feats, reps


'''

Simple convolutional feature extractor

''' 

class FeatureExtractor(nn.Module):

    def __init__(self, filters=1024, output_dim=32, H=3,W=3):
        super(FeatureExtractor, self).__init__()
        self.H = H
        self.W = W
        self.conv1 = nn.Conv2d(3, filters, kernel_size=(H,W), padding=0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters, output_dim) 
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x 


'''

Multiple Linear Classifiers in the last head. All have the exact same
shape.
From a single representation this will classify through N heads.
Problem is to make this happen in parallel instead of sequentially.

N_tasks = 1 + N° of auxiliary tasks
When N_tasks = 1, it is the same as a Linear Classifier

'''

class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim = 100, N_tasks=10, output_dim=1):
        super(MultiHeadClassifier, self).__init__()
        self.input_dim = input_dim
        self.N_tasks = N_tasks
        self.output_dim = output_dim 
        self.classifiers = nn.Linear(self.input_dim, self.N_tasks*self.output_dim)

    def forward(self, x):
        x = self.classifiers(x)
        return x.view(-1, self.N_tasks, self.output_dim) # output1, output2

'''

Manipulates an input representation of dimension "input_dim" in N different ways, by modulating the representation
and then adding a learned vector for each task.}

N_tasks = the amount of auxiliary tasks you want to use + 1 (main task)
Outputs a vector of dimensions (batch_size x N_tasks x input_dim)

'''

class RepresentationManipulator(nn.Module):
    def __init__(self, input_dim = 100, N_tasks=10, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RepresentationManipulator, self).__init__()
        self.input_dim = input_dim
        self.N_tasks = N_tasks
        self.modulators = Parameter(torch.empty(self.N_tasks-1, self.input_dim, **factory_kwargs))
        self.bias = Parameter(torch.empty(self.N_tasks-1, self.input_dim, **factory_kwargs))
        self.s = nn.Sigmoid()
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        bound = 1
        init.uniform_(self.modulators, -bound, bound)
        init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # This code is meant to apply all representation manipulation in parallel
        # Manipulation consists of modulating the input vector x through sigmoid of learned parameters
        # then adding a learned bias for each manipulation.
        # The first manipulation does nothing as it is meant to be used by the main task.
        n_batch, dim = x.shape
        x = x.unsqueeze(1)
        main_task_vector = x
        x = x.expand(-1,self.N_tasks-1, -1)
        temp_modulation = self.s(self.modulators.view(1, self.N_tasks-1, self.input_dim))
        temp_modulation = temp_modulation.expand(n_batch, -1, -1)
        x = temp_modulation * x + self.bias ## n_batch x n_tasks-1 x input_dim
        # add original vector for main task
        x = torch.cat((main_task_vector, x), dim=1)
        return x
    
def get_model(args):
        
    feature_extractor = FeatureExtractor(output_dim=args.hidden_dim,
                                         filters=args.filters,
                                         H=args.dataset_parameters['height'],
                                         W=args.dataset_parameters['width'])
    
    if args.train_method in["aux_tasks", "super_reps"]:
        rep_manipulator = RepresentationManipulator(input_dim = 100, N_tasks=10)
    else:
        rep_manipulator = nn.Identity()
    classifier = MultiHeadClassifier(input_dim = args.hidden_dim, N_tasks = args.n_heads, output_dim=1)

    return FullModel(feature_extractor=feature_extractor, 
                        rep_manipulator=rep_manipulator,
                        classifier=classifier
                        )
