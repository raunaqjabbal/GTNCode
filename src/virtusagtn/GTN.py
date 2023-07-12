import torch as _torch
import torch.nn as _nn
import torch.optim as _optim
from torch import Tensor

import time as _time
import pandas as _pd            # type: ignore
import os as _os
from copy import deepcopy as _deepcopy
import numpy as _np
import gc as _gc

from .utils import _diffopt_state_dict, _divide_chunks, _imshow, _cycle
import higher as _higher        # type: ignore

import typing as _typing


class GTN:
    ''' Base GTN Class that sets up architecture.
    '''
    def __init__(self,
        loss_fn: _typing.Callable[[_torch.Tensor, _torch.Tensor], _torch.Tensor], 
        learnerlist: _typing.List[_torch.nn.Module], 
        num_classes : int,
        batch_size: int = 4, 
        plot_steps: int = 25,
        device: _typing.Optional[_torch.device] = None, 
        #    metrics,
        ) -> None:
        r""" Initializes the structure of the GTN

        Parameters:
            loss_fn: Loss function for the GTN training ``learnerlist``
            
            learnerlist: A list of ``torch.nn.Module``
            
            num_classes: Number of classes in the data
                        
            batch_size: Number of ``inner_loop_steps`` to run before training on ``train_data``  
            
            plot_steps: Number of trained models after which learned data is printed
            
            device: PyTorch device to run ono either ``gpu`` or ``cpu``
        
        """
        
        self.device = device

        
        self.learnerlist = learnerlist
        self.loss_fn = loss_fn
        
        self.num_classes = num_classes
        self.inner_loop_iterations: int
        self.plot_steps = plot_steps
        self.batch_size = batch_size
        # self.metrics = metrics 
        
        
        self.params_to_train: _typing.List[_torch.Any] 
        self.override: _typing.List[_typing.Any]
        self.override_params: _typing.Dict[str,_typing.Any]
        self.outer_opt: _typing.Callable[[_typing.Any],_torch.optim.Optimizer] # Reference to an optimizer
        self.outer_opt_params: _typing.Optional[_typing.Dict[str,_typing.Any]]
        self.inner_opt: _typing.Callable[[_typing.Any],_torch.optim.Optimizer] # Reference to an optimizer
        self.inner_opt_params: _typing.Optional[_typing.Dict[str,_typing.Any]]
        
        self.noise_size: int
        
    def train(self, 
        train_loader: _typing.Collection[_torch.Tensor], 
        test_loader: _typing.Collection[_torch.Tensor],
        path: str = './gtn',
        epochs: int = 3
        ) -> _pd.DataFrame:
        """ A function for training the GTN
        Parameters:
            train_data: PyTorch DataLoader (or iterable) for outer loop training
            
            test_data: PyTorch DataLoader (or iterable) for outer loop validation
             
            path: Path where trained Learners are stored  
            
            epochs: Number of epochs to train a learner
        Returns:
            history: Pandas DataFrame consisting of the history and paths to learners 
        """
        self.batches = list(_divide_chunks(_np.arange(self.inner_loop_iterations), self.batch_size))

        self.epochs = epochs
        self.path = path

        self.steps_per_epoch = len(train_loader)
        self.train_loader = iter(_cycle(train_loader))
        self.test_loader  = iter(_cycle(test_loader)) 

        
        gtn = _pd.DataFrame(columns= ('Path','Inner Loss','Inner Accuracy','Train Loss','Train Accuracy','Test Loss','Test Accuracy'))
        if not _os.path.exists(self.path):
            _os.makedirs(self.path)

        then = _time.time()
        for it in range(len(self.learnerlist)):
            learner = _deepcopy(self.learnerlist[it]).to(self.device)
            inner_optim = self.inner_opt(learner.parameters(), **self.inner_opt_params )
            metrics: _typing.Dict[str,_typing.Any] = {'Inner Loss':[],'Inner Accuracy':[],'Train Loss':[],'Train Accuracy':[],'Test Loss':[],'Test Accuracy':[]}

            for epoch in range(self.epochs):
                
                for batchset in self.batches:
                    self.outer_optim.zero_grad(set_to_none=True)
                    train_data, train_target = next(self.train_loader)
                        
                    with _higher.innerloop_ctx(learner, inner_optim, override = {key: [value] for key,value in zip(list(self.override_params.keys()),self.override)} ) as (flearner, diffopt):
                        
                        for step in batchset:
                            inner_data, inner_target = self.get_innerloop_data(step)
                            inner_data, inner_target= inner_data.to(self.device), inner_target.to(self.device)
                            inner_output = flearner(inner_data)
                            inner_loss = self.loss_fn(inner_output, inner_target) 
                            inner_pred = inner_output.argmax(dim=1, keepdim=True) 
                            inner_accuracy = _np.round(inner_pred.eq(inner_target.view_as(inner_pred)).sum().item() / len(inner_target) * 100, 2)
                            diffopt.step(inner_loss)
                        
                        train_data, train_target= train_data.to(self.device), train_target.to(self.device)
                        train_output = flearner(train_data)
                        train_loss = self.loss_fn(train_output, train_target)
                        train_pred = train_output.argmax(dim=1, keepdim=True)
                        train_accuracy = _np.round(train_pred.eq(train_target.view_as(train_pred)).sum().item() / len(train_target) * 100, 2)
                        
                        train_loss.backward()
                        self.outer_optim.step()

                        learner.load_state_dict(flearner.state_dict())
                        inner_optim.load_state_dict(_diffopt_state_dict(diffopt))
                        
                with _torch.no_grad():
                    learner.eval()
                    test_data, test_target = next(self.test_loader)
                    test_data, test_target= test_data.to(self.device), test_target.to(self.device)
                    test_output = learner(test_data) 
                    test_loss = self.loss_fn(test_output, test_target)
                    test_pred = test_output.argmax(dim=1, keepdim=True)
                    test_accuracy = _np.round(test_pred.eq(test_target.view_as(test_pred)).sum().item() / len(test_target) * 100 ,2)

                    metrics["Inner Accuracy"].append(inner_accuracy)
                    metrics["Inner Loss"].append(_np.round(inner_loss.item(),3))
                    metrics["Train Accuracy"].append(train_accuracy)
                    metrics["Train Loss"].append(_np.round(train_loss.item(),3))
                    metrics["Test Accuracy"].append(test_accuracy)
                    metrics["Test Loss"].append(_np.round(test_loss.item(),3))

                print("E:",it//self.steps_per_epoch, 
                            "\tB:",it%self.steps_per_epoch, 
                            "\t Inner Acc: %5.2f" % inner_accuracy, "%",
                            "  Loss: %.3f" % _np.round(inner_loss.item(),3),
                            " \t Train Acc: %5.2f" % train_accuracy, "%",
                            "  Loss: %.3f" % _np.round(train_loss.item(),3), 
                            "   \t Test Acc: %5.2f" % test_accuracy, "%",
                            "  Loss: %.3f" % _np.round(test_loss.item(),3), 
                            "  \tIT: ",(it+1),
                            sep=""
                            )
                        
            # print()
            inner_optim.zero_grad()
            checkpoint = { 'model': learner, 'optimizer': inner_optim.state_dict() }
            metrics["Path"] = f'{self.path}/{it}.pth'
            _torch.save(checkpoint, metrics['Path'])
            gtn.loc[it] = metrics
            if (it + 1) % self.plot_steps == 0:
                _imshow(train_data)    
            
        del train_loss, inner_loss, test_loss, train_data, inner_data, test_data, learner, inner_optim
        _gc.collect()
        _torch.cuda.empty_cache()


        now = _time.time()
        # curriculum_data.requires_grads=False

        print("\n\nTotal Time Taken: ",now-then, "\t Average Time: ", (now-then)/len(self.learnerlist))
    
    def compileoptimizer(self,
        inner_opt: _typing.Callable[[_typing.Any],_torch.optim.Optimizer] = _torch.optim.SGD, 
        inner_opt_params: _typing.Dict[str,_typing.Any] = {'lr':0.01}, 
        override_params: _typing.Dict[str,_typing.Any] = {'lr': 0.02, 'momentum':0.9}, 
        outer_opt: _typing.Callable[ [_typing.Any],_torch.optim.Optimizer] = _torch.optim.Adam, 
        outer_opt_params: _typing.Dict[str,_typing.Any] = {'lr':0.01, 'betas': (0.9,0.9)},
        ) -> None:
        """ Compiles the outer loop optimizer reference ``outer_opt``
        Parameters:     
            inner_opt: Reference to an optimizer for inner loop training
            
            inner_opt_params: Dictionary of ``inner_opt`` parameters and corresponding values
            
            override_params: Dictionary of ``inner_opt`` parameters that need to be trained
            
            outer_opt: Reference to an optimizer for outer loop meta learning
            
            outer_opt_params: Dictionary of ``inner_opt`` parameters and corresponding values
        """
        self.override_params = override_params
        self.outer_opt = outer_opt
        self.outer_opt_params = outer_opt_params
        self.inner_opt = inner_opt
        self.inner_opt_params = inner_opt_params
        
        
        
        self.override = [_nn.Parameter(Tensor(x).to(self.device)) if isinstance(x,list) 
                                else _nn.Parameter(Tensor([x]).to(self.device))  for x in self.override_params.values()]
        
        self.params_to_train += self.override 
        self.outer_optim = self.outer_opt(self.params_to_train, **self.outer_opt_params)
        
        samplemodel = _deepcopy(self.learnerlist[0]).to(self.device)
        
        inner_optim = self.inner_opt(samplemodel.parameters(), **self.inner_opt_params)
        # print(inner_optim)
        # with _higher.innerloop_ctx(_deepcopy(self.learnerlist[0]), inner_optim, 
        #     override = {key: [value] for key,value in zip(list(self.override_params.keys()),self.override)} ) as (flearner, diffopt):
        #     None
        
        # del inner_optim, samplemodel, flearner, diffopt
        # _gc.collect()
        # _torch.cuda.empty_cache()
    
    def get_innerloop_data(self, step) -> _typing.Tuple[_torch.Tensor, _torch.Tensor]:
        """ Function is overriden by base class that provides data to the Learner for inner loop training
        Parameters:
            None
            
        Returns:
            None
        """
        return (None, None)
        
class TeacherGTN(GTN):
    """ Implements a GTN with a Teacher model 
    """
    def compile(self, 
        teacher: _torch.nn.Module,
        inner_loop_iterations:int = 32,
        use_curriculum: bool = True, 
        noise_size: _typing.Optional[int] = 128, 
        inner_batch_size: _typing.Optional[int] = 128,  
        teacher_noise: _torch.Tensor = None
        ) -> None:
        
        #   teacher = _nn.DataParallel(teacher, device_ids=list(range(_torch.cuda.device_count())))
        
        """ Prepares flow of data and optimizer for Teacher GTN learning
            Calls parent class function ``compileoptimizer()`` for initializing optimizers 
        Parameters:
            teacher: Teacher that takes in ``noise`` and outputs data that will be used by Learners
            
            inner_loop_iterations: Number of inner loop iterations 
            
            use_curriculum: If ``True``, use a trainable curriculum, else generate random data for the Teacher 
            
            noise_size: dim of noise that the ``teacher`` will accept
            
            teacher_noise: Directly provide teacher noise
        Returns:
            None
        """
        self.params_to_train=[]
        
        self.use_curriculum = use_curriculum
        self.params_to_train += list(teacher.parameters())
        self.teacher = teacher.to(self.device)
        self.inner_batch_size = inner_batch_size
        
        if isinstance(teacher_noise, _torch.Tensor):
            self.inner_loop_iterations = int(teacher_noise.shape[0])
            self.teacher_noise = _nn.Parameter(teacher_noise, requires_grad=True) 
            self.params_to_train += [self.teacher_noise]
            self.use_curriculum == True
            
        elif self.use_curriculum == True:
            self.inner_loop_iterations = inner_loop_iterations
            self.teacher_noise = _nn.Parameter(_torch.randn(self.inner_loop_iterations, inner_batch_size ,noise_size), requires_grad=True) 
            self.params_to_train += [self.teacher_noise]
        
        self.teacher_labels = _torch.arange(self.inner_batch_size) % self.num_classes                       
        self.one_hot = _nn.functional.one_hot(self.teacher_labels, self.num_classes)
        
 
    def get_innerloop_data(self, step) -> _typing.Tuple[_torch.Tensor, _torch.Tensor]:
        """ Called during training to feed data to Learner 
        Parameters:
            step: current inner loop iteration number
            
        Returns:
           Data and labels for training in a list 
        """
        if self.use_curriculum == True:
            z_vec = self.teacher_noise[step]
        else:
            z_vec = _torch.randn(self.inner_batch_size, self.noise_size).to(self.device)
        return self.teacher(z_vec.to(self.device), self.one_hot.to(self.device)), self.teacher_labels
    
   
class DataGTN(GTN):
    def compile(self, 
        curriculum_loader: _typing.Collection[_torch.Tensor],
        ):
        """ Initializes parameters for Data GTN learning
            Calls parent class function ``compileoptimizer()`` for initializing optimizers 
        Parameters:
            data: PyTorch DataLoader (or iterable) containing synthetic data for Learner training

        
        Returns:
            None
        """
        #   teacher = _nn.DataParallel(teacher, device_ids=list(range(_torch.cuda.device_count())))
        self.params_to_train=[]
        
        self.inner_loop_iterations = len(curriculum_loader)
        self.curriculum_loader  = iter(_cycle(curriculum_loader)) 

        curriculum_data=[]
        curriculum_labels = []

        for i in range(self.inner_loop_iterations):
            loader = next(self.curriculum_loader)
            curriculum_data  +=[loader[0]]
            curriculum_labels+=[loader[1]]

        self.curriculum_data = _nn.Parameter(_torch.stack((curriculum_data),0).detach(),requires_grad=True)
        self.curriculum_labels = _torch.stack(curriculum_labels,0).detach()
        self.params_to_train += [self.curriculum_data]
        
    def get_innerloop_data(self, step) -> _typing.Tuple[_torch.Tensor,_torch.Tensor]:
        """ Called during training to feed data to Learner 
        Parameters:
            step: current inner loop iteration number
            
        Returns:
           Data and labels for training in a list 
        """
        
        return self.curriculum_data[step], self.curriculum_labels[step]