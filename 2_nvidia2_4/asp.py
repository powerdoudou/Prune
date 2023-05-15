import types
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sparse_masklib import create_mask



class ASP:
    model = None
    verbosity = 0
    optimizer = None
    sparse_parameters = []
    calculate_mask = None

    @classmethod
    def init_model_for_pruning(
        cls,
        model,
        mask_calculator="m4n2_1d",
        verbosity=3,
        whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d],
        custom_layer_dict={},
    ):
        assert cls.model is None, "ASP has been initialized already."
        cls.model = model
        cls.verbosity = verbosity

        if isinstance(mask_calculator, str):
            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            cls.calculate_mask = create_mask_from_pattern

        # function to extract variables that will be sparsified.
        # idea is that you will add one of these functions for each module type that can be sparsified.

        sparse_parameter_list = {
            torch.nn.Linear: ["weight"],
            torch.nn.Conv1d: ["weight"],
            torch.nn.Conv2d: ["weight"],
        }
        if (custom_layer_dict):  # Update default list to include user supplied custom (layer type : parameter tensor), make sure this tensor type is something ASP knows how to prune
            sparse_parameter_list.update(custom_layer_dict)
            whitelist += list(custom_layer_dict.keys())

        for module_type in whitelist:
            assert module_type in sparse_parameter_list, (
                "Module %s :: Don't know how to sparsify module." % module.dtype()
            )
        # find all sparse modules, extract sparse parameters and decorate
        def add_sparse_attributes(module_name, module):
            sparse_parameters = sparse_parameter_list[type(module)]
            for p_name, p in module.named_parameters():
                if p_name in sparse_parameters and p.requires_grad:
                    # check for NVIDIA's TC compatibility: we check along the horizontal direction
                    if p.dtype == torch.float32 and (
                        (p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0
                    ):  # User defines FP32 and APEX internally uses FP16 math
                        print(
                            "[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity"
                            % (module_name, p_name, str(p.size()), str(p.dtype))
                        )
                        continue
                    if p.dtype == torch.float16 and (
                        (p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0
                    ):  # For Conv2d dim= K x CRS; we prune along C
                        print(
                            "[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity"
                            % (module_name, p_name, str(p.size()), str(p.dtype))
                        )
                        continue

                    if cls.verbosity >= 3:
                        print(
                            "[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity"
                            % (module_name, p_name, str(p.size()), str(p.dtype))
                        )

                    mask = torch.ones_like(p).bool()
                    buffname = p_name.split(".")[-1]  # buffer names cannot contain "."
                    module.register_buffer("__%s_mma_mask" % buffname, mask)
                    cls.sparse_parameters.append(
                        (module_name, module, p_name, p, mask)
                    )
                else:
                    if cls.verbosity >= 3:
                        print(
                            "[ASP] Not sparsifying %s::%s of size=%s and type=%s"
                            % (module_name, p_name, str(p.size()), str(p.dtype))
                        )

        for name, sparse_module in eligible_modules(model, tuple(whitelist)):
            add_sparse_attributes(name, sparse_module)

    @classmethod
    def init_optimizer_for_pruning(cls, optimizer):
        assert cls.optimizer is None, "ASP has initialized optimizer already."
        assert (
            cls.calculate_mask is not None
        ), "Called ASP.init_optimizer_for_pruning before ASP.init_model_for_pruning."

        # store pointer to original optimizer step method
        cls.optimizer = optimizer
        cls.optimizer.__step = optimizer.step

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for (
                    module_name,
                    module,
                    p_name,
                    p,
                    mask,
                ) in cls.sparse_parameters:
                    if p.grad is not None:  # thx pjudd
                        p.grad.mul_(mask)
            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for (
                    module_name,
                    module,
                    p_name,
                    p,
                    mask,
                ) in cls.sparse_parameters:
                    p.mul_(mask)
            return rval

        cls.optimizer.step = types.MethodType(__step, cls.optimizer)

    @classmethod
    def compute_sparse_masks(cls): #!aaaa
        with torch.no_grad():
            for module_name, module, p_name, p, mask in cls.sparse_parameters:
                mask.set_(cls.calculate_mask(p)) # torch.Size([8, 16]) # mask = cls.calculate_mask(p) # in place op
                p.mul_(
                    mask
                )  # in-place multiplication, so pruned weights are 0-values, hence checkpoint will have 0s for pruned weights

    @classmethod
    def prune_trained_model(cls, model, optimizer):
        # add mask buffers to model (init_model_for_pruning), augment optimizer (init_optimizer_for_pruning) and compute masks (compute_sparse_masks)
        cls.init_model_for_pruning(
            model,
            mask_calculator="m4n2_1d",
            verbosity=2,
            whitelist=[torch.nn.Linear, torch.nn.Conv2d],
        )
        cls.init_optimizer_for_pruning(optimizer)
        cls.compute_sparse_masks()


# --------------------  --------------------
def eligible_modules(model, whitelist_layer_types):
    eligible_modules_list = []
    for name, mod in model.named_modules():
        if (isinstance(mod, whitelist_layer_types)):
            eligible_modules_list.append((name, mod))
    return eligible_modules_list

if __name__=='__main__':
    model = nn.Sequential(
            nn.Linear(1, 2),
            nn.PReLU(),
            nn.Linear(2, 3),
        ).cuda()
    sparse_parameter_list = {
            torch.nn.Linear: ["weight"],
            torch.nn.Conv1d: ["weight"],
            torch.nn.Conv2d: ["weight"],
        }
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    myasp=ASP()
    whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]    
            
    for name, sparse_module in eligible_modules(model, tuple(whitelist)):
        # add_sparse_attributes(name, sparse_module)
        module_name=name
        module=sparse_module
        sparse_parameters = sparse_parameter_list[type(module)]#['weight']
        # print(sparse_parameters)#weight
        # print(module)
        for p_name, p in module.named_parameters():
            if p_name in sparse_parameters and p.requires_grad:       
                mask = torch.ones_like(p).bool()
                buffname = p_name
                module.register_buffer("__%s_mma_mask" % buffname, mask)
                sparse_parameters.append((module_name, module, p_name, p, mask))        
        # print(sparse_parameters)
            # if p_name in sparse_parameters and p.requires_grad:
    for name,param in model.named_buffers():
        print(name)
        print(param)