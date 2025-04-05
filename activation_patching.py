import pdb

class StopForward(Exception):
    pass

class InterveneRS:
    def __init__(self, 
                 model, 
                 layer_n,
                 position, 
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.layer = layer_n
        self.position = position
        self.verbose = False
        self.hook = None
        self.model.eval()

        def hook(module, input, output):
            n = int(output.shape[0]/2) # lengths of clean and corrupted inputs
            output[:n, self.position, :] = output[n:, self.position, :] # replace corrupted with clean or vice versa
            return output
        self.hook1 = self.model.blocks[self.layer].hook_resid_pre.register_forward_hook(hook)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        self.hook1.remove()

class InterveneMLP:
    def __init__(self, 
                 model, 
                 layer_n,
                 position, 
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.layer = layer_n
        self.position = position
        self.verbose = False
        self.hook = None
        self.model.eval()

        def hook(module, input, output):
            n = int(output.shape[0]/2) # lengths of clean and corrupted inputs
            output[:n, self.position, :] = output[n:, self.position, :] # replace corrupted with clean or vice versa
            return output
        self.hook1 = self.model.blocks[self.layer].hook_mlp_out.register_forward_hook(hook)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        self.hook1.remove()

class InterveneAttention:
    def __init__(self, 
                 model, 
                 layer_n,
                 position, 
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.layer = layer_n
        self.position = position
        self.verbose = False
        self.hook = None
        self.model.eval()

        def hook(module, input, output):
            # pdb.set_trace()
            n = int(output.shape[0]/2) # lengths of clean and corrupted inputs
            output[:n, self.position, :] = output[n:, self.position, :] # replace corrupted with clean or vice versa
            return output
        self.hook1 = self.model.blocks[self.layer].hook_attn_out.register_forward_hook(hook)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        self.hook1.remove()

class InterveneAttentionHead:
    def __init__(self, 
                 model, 
                 positions,
                 intervene_heads,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.positions = positions
        self.intervene_heads = intervene_heads
        self.verbose = False
        self.hooks = []
        self.model.eval()

        def get_hook(layer, head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                n = int(output.shape[0]/2)
                for each in self.positions:
                    output[:n, each, head, :] = output[n:, each, head, :]
                # for batch_idx, token_idx in self.positions:
                #     output[batch_idx, token_idx, head] = output[batch_idx+n, token_idx, head]
                return output
            return hook
            
        for layer, head in self.intervene_heads:
            self.hooks.append(self.model.blocks[layer].attn.hook_z.register_forward_hook(get_hook(layer, head)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

class InterveneOV:
    def __init__(self, 
                 model, 
                 intervene_heads,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_heads = intervene_heads
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_hook(layer, head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, head, :] = self.coeff * output[:, :, head, :] #1.3 for 4-paren
                return output
            return hook
            
        for layer, head in self.intervene_heads:
            self.hooks.append(self.model.blocks[layer].attn.hook_z.register_forward_hook(get_hook(layer, head)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()


class InterveneNeurons:
    def __init__(self, 
                 model, 
                 intervene_neurons,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_neurons = intervene_neurons
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_hook(layer, neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] #1.3 for 4-paren
                return output
            return hook
            
        for neuron in self.intervene_neurons:
            l = int(neuron.split("N")[0].split("L")[1])
            n = int(neuron.split("N")[1])
            # intervene on the first layer output
            self.hooks.append(self.model.blocks[l].mlp.hook_pre.register_forward_hook(get_hook(l, n)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()