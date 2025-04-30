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

class InterveneMLPNeuron:
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

        def get_hook(neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
                return output
            return hook
            
        for layer, neuron in self.intervene_neurons:
            hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(get_hook(neuron))
            self.hooks.append(hook)

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class InterveneOV_Neuron:
    def __init__(self, 
                 model, 
                 intervene_heads,
                 intervene_neurons,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_heads = intervene_heads
        self.intervene_neurons = intervene_neurons
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_head_hook(head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, head, :] = self.coeff * output[:, :, head, :] #1.3 for 4-paren
                return output
            return hook
        
        def get_neuron_hook(neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
                return output
            return hook
        
        if self.intervene_heads:
            for layer, head in self.intervene_heads:
                hook = self.model.blocks[layer].attn.hook_z.register_forward_hook(get_head_hook(head))
                self.hooks.append(hook)

        if self.intervene_neurons:
            for layer, neuron in self.intervene_neurons:
                hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(get_neuron_hook(neuron))
                self.hooks.append(hook)

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

class InterveneOV_Neuron_V1:
    def __init__(self, 
                 model, 
                 intervene_heads=None,
                 intervene_neurons=None,
                 coeff=1.0,
                 stop=False, 
                 verbose=False) -> None:
        """
        Context manager to intervene on attention heads and/or MLP neurons.

        Args:
            model: The transformer model (e.g., HookedTransformer).
            intervene_heads: List of (layer, head) tuples to intervene.
            intervene_neurons: List of (layer, neuron) tuples to intervene.
            coeff: Multiplicative coefficient to apply.
            stop: (Optional) Whether to stop execution after intervention. (currently unused)
            verbose: (Optional) Whether to print debug info.
        """
        self.model = model
        self.intervene_heads = intervene_heads or []
        self.intervene_neurons = intervene_neurons or []
        self.coeff = coeff
        self.stop = stop
        self.verbose = verbose
        self.hooks = []
        self.model.eval()

        self._register_hooks()

    def _register_hooks(self):
        """Register all head and neuron intervention hooks."""
        if self.intervene_heads:
            self._register_head_hooks()
        
        if self.intervene_neurons:
            self._register_neuron_hooks()

    def _register_head_hooks(self):
        """Register hooks for intervening on attention heads."""
        for layer, head in self.intervene_heads:
            hook_fn = self._get_head_hook(head)
            hook = self.model.blocks[layer].attn.hook_z.register_forward_hook(hook_fn)
            self.hooks.append(hook)
            if self.verbose:
                print(f"Registered head hook: Layer {layer}, Head {head}")

    def _register_neuron_hooks(self):
        """Register hooks for intervening on MLP neurons."""
        for layer, neuron in self.intervene_neurons:
            hook_fn = self._get_neuron_hook(neuron)
            hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(hook_fn)
            self.hooks.append(hook)
            if self.verbose:
                print(f"Registered neuron hook: Layer {layer}, Neuron {neuron}")

    def _get_head_hook(self, head):
        """Return a hook function to intervene on a specific attention head."""
        def hook(module, input, output):
            output[:, :, head, :] = self.coeff * output[:, :, head, :]
            return output
        return hook

    def _get_neuron_hook(self, neuron):
        """Return a hook function to intervene on a specific MLP neuron."""
        def hook(module, input, output):
            output[:, :, neuron] = self.coeff * output[:, :, neuron]
            return output
        return hook

    def close(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        if self.verbose:
            print("All hooks removed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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




class InterveneOV_HF:
    def __init__(self, model, intervene_heads, coeff, stop=False, verbose=False) -> None:
        self.model = model
        self.intervene_heads = intervene_heads
        self.verbose = verbose
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def intervention_hook(layer, head):
            def hook(module, input, output):
                output[:, :, head, :] = self.coeff * output[:, :, head, :]
                return output
            return hook

        for layer, head in self.intervene_heads:
            self.hooks.append(self.model.model.layers[layer].self_attn.register_forward_hook(intervention_hook(layer, head)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        for hook in self.hooks:
            hook.remove()



class InterveneMLPRemove:
    def __init__(self, model, verbose=False):
        self.model = model
        self.model.eval()
        self.verbose = verbose
        self.hooks = []
        self.mlp_outputs = []
        
        def capture_mlp_out(module, input, output):
            # Store each MLP output during forward pass
            self.mlp_outputs.append(output.clone())
            return output

        def remove_mlp_total_out(module, input, output):
            # Sum all saved MLP outputs and subtract them from residual
            if self.verbose:
                print(f"Subtracting total MLP contribution of shape {output.shape}")
            total_mlp_contrib = sum(self.mlp_outputs)
            return output - total_mlp_contrib

        # Register hook to capture MLP outputs from every layer
        for block in self.model.blocks:
            h = block.hook_mlp_out.register_forward_hook(capture_mlp_out)
            self.hooks.append(h)

        # Register hook at the last block’s residual stream output after MLP
        last_block = self.model.blocks[-1]
        h_final = last_block.hook_resid_post.register_forward_hook(remove_mlp_total_out)
        self.hooks.append(h_final)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.mlp_outputs.clear()

class InterveneAttentionRemove:
    def __init__(self, model, verbose=False):
        self.model = model
        self.model.eval()
        self.verbose = verbose
        self.hooks = []
        self.attn_outputs = []

        def capture_attn_out(module, input, output):
            # Store attention output from each layer
            self.attn_outputs.append(output.clone())
            return output

        def remove_total_attn_out(module, input, output):
            # Subtract total attention contributions from residual stream
            if self.verbose:
                print(f"Subtracting total attention contribution of shape {output.shape}")
            total_attn_contrib = sum(self.attn_outputs)
            return output - total_attn_contrib
        
        # Register hook to collect all attention outputs
        for block in self.model.blocks:
            h = block.hook_attn_out.register_forward_hook(capture_attn_out)
            self.hooks.append(h)

        # Register a hook at the final attention residual post
        final_block = self.model.blocks[-1]
        h_final = final_block.hook_resid_post.register_forward_hook(remove_total_attn_out)
        self.hooks.append(h_final)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attn_outputs.clear()


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
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
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