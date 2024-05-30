import torch


class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def _quantize(message, scale, zero_point, dtype):
        quantize_msg = torch.quantize_per_tensor(message * scale,
                                                 scale=1.0,
                                                 zero_point=zero_point,
                                                 dtype=dtype
                                                 )

        dequantize_msg = torch.dequantize(quantize_msg)
        # dequantize_msg.register_hook(lambda grad: print(f'In_quantize quantize_msg grad is {grad}'))
        # discrete_msg = dequantize_msg.detach().clone()
        # dequantize_msg /= scale
        return dequantize_msg #, discrete_msg

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, scale, zero_point, dtype):
        #def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input)
        return QuantizeFunction._quantize(input, scale, zero_point, dtype)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # Needs to return outputs as the number of parameters to forward
        return grad_output, None, None, None


