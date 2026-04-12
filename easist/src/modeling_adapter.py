import torch
from torch import nn
from typing import List

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class ConvWithFFNAdapter(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_sizes: List[int] = (5, 5),
    ):
        super(ConvWithFFNAdapter, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels,
                in_channels * 2,
                k,
                stride=2,
                padding=2,
            )
            for i, k in enumerate(kernel_sizes)
        )

        self.conv_ln = torch.nn.LayerNorm(in_channels, 1e-5, True)
        self.proj = nn.Linear(in_channels, out_channels, bias=False)

    def get_out_seq_lens_tensor(self, input_lengths):
        input_lengths = input_lengths.clone()
        for _ in range(self.n_layers):
            input_lengths = torch.div(input_lengths - 1, 2, rounding_mode="floor") + 1
        return input_lengths

    def forward(self, src_tokens, src_lengths):
        x = src_tokens.transpose(1, 2).contiguous()  # B x T x (C x D) -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        x = x.transpose(1, 2).contiguous()  # -> T x B x (C x D)

        x = self.conv_ln(x)
        x = self.proj(x)

        output_length = self.get_out_seq_lens_tensor(src_lengths)
        padding_mask = lengths_to_padding_mask(output_length)

        return x, padding_mask


class FFNAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mid_dim: int,
    ):
        super(FFNAdapter, self).__init__()

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_ln = torch.nn.LayerNorm(out_dim, 1e-5, True)
        self.fc1 = nn.Linear(out_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.activation = nn.GELU()
        self.ln = torch.nn.LayerNorm(out_dim, 1e-5, True)

    def get_out_seq_lens_tensor(self, input_lengths):
        return input_lengths

    def forward(self, x, src_lengths=None):
        x = self.proj_ln(self.proj(x))
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.ln(residual + x)
        
        if src_lengths is None:
            return x
        else:
            padding_mask = lengths_to_padding_mask(src_lengths)

            return x, padding_mask


class AdapterWithBlockConv(ConvWithFFNAdapter):
    def __init__(self, in_d, out_d, mid_d, kernel_sizes=(5,5), block_size=16):
        super(AdapterWithBlockConv, self).__init__(in_d, out_d, mid_d, kernel_sizes=kernel_sizes)

        self.block_size = block_size

    def _get_conv_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        if isinstance(input_lengths, torch.Tensor):
            input_lengths = input_lengths.clone()

        def _conv_out_length(input_lengths, padding, kernel_size, stride):
            return torch.div(input_lengths + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1

        for conv in self.conv_layers:
            input_lengths = _conv_out_length(
                input_lengths, 
                conv.padding[0], 
                conv.kernel_size[0], 
                conv.stride[0]
            )

        return input_lengths

    def forward(self, x, src_lengths=None):
        x = x.transpose(1, 2)
        batch_size, feature_dim, seq_length = x.size()

        self.num_blocks = (seq_length + self.block_size - 1) // self.block_size
        padded_length = self.num_blocks * self.block_size

        if seq_length < padded_length:
            padding_size = padded_length - seq_length
            x = nn.functional.pad(x, (0, padding_size))
        
        x = x.view(batch_size, feature_dim, self.num_blocks, self.block_size)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * self.num_blocks, feature_dim, self.block_size)
        
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        
        _, num_channels, reduced_block_size = x.size()
        x = x.view(batch_size, self.num_blocks, num_channels, reduced_block_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_channels, -1)
        
        output_length = self._get_conv_output_lengths(seq_length)
        x = x[:, :, :output_length]
        x = x.transpose(1, 2)

        x = self.conv_ln(x)
        x = self.proj(x)
        if src_lengths is None:
            return x
        else:
            output_length = self._get_conv_output_lengths(src_lengths)
            padding_mask = lengths_to_padding_mask(output_length)

            return x, padding_mask


def build_adapter(config, llama_config, speech_config):

    adapter_type = config.adapter_type
    adapter_d = config.adapter_inner_dim

    if hasattr(speech_config, 'd_model'):
        in_d = speech_config.d_model
    else:
        in_d = speech_config.hidden_size
    out_d = llama_config.hidden_size

    if adapter_type == "ffn":
        return FFNAdapter(in_d, out_d, adapter_d)
    elif adapter_type == "conv":
        return ConvWithFFNAdapter(in_d, out_d, adapter_d, config.conv_kernel_sizes)
    elif adapter_type == "blockconv":
        return AdapterWithBlockConv(in_d, out_d, adapter_d, kernel_sizes=config.conv_kernel_sizes, block_size=speech_config.main_context)
    elif adapter_type == "cif":
        raise NotImplementedError("adapter_type='cif' has been removed from this codebase.")
    else:
        raise ValueError(f"Unsupported adapter_type '{adapter_type}'. Supported types: ffn, conv, blockconv.")
