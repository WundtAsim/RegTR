"""Functions to manipulate sequences, e.g. packing/padding"""
import torch
import torch.nn as nn


def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_sequences(padded, seq_lens):
    """Reverse of pad_sequence"""
    sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
    return sequences


def split_src_tgt(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    B = len(stack_lengths) // 2
    separate = torch.split(feats, stack_lengths, dim=dim)
    return separate[:B], separate[B:]

def pad_sequence_3d(sequences, padding_value=0):
    '''
    for 3d sequences padding
    '''
    # Find max dimensions
    max_dim1 = max([s.size(0) for s in sequences])
    max_dim2 = max([s.size(1) for s in sequences])
    max_dim3 = max([s.size(2) for s in sequences])

    # Create a tensor filled with padding_value
    out_dims = (len(sequences), max_dim1, max_dim2, max_dim3)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    # Copy sequences to the right places in out_tensor
    for i, tensor in enumerate(sequences):
        dim1, dim2, dim3 = tensor.size()
        out_tensor[i, :dim1, :dim2, :dim3] = tensor

    return out_tensor