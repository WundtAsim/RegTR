import torch

@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int = 128,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,) the index of the node that each point belongs to
        node_sizes (LongTensor): (M,) the number of points in each node
        node_masks (BoolTensor): (M,) the mask of nodes that have points
        node_knn_indices (LongTensor): (M, K) the indices of the knn points of each node
        node_knn_masks (BoolTensor) (M, K) the mask of the knn points of each node
    """
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool, device=points.device)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0], device=points.device)  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)

    return point_to_node, node_masks, node_knn_indices
    

# util functions-------------------------------------------------------------------------------------------------
def pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.
    Args:
        x (Tensor): (*, N, C) 
        y (Tensor): (*, M, C) 
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """

    channel_dim = -1
    xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]

    x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # x2+y2+z2 (*, N, C) -> (*, N) -> (*, N, 1)
    y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # x2+y2+z2 (*, M, C) -> (*, M) -> (*, 1, M)
    sq_distances = x2 - 2 * xy + y2 # (x1-x2)2 +(y1-y2)2 + (z1-z2)2 (*, N, M)
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output
