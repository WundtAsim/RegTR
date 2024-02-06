import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.pcd_partition import point_to_node_partition, index_select, pairwise_distance

class PPFEmbeddingSin(nn.Module):
    """
    PPF embedding, using sine function.
    Args:

    """
    def __init__(self, 
                 d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.scale = 1 # 2 * math.pi
        self.angle_k = 50  # the number of nearest points used to compute the angle

        # self.glo_embedding = SinusoidalPositionalEmbedding(d_model=d_model)
        self.glo_proj = nn.Linear(4, d_model)
        self.reduction_a = 'max'

        self.n_dim = 3
        self.num_pos_feats = d_model // 3 // 2 * 2
        self.temperature = 10000
        self.padding = d_model - self.num_pos_feats * self.n_dim
    
    @torch.no_grad()
    def get_embedding_indices(self, 
                points, 
                nodes, 
                points_normals,
                nodes_normals):
        r"""Compute the indices of PPF(Point Pair Feature) embedding.

        Args:
            points: List(B) of (N, 3)
            nodes: List(B) of (M, 3)
            points_normals: List(B) of (N, 3)
            nodes_normals: List(B) of (M, 3)
        Returns:
            d_indices: List(B) of (M, K)
            a_n1_d_indices: List(B) of (M, K)
            a_n2_d_indices: List(B) of (M, K)
            a_n1_n2_indices: List(B) of (M, K)
        """
        d_indices = []
        a_n1_d_indices = []
        a_n2_d_indices = []
        a_n1_n2_indices = []
        for b in range(len(points)):

            # get point to node partition: patch
            _, node_masks, node_knn_indices = point_to_node_partition(
                points=points[b], nodes=nodes[b], point_limit=self.angle_k) # node_knn_indices: (M, K)
            knn_points = index_select(points[b], node_knn_indices, dim=0) # (M, K, 3)
            knn_normals = index_select(points_normals[b], node_knn_indices, dim=0) # (M, K, 3)
            
            # get the distance matrix of each node to the knn points
            dist_map = torch.norm(knn_points - nodes[b].unsqueeze(1), dim=-1) # (M, K)
            d_indices.append((dist_map * self.scale).cuda()) # (M, K)

            # get the angle between the i normal and ij vector
            ij_vector = knn_points - nodes[b].unsqueeze(1)  # (M, K, 3)
            normal_vector = nodes_normals[b].unsqueeze(1).expand(-1, knn_points.shape[-2], -1) # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)  # (M, K)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)  # (M, K)
            angles = torch.atan2(sin_values, cos_values) # (M, K)
            dot_product = torch.sum(ij_vector * normal_vector, dim=-1) # (M, K) # make sure the angle is in the range of [0, pi]
            mask = dot_product < 0
            angles[mask] = -angles[mask] # (M, K)
            a_n1_d_indices.append(angles.cuda())

            # get the angle between the j normal and ij vector
            ij_vector = nodes[b].unsqueeze(1) - knn_points  # (M, K, 3)
            normal_vector = knn_normals # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)  # (M, K)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)  # (M, K)
            angles = torch.atan2(sin_values, cos_values) # (M, K)
            dot_product = torch.sum(ij_vector * normal_vector, dim=-1) # (M, K)
            mask = dot_product < 0
            angles[mask] = -angles[mask] # (M, K)
            a_n2_d_indices.append(angles.cuda())

            # get the angle between i normal and j normal
            i_normal = nodes_normals[b].unsqueeze(1).expand(-1, knn_points.shape[-2], -1) # (M, K, 3)
            j_normal = knn_normals # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(i_normal,j_normal, dim=-1), dim=-1)  # (M, K)
            cos_values = torch.sum(nodes_normals[b].unsqueeze(1) * knn_normals, dim=-1)  # (M, K)
            angles = torch.atan2(sin_values, cos_values) # (M, K)
            dot_product = torch.sum(nodes_normals[b].unsqueeze(1) * knn_normals, dim=-1) # (M, K)
            mask = dot_product < 0
            angles[mask] = -angles[mask] # (M, K)
            a_n1_n2_indices.append(angles.cuda())

        return d_indices, a_n1_d_indices, a_n2_d_indices, a_n1_n2_indices

    def forward(self, 
                points, 
                nodes, 
                points_normals,
                nodes_normals) -> torch.Tensor:
        r""" FFP embedding, using sine function.
        Args:
            points: List(B) of (N, 3)
            nodes: List(B) of (M, 3)
            points_normals: List(B) of (N, 3)
            nodes_normals: List(B) of (M, 3)
        Returns:
            global_embeddings: List(B) of (M, D)
            local_embeddings: List(B) of (M, D)
            
        """     

        # get the indices of PPF embedding
        d_embeddings, a_n1_d_embeddings, a_n2_d_embeddings, a_n1_n2_embeddings = self.get_embedding_indices(
            points=points, nodes=nodes, points_normals=points_normals, nodes_normals=nodes_normals)
        
        global_embeddings = []
        local_embeddings = []

        for b in range(len(d_embeddings)):
            # get the embeddings of global embedding
            global_embedding = torch.stack([d_embeddings[b], a_n1_d_embeddings[b], 
                                            a_n2_d_embeddings[b], a_n1_n2_embeddings[b]], dim=-1) # (M, K, 4)
            # get projection embeddings of PPF embedding
            global_embedding = self.glo_proj(global_embedding) # (M, K, D)
            # get the pool embeddings of Geo embedding
            if self.reduction_a == 'max':
                global_embedding = torch.max(global_embedding, dim=-2)[0] # (M, D)
            elif self.reduction_a == 'mean':
                global_embedding = torch.mean(global_embedding, dim=-2)[0] # (M, D)
            else:
                raise ValueError(f'Unsupported reduction_a: {self.reduction_a}')
            
            # get the global final embeddings
            global_embeddings.append(global_embedding)

            # get the position embeddings of xyz sine function
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=nodes[b].device)
            dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
            nodes[b] = nodes[b] * self.scale
            pos_divided = nodes[b].unsqueeze(-1) / dim_t
            pos_sin = pos_divided[..., 0::2].sin()
            pos_cos = pos_divided[..., 1::2].cos()
            pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*nodes[b].shape[:-1], -1)
            pos_emb = F.pad(pos_emb, (0, self.padding)) 
            local_embeddings.append(pos_emb)
            
        return global_embeddings, local_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings
    
class GeoEmbedding(nn.Module):
    """
    """
    def __init__(self, d_model: int = 256, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.angle_k = 50  # the number of nearest points used to compute the angle
        self.glo_embedding = SinusoidalPositionalEmbedding(d_model=d_model)
        self.reduction_a = 'max'

        self.n_dim = 3
        self.num_pos_feats = d_model // 3 // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim
        self.scale = 2 * math.pi
        self.proj_d = nn.Linear(d_model, d_model)
    @torch.no_grad()
    def get_embedding_indices(self, 
                points, 
                nodes, 
                points_normals,
                nodes_normals):
        r"""Compute the indices of PPF(Point Pair Feature) embedding.

        Args:
            points: List(B) of (N, 3)
            nodes: List(B) of (M, 3)
            points_normals: List(B) of (N, 3)
            nodes_normals: List(B) of (M, 3)
        Returns:
            d_indices: (M, M)
            a_ij_indices: (M, M)
            a_n1_d_indices: (M, K)
            a_n2_d_indices: (M, K)
            a_n1_n2_indices: (M, K)

        """
        d_indices = []
        a_ij_indices = []
        for b in range(len(points)):
            # get the distance matrix of each node to the other nodes
            dist_map = torch.sqrt(pairwise_distance(nodes[b], nodes[b])) # (M, M)
            d_indices.append((dist_map * self.scale).cuda()) # (M, M)

            # get point to node partition: patch
            _, node_masks, node_knn_indices = point_to_node_partition(
                points=points[b], nodes=nodes[b], point_limit=self.angle_k) # node_knn_indices: (M, K)
            knn_points = index_select(points[b], node_knn_indices, dim=0) # (M, K, 3)
            knn_normals = index_select(points_normals[b], node_knn_indices, dim=0) # (M, K, 3)

            # get the angle between ij and knn points
            ij_vector = nodes[b].unsqueeze(1) - nodes[b].unsqueeze(0)  # (M, M, 3)
            ij_vector = ij_vector.unsqueeze(2).expand(-1, -1, self.angle_k, -1) # (M, M, K, 3)
            knn_vector = knn_points - nodes[b].unsqueeze(1)  # (M, K, 3)
            knn_vector = knn_vector.unsqueeze(1).expand(-1, nodes[b].shape[-2], -1, -1) # (M, M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, knn_vector, dim=-1), dim=-1)  # (M, M, K)
            cos_values = torch.sum(ij_vector * knn_vector, dim=-1)  # (M, M, K)
            a_ij_indices.append(torch.atan2(sin_values, cos_values).cuda())  # (M, M, K)

        return d_indices, a_ij_indices
    def forward(self, 
                points, 
                nodes, 
                points_normals,
                nodes_normals) -> torch.Tensor:
        r""" Geotransformer embedding, using sine function.
        Args:
            points: List(B) of (N, 3)
            nodes: List(B) of (M, 3)
            points_normals: List(B) of (N, 3)
            nodes_normals: List(B) of (M, 3)
        Returns:
            global_embeddings: List(B) of (M, M, D)
            local_embeddings: List(B) of (M, D)
            
        """     

        # get the indices of PPF embedding
        d_embeddings, a_ij_embeddings = self.get_embedding_indices(
            points=points, nodes=nodes, points_normals=points_normals, nodes_normals=nodes_normals)
        
        global_embeddings = []
        local_embeddings = []

        for b in range(len(d_embeddings)):
            # get the embeddings of global embedding
            d_embeddings[b] = self.glo_embedding(d_embeddings[b]) # (M, M, D)
            a_ij_embeddings[b] = self.glo_embedding(a_ij_embeddings[b]) # (M, M, K, D)

            # get projection embeddings of Geo embedding
            d_embeddings[b] = self.proj_d(d_embeddings[b]) # (M, M, D)
            a_ij_embeddings[b] = self.proj_d(a_ij_embeddings[b]) # (M, M, K, D)

            # get the pool embeddings of Geo embedding
            if self.reduction_a == 'max':
                a_ij_embeddings[b] = torch.max(a_ij_embeddings[b], dim=2)[0] # (M, M, D)
            elif self.reduction_a == 'mean':
                a_ij_embeddings[b] = torch.mean(a_ij_embeddings[b], dim=2)[0] # (M, M, D)
            else:
                raise ValueError(f'Unsupported reduction_a: {self.reduction_a}')
            
            # get the final embeddings
            global_embeddings.append(d_embeddings[b] + a_ij_embeddings[b])

            # get the embeddings of position
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=nodes[b].device)
            dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

            nodes[b] = nodes[b] * self.scale
            pos_divided = nodes[b].unsqueeze(-1) / dim_t
            pos_sin = pos_divided[..., 0::2].sin()
            pos_cos = pos_divided[..., 1::2].cos()
            pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*nodes[b].shape[:-1], -1)
            pos_emb = F.pad(pos_emb, (0, self.padding)) 
            local_embeddings.append(pos_emb)
            
        return global_embeddings, local_embeddings