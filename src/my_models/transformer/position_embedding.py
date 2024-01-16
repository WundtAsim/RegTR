import math
import torch
import torch.nn as nn
import numpy as np
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
        self.scale = 2 * math.pi
        self.angle_k = 128  # the number of nearest points used to compute the angle

        self.embedding = SinusoidalPositionalEmbedding(d_model=d_model)
        self.proj_d = nn.Linear(d_model, d_model)
        self.proj_a_ij = nn.Linear(d_model, d_model)
        self.proj_a_patch = nn.Linear(3*d_model, d_model)
        self.reduction_a = 'max'
    
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
        a_n1_d_indices = []
        a_n2_d_indices = []
        a_n1_n2_indices = []
        for b in range(len(points)):
            # get the distance matrix of each node to the other nodes
            dist_map = torch.sqrt(pairwise_distance(nodes[b], nodes[b])) # (M, M)
            d_indices.append((dist_map * self.scale).cuda()) # (M, M

            # get the angle between the normal and other nodes
            ij_vector = nodes[b].unsqueeze(1) - nodes[b].unsqueeze(0)  # (M, M, 3)
            normal_vector = nodes_normals[b].unsqueeze(1).expand(-1, nodes[b].shape[-2], -1) # (M, M, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)  # (M, M)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)  # (M, M)
            a_ij_indices.append(torch.atan2(sin_values, cos_values).cuda())  # (M, M)

            # get point to node partition: patch
            _, node_masks, node_knn_indices = point_to_node_partition(
                points=points[b], nodes=nodes[b], point_limit=self.angle_k) # node_knn_indices: (M, K)
            knn_points = index_select(points[b], node_knn_indices, dim=0) # (M, K, 3)
            
            knn_normals = index_select(points_normals[b], node_knn_indices, dim=0) # (M, K, 3)

            # get the angle of n1-ij
            ij_vector = knn_points - nodes[b].unsqueeze(1)  # (M, K, 3)
            normal_vector = nodes_normals[b].unsqueeze(1).expand(-1, self.angle_k, -1) # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)  # (M, K)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)  # (M, K)
            a_n1_d_indices.append(torch.atan2(sin_values, cos_values).cuda())  # (M, K)
            
            # get the angle of n2-ji
            ij_vector = nodes[b].unsqueeze(1) - knn_points  # (M, K, 3)
            normal_vector = knn_normals # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)
            a_n2_d_indices.append(torch.atan2(sin_values, cos_values).cuda())  # (M, K)
            
            # get the angle of n1-n2
            ij_vector = nodes_normals[b].unsqueeze(1).expand(-1, self.angle_k, -1) # (M, K, 3)
            normal_vector = knn_normals # (M, K, 3)
            sin_values = torch.linalg.norm(torch.cross(ij_vector, normal_vector, dim=-1), dim=-1)
            cos_values = torch.sum(ij_vector * normal_vector, dim=-1)
            a_n1_n2_indices.append(torch.atan2(sin_values, cos_values).cuda())  # (M, K)

        return d_indices, a_ij_indices, a_n1_d_indices, a_n2_d_indices, a_n1_n2_indices

    def forward(self, 
                points: torch.Tensor, 
                nodes: torch.Tensor, 
                points_normals: torch.Tensor,
                nodes_normals: torch.Tensor) -> torch.Tensor:
        r""" FFP embedding, using sine function.
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
        d_embeddings, a_ij_embeddings, a_n1_d_embeddings, a_n2_d_embeddings, a_n1_n2_embeddings = self.get_embedding_indices(
            points=points, nodes=nodes, points_normals=points_normals, nodes_normals=nodes_normals)
        
        global_embeddings = []
        local_embeddings = []

        for b in range(len(d_embeddings)):
            # get the embeddings of PPF embedding
            d_embeddings[b] = self.embedding(d_embeddings[b]) # (M, M, D)
            a_ij_embeddings[b] = self.embedding(a_ij_embeddings[b]) # (M, M, D)
            a_n1_d_embeddings[b] = self.embedding(a_n1_d_embeddings[b]) # (M, K, D)
            a_n2_d_embeddings[b] = self.embedding(a_n2_d_embeddings[b]) # (M, K, D)
            a_n1_n2_embeddings[b] = self.embedding(a_n1_n2_embeddings[b]) # (M, K, D)
            a_patch_embeddings = torch.cat([a_n1_d_embeddings[b], a_n2_d_embeddings[b], a_n1_n2_embeddings[b]], dim=-1) # (M, K, 3*D)

            # get projection embeddings of PPF embedding
            d_embeddings[b] = self.proj_d(d_embeddings[b]) # (M, M, D)
            a_ij_embeddings[b] = self.proj_d(a_ij_embeddings[b]) # (M, M, D)
            a_patch_embeddings = self.proj_a_patch(a_patch_embeddings) # (M, K, D)

            # get the pool embeddings of PPF embedding
            if self.reduction_a == 'max':
                a_patch_embeddings = torch.max(a_patch_embeddings, dim=1)[0] # (M, D)
            elif self.reduction_a == 'mean':
                a_patch_embeddings = torch.mean(a_patch_embeddings, dim=1)[0] # (M, D)
            else:
                raise ValueError(f'Unsupported reduction_a: {self.reduction_a}')
            
            # get the final embeddings
            global_embeddings.append(d_embeddings[b] + a_ij_embeddings[b])
            local_embeddings.append(a_patch_embeddings)
            
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