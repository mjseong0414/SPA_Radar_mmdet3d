
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DFusionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DFusionTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DFusionCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                pts_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            pts_feats : Input for point data feature
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=(mlvl_feats, pts_feats),
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DFusionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default???
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DFusionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class Detr3DFusionCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DFusionCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.img_attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        self.pts_attention_weights = nn.Linear(embed_dims,1*3*num_points)
        self.fusion_attention_weights =nn.Linear(embed_dims,2*embed_dims)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.pts_output_proj = nn.Linear(embed_dims, embed_dims)
        self.fusion_proj = nn.Linear(2*embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.img_attention_weights, val=0., bias=0.)
        constant_init(self.pts_attention_weights, val=0., bias=0.)
        constant_init(self.fusion_attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.fusion_proj, distribution='uniform', bias=0.)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        breakpoint()
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2) 

        bs, num_query, _ = query.size()

        img_attention_weights = self.img_attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        pts_attention_weights = self.pts_attention_weights(query).view(
            bs, 1, num_query, 1, self.num_points, 3)
        fusion_attention_weights = self.fusion_attention_weights(query).view(
            num_query,self.num_points,2*self.embed_dims)
        mlvl_feats , pts_feats = value
        # breakpoint()
        reference_points_3d, output, mask = img_feature_sampling(
            mlvl_feats , reference_points, self.pc_range, kwargs['img_metas'])
        pts_output = radar_feature_sampling(pts_feats , reference_points)

        attention_img = img_attention_weights.sigmoid() * mask
        attention_pts = pts_attention_weights.sigmoid()
        pts_output = torch.nan_to_num(pts_output)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        output = output * attention_img
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        pts_output = pts_output * attention_pts
        pts_output = pts_output.sum(-1).sum(-1).sum(-1)
        pts_output = pts_output.permute(2, 0, 1)

        output = self.output_proj(output)
        pts_output = self.pts_output_proj(pts_output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        
        fusion_output = torch.cat((output,pts_output),2).to('cuda')*fusion_attention_weights.sigmoid()
        fusion_output = self.fusion_proj(fusion_output)
        # return self.dropout(fusion_output) ### v4
        return self.dropout(fusion_output) + inp_residual + pos_feat ### v3, v5
        # return self.dropout(output) + self.dropout(pts_output) + inp_residual + pos_feat ### v2

###############################################################################
def img_feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    breakpoint()
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask

def radar_feature_sampling(pts_feats, reference_points):
    breakpoint()
    # breakpoint()
    B, num_query = reference_points.size()[:2]
    reference_points_bev = reference_points.clone()
    reference_points_bev = (reference_points_bev[...,0:2]-0.5)/0.5
    reference_points_bev = reference_points_bev.view(1,900,1,2).repeat(3,1,1,1)
    sampled_feats = []
    for lvl,feat in enumerate(pts_feats):
        B, C, H, W = feat.size()
        C_3 = int(C/3)
        feat = feat.view(B*3, C_3, H, W)
        sampled_feat = F.grid_sample(feat, reference_points_bev)
        sampled_feat = sampled_feat.view(B, 3, C_3, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
        # breakpoint()
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C_3, num_query, 1,  1, 3)
    return sampled_feats

#############################################################################################################

@ATTENTION.register_module()
class Detr3DFusionDeformableCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DFusionDeformableCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.offset_sampling = nn.Linear(embed_dims, 18)
        self.img_attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        self.pts_attention_weights = nn.Linear(embed_dims,1*3*num_points)
        self.fusion_attention_weights =nn.Linear(embed_dims,2*embed_dims)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.pts_output_proj = nn.Linear(embed_dims, embed_dims)
        self.fusion_proj = nn.Linear(2*embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.img_attention_weights, val=0., bias=0.)
        constant_init(self.pts_attention_weights, val=0., bias=0.)
        constant_init(self.fusion_attention_weights, val=0., bias=0.)
        constant_init(self.offset_sampling, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.fusion_proj, distribution='uniform', bias=0.)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2) 

        bs, num_query, _ = query.size()
        
        offsets = self.offset_sampling(query).view(bs, num_query, 6, 3).sigmoid()
        ref = reference_points.clone()
        ref = ref.view(bs, num_query, 1, 3)
        ref_offsets = torch.cat([ref, offsets], dim=2) # 0 is reference point, 1 ~ 3 are camera, 4 ~ 6 are radar

        img_attention_weights = self.img_attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        pts_attention_weights = self.pts_attention_weights(query).view(
            bs, 1, num_query, 1, self.num_points, 3)
        
        fusion_attention_weights = self.fusion_attention_weights(query).view(
            num_query,bs,2*self.embed_dims)
        mlvl_feats , pts_feats = value
        # breakpoint()
        output, mask = img_feature_sampling_deformable(
            mlvl_feats , ref_offsets, self.pc_range, kwargs['img_metas'])
        pts_output = radar_feature_sampling_deformable(pts_feats , ref_offsets)

        attention_img = img_attention_weights.sigmoid() * mask
        attention_pts = pts_attention_weights.sigmoid()
        pts_output = torch.nan_to_num(pts_output)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        output = output * attention_img
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        pts_output = pts_output * attention_pts
        pts_output = pts_output.sum(-1).sum(-1).sum(-1)
        pts_output = pts_output.permute(2, 0, 1)
        
        output = self.output_proj(output)
        pts_output = self.pts_output_proj(pts_output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points)).permute(1, 0, 2)
        
        fusion_output = torch.cat((output,pts_output),2).to('cuda')*fusion_attention_weights.sigmoid()
        fusion_output = self.fusion_proj(fusion_output)
        # return self.dropout(fusion_output) ### v4
        return self.dropout(fusion_output) + inp_residual + pos_feat ### v3, v5
        # return self.dropout(output) + self.dropout(pts_output) + inp_residual + pos_feat ### v2

def img_feature_sampling_deformable(mlvl_feats, ref_offsets, pc_range, img_metas):
    ref_offsets_3d= ref_offsets[:, :, :4]
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = ref_offsets_3d.new_tensor(lidar2img) # (B, N, 4, 4)
    ref_offsets_3d = ref_offsets_3d.clone()
    ref_offsets_3d[..., 0:1] = ref_offsets_3d[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0] 
    ref_offsets_3d[..., 1:2] = ref_offsets_3d[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    ref_offsets_3d[..., 2:3] = ref_offsets_3d[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4) => homogeneous
    reference_points = torch.cat((ref_offsets_3d, torch.ones_like(ref_offsets_3d[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)# [1, 6, 900, 4, 4, 1]
    lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, num_query, 4, 1, 1) # [1, 6, 900, 4, 4, 4]
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1) # [1, 6, 900, 4, 4]
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps) # mask [1, 6, 900, 4, 1]
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2 # [1, 6, 900, 4, 2]
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 4, 1).permute(0, 2, 3, 1, 4, 5) # [1, 1, 900, 6, 4, 1]
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W) 
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 4, 2) # [6, 900, 4, 2]
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl) # [6, 256, 900, 4]
        sampled_feat = sampled_feat.view(B, N, C, num_query, 4).permute(0, 2, 3, 1, 4) # [1, 256, 900, 6, 4]
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1) # [1, 256, 900, 6, 4(num_points), 4(level)]
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  4, len(mlvl_feats)) # [1, 256, 900, 6, 4(num_points), 4(level)]
    return sampled_feats, mask

def radar_feature_sampling_deformable(pts_feats, ref_offsets):
    B, num_query = ref_offsets.size()[:2]
    ref_point = ref_offsets[:, :, :1]
    radar_point = ref_offsets[:, :, 4:]
    sampling_points_bev = torch.cat([ref_point, radar_point], dim=2)
    sampling_points_bev = (sampling_points_bev[...,0:2]-0.5)/0.5
    sampling_points_bev = sampling_points_bev.repeat(3,1,1,1)
    sampled_feats = []
    for lvl,feat in enumerate(pts_feats):
        B, C, H, W = feat.size()
        C_3 = int(C/3)
        feat = feat.view(B*3, C_3, H, W)
        sampled_feat = F.grid_sample(feat, sampling_points_bev) # [3, 256, 900, num_point]
        sampled_feat = sampled_feat.view(B, 3, C_3, num_query, 4).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1) # [1, 256, 900, lvl, num_point, 1]
    sampled_feats = sampled_feats.view(B, C_3, num_query, 1, 4, 3)
    return sampled_feats # [1, 256, 900, 3, 4, 1]