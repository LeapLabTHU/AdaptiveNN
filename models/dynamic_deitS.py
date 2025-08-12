import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, Mlp



__all__ = [
    'dynamic_deitS',
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_base_patch16_384',
]

# torch.autograd.set_detect_anomaly(True) 

@register_model
def dynamic_deitS(**kwargs):
    model = dynamic_deitS(**kwargs)
    return model


class policy_net_patch(nn.Module):
    def __init__(
        self, 
        feature_in_chans, hidden_chans=128, kernel_size=7, feature_size=7, seq_len=4, is_policy_net=True
        ):
        super().__init__()

        self.ln_list = nn.ModuleList(
            [nn.ModuleList(
                [
                    partial(nn.LayerNorm, eps=1e-6)(1024),
                    partial(nn.LayerNorm, eps=1e-6)(512),
                ]
            ) for _ in range(seq_len)])
        self.act_layer = nn.GELU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(
            in_channels=feature_in_chans, out_channels=feature_in_chans, 
            kernel_size=kernel_size, stride=1, padding= (kernel_size - 1) // 2, groups=feature_in_chans, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels=feature_in_chans, out_channels=hidden_chans, 
            kernel_size=1, stride=1, padding=0, bias=False)
        self.linear1 = nn.Linear(hidden_chans * feature_size * feature_size, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 512, bias=False)

        if is_policy_net:
            self.output_head = nn.Sequential(
                nn.Linear(512, 2, bias=False),
                nn.Sigmoid())
        else:
            self.output_head = nn.Sequential(
                nn.Linear(512, 1, bias=False),)


    def forward(self, x, step_index):

        x = self.conv1(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.act_layer(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.ln_list[step_index][0](x)
        x = self.act_layer(x)

        x = self.linear2(x)
        x = self.ln_list[step_index][1](x)
        x = self.act_layer(x)

        x = self.output_head(x)

        return x


class dynamic_deitS(nn.Module):
    def __init__(
        self,
        seq_len, feature_in_chans, recover_n, remaining_blocks,
        glance_input_size, glance_net_depth, glance_net_mlp_ratio, glance_net_drop_path, 
        focus_patch_size, focus_net_reg_size, focus_net_depth, focus_net_mlp_ratio, focus_net_drop_path, 
        multi_cls_drop_path,
        policy_net_hidden_chans, policy_net_kernel_size,
        **kwargs):
        super().__init__()

        self.seq_len = seq_len

        self.glance_net_depth = glance_net_depth
        self.glance_net_mlp_ratio = glance_net_mlp_ratio
        self.glance_net_drop_path = glance_net_drop_path
        self.glance_input_size = glance_input_size
        self.glance_feature_size = self.glance_input_size // 16

        self.focus_net_depth = focus_net_depth
        self.focus_net_mlp_ratio = focus_net_mlp_ratio
        self.focus_net_drop_path = focus_net_drop_path
        self.focus_patch_size = focus_patch_size
        self.focus_feature_size = self.focus_patch_size // 16
        self.feature_in_chans = feature_in_chans

        self.multi_cls_drop_path = multi_cls_drop_path


        self.recover_n = recover_n
        self.remaining_blocks = remaining_blocks
        self.focus_net_reg_size = focus_net_reg_size


        glance_net_kwargs = kwargs.copy()
        glance_net_kwargs['depth'] = self.glance_net_depth
        glance_net_kwargs['mlp_ratio'] = self.glance_net_mlp_ratio
        glance_net_kwargs['drop_path_rate'] = self.glance_net_drop_path
        glance_net_kwargs['img_size'] = self.glance_input_size
        print('Configs of glance_net:')
        print(glance_net_kwargs)
        self.glance_net = deit_small_patch16_224(**glance_net_kwargs)

        kwargs['depth'] = self.focus_net_depth
        kwargs['mlp_ratio'] = self.focus_net_mlp_ratio
        kwargs['drop_path_rate'] = self.focus_net_drop_path
        kwargs['img_size'] = self.focus_patch_size
        print('Configs of focus_net:')
        print(kwargs)
        self.focus_net = deit_small_patch16_224(**kwargs)

        print('multi_cls_drop_path:', self.multi_cls_drop_path)

        self.policy_net_classifier_list = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            self.glance_net.partial_create_block_func(
                                drop_path=self.multi_cls_drop_path
                                ) 
                                for __ in range(remaining_blocks)
                                ]
                                )
                    for _ in range(seq_len)
                    ]
            )
        self.policy_net_norm = nn.ModuleList(
                [partial(nn.LayerNorm, eps=1e-6)(self.glance_net.num_features) for _ in range(seq_len + 1)]
            )
        self.policy_net_avgpool = nn.AdaptiveAvgPool1d(1)
        self.policy_net_pre_logits = nn.Identity()
        self.policy_net_head = nn.ModuleList(
                [nn.Linear(self.glance_net.num_features, self.glance_net.num_classes) for _ in range(seq_len + 1)]
            )


        self.policy_net_patch_norm_list_stateV = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        
        
        self.policy_net_patch_norm_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_patch_stateV = policy_net_patch(
            feature_in_chans=self.feature_in_chans, 
            hidden_chans=policy_net_hidden_chans, 
            kernel_size=policy_net_kernel_size, 
            feature_size=self.glance_feature_size, 
            seq_len=self.seq_len, 
            is_policy_net=False
        )


        self.policy_net_patch = policy_net_patch(
            feature_in_chans=self.feature_in_chans, 
            hidden_chans=policy_net_hidden_chans, 
            kernel_size=policy_net_kernel_size, 
            feature_size=self.glance_feature_size, 
            seq_len=self.seq_len, 
            is_policy_net=True
        )
                

        n = self.recover_n
        offset_x = torch.arange(0, n) - (n - 1) // 2
        offset_y = torch.arange(0, n) - (n - 1) // 2
        offset_xy = torch.stack(torch.meshgrid([offset_x, offset_y], indexing='xy'))
        offset_xy = torch.flatten(offset_xy, -2).reshape(1, 2, 1, 1, n**2)
        self.register_buffer("offset_xy", offset_xy)

        self.policy_net_recover_reuseNorm_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover_reuseConvHead = nn.Conv2d(
            in_channels=self.feature_in_chans, 
            out_channels=self.feature_in_chans, 
            kernel_size=7, stride=1, padding=3, 
            groups=self.feature_in_chans,
            bias=False
            )

        self.policy_net_recover_reuseMLP_preLN_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover_reuseMLP = Mlp(
                    in_features=self.feature_in_chans,
                    hidden_features=self.feature_in_chans * 4,
                    act_layer=nn.GELU,
                    drop=0.
                )
        self.policy_net_recover_preLN_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover = Mlp(
                    in_features=self.feature_in_chans,
                    hidden_features=self.feature_in_chans * 4,
                    out_features=self.recover_n ** 2,
                    act_layer=nn.GELU,
                    drop=0.
            )


    def forward(self, imgs=None, seq_l=4, old_states=None, old_actions=None, batch_index=None, flag='forward_backbone', ppo_std_this_iter=None):

        if flag == 'forward_backbone':
            return self.forward_backbone(imgs, seq_l, ppo_std_this_iter=ppo_std_this_iter)
        elif flag == 'evaluate_policy_net':
            return self.evaluate_policy_net(seq_l=seq_l, old_states=old_states, old_actions=old_actions, batch_index=batch_index, ppo_std_this_iter=ppo_std_this_iter)
        # elif flag == 'infer_focus_net_only':
        #     return self.infer_focus_net_only(imgs)
        else:
            raise NotImplementedError

    def infer_focus_net_only(self, imgs):
        return self.focus_net(imgs, infer_focus_net_only=self.focus_net_reg_size)

    def evaluate_policy_net(self, seq_l=None, old_states=None, old_actions=None, batch_index=None, ppo_std_this_iter=None):
        logprobs = []
        state_values = []
        dist_entropy = []

        for focus_step_index in range(seq_l):

            input_feat = old_states[focus_step_index][batch_index]
            B, L, C = input_feat.shape

            with torch.cuda.amp.autocast(enabled=True):
                _state_values = self.policy_net_patch_stateV((
                    self.policy_net_patch_norm_list_stateV[focus_step_index](
                        input_feat.detach())
                        ).permute(0, 2, 1).reshape(B, C, int(L ** 0.5), int(L ** 0.5)).contiguous(),
                        focus_step_index
                    )
                state_values.append(_state_values)

                _actions = self.policy_net_patch((
                    self.policy_net_patch_norm_list[focus_step_index](
                        input_feat.detach())
                        ).permute(0, 2, 1).reshape(B, C, int(L ** 0.5), int(L ** 0.5)).contiguous(),
                        focus_step_index
                    )

                actions = _actions.clone()
                
                with torch.cuda.amp.autocast(enabled=False):
                    actions = actions.float()

                    action_var = torch.full((2,), ppo_std_this_iter, device=input_feat.device)
                    cov_mat = torch.diag(action_var)
                    dist = torch.distributions.multivariate_normal.MultivariateNormal(actions, scale_tril=cov_mat)

                    logprobs.append(dist.log_prob(old_actions[focus_step_index][batch_index]).unsqueeze(-1))
                    dist_entropy.append(dist.entropy().unsqueeze(-1))

        logprobs = torch.cat(logprobs, dim=1)
        state_values = torch.cat(state_values, dim=1)
        dist_entropy = torch.cat(dist_entropy, dim=1)

        return logprobs, state_values, dist_entropy
                

    def forward_backbone(self, imgs, seq_l=4, ppo_std_this_iter=None):

        # size: seq_l
        # x_glance, x, actions_logprobs, _state_values.squeeze(), pos_std, pos_mean
        expected_outputs = {
            'x_glance': [], 
            'x_focus': [], 
            'actions': [], 
            'actions_logprobs': [], 
            '_state_values': [], 
            'states': [],
            'pos_std': [], 
            'pos_mean': [],
            'outputs_reg_focus_net': None,
        }

        expected_outputs['outputs_reg_focus_net'] = self.infer_focus_net_only(imgs)

        x = torch.nn.functional.interpolate(
            imgs, size=(self.glance_input_size, self.glance_input_size), mode='bicubic'
            )

        global_features = self.glance_net(x, remaining_blocks = self.remaining_blocks)

        x_glance = self.policy_net_norm[0](
            self.glance_net(global_features, remaining_blocks = -self.remaining_blocks)
            )  # B L C
        x_glance = self.policy_net_avgpool(x_glance.transpose(1, 2))  # B C 1
        x_glance = torch.flatten(x_glance, 1)
        x_glance = self.policy_net_pre_logits(x_glance)
        x_glance = self.policy_net_head[0](x_glance)

        expected_outputs['x_glance'].append(x_glance)

        updated_features = global_features
        for focus_step_index in range(seq_l):


            B, L, C = updated_features.shape

            with torch.cuda.amp.autocast(enabled=True):
                expected_outputs['states'].append(updated_features.detach())
                _state_values = self.policy_net_patch_stateV((
                    self.policy_net_patch_norm_list_stateV[focus_step_index](
                        updated_features.detach())
                        ).permute(0, 2, 1).reshape(B, C, int(L ** 0.5), int(L ** 0.5)).contiguous(),
                        focus_step_index
                    )
                _actions = self.policy_net_patch((
                    self.policy_net_patch_norm_list[focus_step_index](
                        updated_features.detach())
                        ).permute(0, 2, 1).reshape(B, C, int(L ** 0.5), int(L ** 0.5)).contiguous(),
                        focus_step_index
                    )

                actions = _actions.clone()


                if self.training:
                    with torch.cuda.amp.autocast(enabled=False):
                        actions = actions.float()
                        action_var = torch.full((2,), ppo_std_this_iter, device=updated_features.device)
                        cov_mat = torch.diag(action_var)
                        dist = torch.distributions.multivariate_normal.MultivariateNormal(actions, scale_tril=cov_mat)
                        actions = dist.sample()
                        actions = torch.clamp(actions, min=0, max=1)
                        actions_logprobs = dist.log_prob(actions)
                else:
                    actions_logprobs = torch.ones(B, device=updated_features.device)
                    
                actions = actions.detach()
                expected_outputs['actions'].append(actions.detach())
                actions = torch.cat(
                    [actions, torch.ones_like(actions)], dim=1
                    )


                x_patch, reuse_feat_grid, grid = self.get_img_patches(imgs, actions, self.focus_patch_size, imgs.shape[-1])

                feat_to_reuse = self.policy_net_recover_reuseMLP(
                    self.policy_net_recover_reuseMLP_preLN_list[focus_step_index](
                        self.get_reuse_faet(
                            input_faet=updated_features,
                            reuse_feat_grid=reuse_feat_grid,
                            focus_step_index=focus_step_index
                            )
                            )
                            )

            
                pos_std, pos_mean = torch.std_mean(actions[:, 0:2], dim=0)

            expected_outputs['actions_logprobs'].append(actions_logprobs.unsqueeze(-1))
            expected_outputs['_state_values'].append(_state_values)
            expected_outputs['pos_std'].append(pos_std)
            expected_outputs['pos_mean'].append(pos_mean)

            local_features = self.focus_net(x_patch, remaining_blocks = 0, context_embed = feat_to_reuse)

            B, wHwW, C = local_features.shape
            assert wHwW == self.focus_feature_size ** 2

            with torch.cuda.amp.autocast(enabled=True):
                recover_policy_weight = self.policy_net_recover(
                    self.policy_net_recover_preLN_list[focus_step_index](local_features)
                    )

                neighbor_coords_index, outlier_mask = self.window_neighbor_grid(B, grid, self.focus_feature_size, self.glance_feature_size)

                recover_policy_weight = recover_policy_weight * outlier_mask

                relative_position_weight = torch.zeros(
                    (B, wHwW, self.glance_feature_size ** 2), device=recover_policy_weight.device, dtype=recover_policy_weight.dtype
                    ).scatter_add_(
                    dim=2, index=neighbor_coords_index, src=recover_policy_weight
                    ).permute(0, 2, 1)

        
                updated_features = relative_position_weight @ local_features + updated_features


            x = self.policy_net_norm[focus_step_index + 1](
                self.policy_net_classifier_list[focus_step_index](updated_features)
                )  # B L C
            x = self.policy_net_avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            x = self.policy_net_pre_logits(x)
            x = self.policy_net_head[focus_step_index + 1](x)
            expected_outputs['x_focus'].append(x)

        return expected_outputs
    

    def get_reuse_faet(self, input_faet, reuse_feat_grid, focus_step_index):
        B, L, C = input_faet.shape
        input_faet = self.policy_net_recover_reuseNorm_list[focus_step_index](input_faet)
        input_faet = input_faet.permute(0, 2, 1).reshape(B, C, int(L ** 0.5), int(L ** 0.5)).contiguous()
        input_faet = self.policy_net_recover_reuseConvHead(input_faet)

        feat_to_reuse = F.grid_sample(input_faet, reuse_feat_grid, mode='bilinear')

        feat_to_reuse = feat_to_reuse.reshape(B, C, self.focus_feature_size ** 2).permute(0, 2, 1).contiguous()

        return feat_to_reuse


    def window_neighbor_grid(self, B, sub_grid, window_size, feature_size):
        grid = sub_grid
        neighbor_coords = (grid + 0.5).int().unsqueeze(-1) + self.offset_xy
        
        neighbor_coords = neighbor_coords.reshape(B, 2, window_size * window_size * self.recover_n * self.recover_n)
        outlier_index = (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 0] <= (feature_size - 1)) & (neighbor_coords[:, 1] <= (feature_size - 1))
        outlier_mask = outlier_index.long().view(B, window_size * window_size, self.recover_n * self.recover_n)

        neighbor_coords = torch.clamp(
            neighbor_coords, min=0, max=(feature_size - 1)
            )

        neighbor_coords_index = neighbor_coords[:, 0, :] + neighbor_coords[:, 1, :] * feature_size
        
        return neighbor_coords_index.view(B, window_size * window_size, self.recover_n * self.recover_n), outlier_mask


    def get_img_patches(self, input_frames, actions, patch_size, image_size):
        batchsize = actions.size(0)
        theta = torch.zeros((batchsize, 2, 3), device=input_frames.device)
        
        patch_scale = torch.ones_like(actions[:, 2:4])

        patch_coordinate = (actions[:, :2] * (image_size - patch_size * patch_scale))
        x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size * patch_scale[:, 1], \
                         patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size * patch_scale[:, 0]

        theta[:, 0, 0], theta[:, 1, 1] = patch_size * patch_scale[:, 1] / image_size, patch_size * patch_scale[:, 0] / image_size
        theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

        grid = F.affine_grid(theta.float(), torch.Size((batchsize, 1, patch_size, patch_size)))
        patches = F.grid_sample(input_frames, grid, mode='bicubic')

        _grid = F.affine_grid(theta.float(), torch.Size(
            (batchsize, 1, self.focus_feature_size, self.focus_feature_size)))
        recover_feature_grid = ((_grid + 1) / 2 * (self.glance_feature_size - 1)).permute(0, 3, 1, 2)

        return patches, _grid, recover_feature_grid




def update_ckpt(checkpoint, new_size):

    model_ckp = checkpoint['model']
    model_ckp_new = {}

    for k in model_ckp.keys():
        if 'pos_embed' not in k:
            model_ckp_new[k] = model_ckp[k]
        if 'pos_embed' in k:
            print('bilinear interpolate & load:', k)
            
            temp_pb = model_ckp[k]
            pre_H_or_W = int(temp_pb.size(1) ** 0.5)
            num_embed_dim = temp_pb.size(2)
            new_H_or_W = new_size
            
            pos_embed_temp = temp_pb.transpose(1, 2).view(1, -1, pre_H_or_W, pre_H_or_W)
            pos_embed_temp = torch.nn.functional.interpolate(
                pos_embed_temp, size=(new_H_or_W, new_H_or_W), mode='bilinear', align_corners=False)
            pos_embed_temp = pos_embed_temp.flatten(2).transpose(1, 2)
            
            model_ckp_new[k] = pos_embed_temp

    return model_ckp_new



@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(depth=12, mlp_ratio=4, pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=depth, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


