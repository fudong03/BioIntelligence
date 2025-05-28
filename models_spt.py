import torch
import torch.nn as nn
from functools import partial

from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models import register_model

from attention import Block


__all__ = [
    'spt_tiny_embed192', 'spt_small_embed384', 'spt_base_embed768',
]

class SequenceProteinTransformer(nn.Module):

    def __init__(self, max_seq=512, num_classes=6, input_dim=20, embed_dim=192,
                 num_heads=3, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depth=12):

        super().__init__()
        self.num_classes = num_classes
        self.depths = depth

        self.proj = nn.Identity() if input_dim == embed_dim else nn.Linear(input_dim, embed_dim)

        self.max_seq = max_seq
        self.num_patches = self.max_seq + 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer)
            for _ in range(depth)])

        self.norm = norm_layer(embed_dim)

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # init weight
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, N = x.shape[0], x.shape[1]

        # optional: linear projection
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x + self.pos_embed[:, :(N + 1), :])

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        """
            x.shape = (batch_size, seq_len, input_dim) for classification
            x.shape = (batch_size, input_dim, width, height) for explanation, where seq_len = width * height
        :param x:
        :param explain:
        :return:
        """

        x = self.forward_features(x)
        x = x[:, 0]
        x = self.head(x)

        return x


@register_model
def spt_tiny_embed192(**kwargs):
    model = SequenceProteinTransformer(embed_dim=192, num_heads=4, mlp_ratio=4,
                                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depth=12, **kwargs)
    return model


@register_model
def spt_small_embed384(**kwargs):
    model = SequenceProteinTransformer(embed_dim=384, num_heads=6, mlp_ratio=4,
                                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depth=12, **kwargs)
    return model


@register_model
def spt_base_embed768(**kwargs):
    model = SequenceProteinTransformer(embed_dim=768, num_heads=12, mlp_ratio=4,
                                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depth=12, **kwargs)
    return model