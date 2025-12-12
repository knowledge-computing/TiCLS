"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from adet.utils.curve_utils import BezierSampler

class CtrlPointHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def run_inference_w_BART(
        self,
        model,
        vis_embeds,                   # (bs, n_q, L, D)  or  (bs*n_q, L, D)
        max_len: int = 27,
        return_log_probs: bool = False,
    ):
        """
        Returns:
            Tensor  (bs, n_q, max_len, vocab)   ←  n_pts == max_len
                    (log softmax if return_log_probs=True)
        """
        model.eval()
        pad, bos, eos = (
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
            model.config.eos_token_id,
        )

        if vis_embeds.dim() == 4:                       
            bs, n_q, L, D = vis_embeds.shape
            embeds_flat = vis_embeds.view(bs * n_q, L, D)
        elif vis_embeds.dim() == 3:                     
            bs, n_q = vis_embeds.size(0), 1
            embeds_flat = vis_embeds
        else:
            raise ValueError("vis_embeds must be 3D or 4D tensor")

    
        flat_logits = []                                
        for enc in embeds_flat:                         
            enc = enc.unsqueeze(0)                      
            gen = torch.full((1, 1), bos,
                            dtype=torch.long, device=vis_embeds.device)

            while gen.size(1) < max_len:
                logits = model(input_ids=gen,
                            encoder_hidden_states=enc)  # (1, T, vocab)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, next_token], dim=1)
                if next_token.item() == eos:
                    break

            flat_logits.append(logits.squeeze(0))       # (T, vocab)

        vocab = flat_logits[0].size(-1)

        padded = torch.zeros(len(flat_logits), max_len, vocab,
                            device=vis_embeds.device)
        for i, l in enumerate(flat_logits):
            valid_len = min(l.size(0), max_len)
            padded[i, :valid_len] = l[:valid_len]

        logits_reshaped = padded.view(bs, n_q, max_len, vocab)

        if return_log_probs:
            logits_reshaped = torch.log_softmax(logits_reshaped, dim=-1)

        return logits_reshaped

    def forward(self, outputs, targets, lm_model=None):

        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            torch.cuda.empty_cache()
            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)
            # tgt_ids = torch.cat([v["labels"] for v in targets])

            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())

            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)
            
            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu()

            # indices = [linear_sum_assignment(
            #     c[i] + self.text_weight * texts_cost_list[i]
            # ) for i, c in enumerate(C.split(sizes, -1))]

            indices = [linear_sum_assignment(
                c[i]
            ) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BezierHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)

            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(start_dim=-2),
                p=1
            )

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["beziers"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                  coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                  num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                  focal_alpha=cfg.FOCAL_ALPHA,
                                  focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA)