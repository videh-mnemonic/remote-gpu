"""Bounded synthetic-data training for diverse Hugging Face model families.

The harness imports exact locally mounted Transformers/Diffusers source trees,
constructs small random models, and performs real forward/backward/AdamW steps.
It downloads neither pretrained weights nor datasets.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import time
import types
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class Workload:
    model: torch.nn.Module
    inputs: dict
    loss: Callable[[object], torch.Tensor]
    feature: str


def tokens(shape: tuple[int, ...], vocab_size: int) -> torch.Tensor:
    return torch.randint(3, vocab_size, shape, dtype=torch.long)


def build_gpt2() -> Workload:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=4096, n_positions=128, n_embd=256, n_layer=4, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=1, eos_token_id=2,
    )
    ids = tokens((4, 128), config.vocab_size)
    return Workload(GPT2LMHeadModel(config), {"input_ids": ids, "labels": ids.clone()}, lambda out: out.loss,
                    "decoder-only transformer; Conv1D projections; causal loss")


def build_llama() -> Workload:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=4096, hidden_size=256, intermediate_size=768,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.0, use_cache=False,
    )
    ids = tokens((4, 128), config.vocab_size)
    return Workload(LlamaForCausalLM(config), {"input_ids": ids, "labels": ids.clone()}, lambda out: out.loss,
                    "RoPE; grouped-query SDPA; RMSNorm; gated MLP")


def build_t5() -> Workload:
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(
        vocab_size=4096, d_model=256, d_kv=64, d_ff=512,
        num_layers=3, num_decoder_layers=3, num_heads=4,
        dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0,
    )
    return Workload(
        T5ForConditionalGeneration(config),
        {"input_ids": tokens((4, 96), config.vocab_size), "labels": tokens((4, 64), config.vocab_size)},
        lambda out: out.loss,
        "encoder-decoder; cross-attention; relative position bias",
    )


def build_mixtral() -> Workload:
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        vocab_size=4096, hidden_size=256, intermediate_size=512,
        num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=256,
        router_jitter_noise=0.0, attention_dropout=0.0, use_cache=False,
    )
    ids = tokens((4, 128), config.vocab_size)
    return Workload(MixtralForCausalLM(config), {"input_ids": ids, "labels": ids.clone()}, lambda out: out.loss,
                    "sparse mixture-of-experts routing; grouped-query attention")


def build_mamba2() -> Workload:
    from transformers import Mamba2Config, Mamba2ForCausalLM

    config = Mamba2Config(
        vocab_size=4096, hidden_size=256, expand=2, num_heads=8, head_dim=64,
        state_size=32, num_hidden_layers=4, n_groups=4, chunk_size=64,
        use_cache=False,
    )
    ids = tokens((4, 128), config.vocab_size)
    return Workload(Mamba2ForCausalLM(config), {"input_ids": ids, "labels": ids.clone()}, lambda out: out.loss,
                    "state-space scan; grouped evolution matrices; causal convolution")


def build_vit() -> Workload:
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(
        image_size=64, patch_size=8, num_channels=3, hidden_size=256,
        num_hidden_layers=4, num_attention_heads=4, intermediate_size=512,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, num_labels=100,
    )
    return Workload(
        ViTForImageClassification(config),
        {"pixel_values": torch.randn(8, 3, 64, 64), "labels": torch.randint(0, 100, (8,))},
        lambda out: out.loss,
        "patch embedding; vision attention; classification loss",
    )


def build_convnext() -> Workload:
    from transformers import ConvNextConfig, ConvNextForImageClassification

    config = ConvNextConfig(
        num_channels=3, patch_size=4, hidden_sizes=[64, 128, 256], depths=[2, 2, 2],
        num_stages=3, drop_path_rate=0.0, hidden_act="gelu", num_labels=100,
    )
    return Workload(
        ConvNextForImageClassification(config),
        {"pixel_values": torch.randn(8, 3, 64, 64), "labels": torch.randint(0, 100, (8,))},
        lambda out: out.loss,
        "depthwise convolution; channels-last LayerNorm; classification",
    )


def build_swin() -> Workload:
    from transformers import SwinConfig, SwinForImageClassification

    config = SwinConfig(
        image_size=64, patch_size=4, num_channels=3, embed_dim=32,
        depths=[2, 2, 2], num_heads=[2, 4, 8], window_size=4,
        mlp_ratio=3.0, drop_path_rate=0.0, num_labels=100,
    )
    return Workload(
        SwinForImageClassification(config),
        {"pixel_values": torch.randn(8, 3, 64, 64), "labels": torch.randint(0, 100, (8,))},
        lambda out: out.loss,
        "hierarchical vision; shifted-window attention; patch merging",
    )


def build_segformer() -> Workload:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    config = SegformerConfig(
        num_channels=3, num_encoder_blocks=3, depths=[2, 2, 2],
        sr_ratios=[8, 4, 2], hidden_sizes=[32, 64, 128],
        patch_sizes=[7, 3, 3], strides=[4, 2, 2],
        num_attention_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
        drop_path_rate=0.0, decoder_hidden_size=64, num_labels=12,
    )
    return Workload(
        SegformerForSemanticSegmentation(config),
        {"pixel_values": torch.randn(4, 3, 64, 64),
         "labels": torch.randint(0, 12, (4, 64, 64))},
        lambda out: out.loss,
        "hierarchical spatial-reduction attention; all-MLP segmentation decoder",
    )


def build_wav2vec2() -> Workload:
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

    config = Wav2Vec2Config(
        vocab_size=32, hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
        intermediate_size=256, conv_dim=(32, 32, 64), conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3), num_conv_pos_embedding_groups=8,
        num_conv_pos_embeddings=32, classifier_proj_size=64, num_labels=12,
        hidden_dropout=0.0, attention_dropout=0.0, feat_proj_dropout=0.0,
        final_dropout=0.0, layerdrop=0.0, mask_time_prob=0.0,
    )
    return Workload(
        Wav2Vec2ForSequenceClassification(config),
        {"input_values": torch.randn(4, 16000), "labels": torch.randint(0, 12, (4,))},
        lambda out: out.loss,
        "raw-audio convolutional frontend; transformer encoder; temporal reduction",
    )


def build_whisper() -> Workload:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=4096, num_mel_bins=80, d_model=192,
        encoder_layers=3, decoder_layers=3, encoder_attention_heads=3,
        decoder_attention_heads=3, encoder_ffn_dim=384, decoder_ffn_dim=384,
        max_source_positions=128, max_target_positions=64,
        decoder_start_token_id=1, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, use_cache=False,
    )
    return Workload(
        WhisperForConditionalGeneration(config),
        {"input_features": torch.randn(4, 80, 256),
         "labels": tokens((4, 48), config.vocab_size)},
        lambda out: out.loss,
        "log-Mel convolutional frontend; audio encoder-decoder; cross-attention",
    )


def build_clip() -> Workload:
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

    text = CLIPTextConfig(
        vocab_size=4096, hidden_size=256, intermediate_size=512,
        num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=64,
        attention_dropout=0.0, bos_token_id=1, eos_token_id=2,
    )
    vision = CLIPVisionConfig(
        image_size=64, patch_size=8, num_channels=3, hidden_size=256,
        intermediate_size=512, num_hidden_layers=3, num_attention_heads=4,
        attention_dropout=0.0,
    )
    config = CLIPConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=128,
    )
    return Workload(
        CLIPModel(config),
        {"input_ids": tokens((8, 64), text.vocab_size), "pixel_values": torch.randn(8, 3, 64, 64),
         "return_loss": True},
        lambda out: out.loss,
        "dual text/vision encoders; normalized embeddings; contrastive loss",
    )


def build_detr() -> Workload:
    from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig

    backbone = ResNetConfig(
        num_channels=3, embedding_size=32, hidden_sizes=[32, 64, 128, 256],
        depths=[1, 1, 1, 1], out_features=["stage4"],
    )
    config = DetrConfig(
        backbone=None, use_timm_backbone=False, use_pretrained_backbone=False,
        backbone_config=backbone,
        d_model=128, encoder_layers=2, decoder_layers=2,
        encoder_ffn_dim=256, decoder_ffn_dim=256, encoder_attention_heads=4,
        decoder_attention_heads=4, num_queries=20, num_labels=10,
        dropout=0.0, attention_dropout=0.0, auxiliary_loss=False,
    )
    labels = [
        {"class_labels": torch.randint(0, 10, (3,)), "boxes": torch.rand(3, 4)}
        for _ in range(4)
    ]
    return Workload(
        DetrForObjectDetection(config),
        {"pixel_values": torch.randn(4, 3, 64, 64), "labels": labels},
        lambda out: out.loss,
        "CNN backbone; encoder-decoder; Hungarian matching; box/GIoU losses",
    )


def build_videomae() -> Workload:
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    config = VideoMAEConfig(
        image_size=64, patch_size=8, num_channels=3, num_frames=8,
        tubelet_size=2, hidden_size=192, num_hidden_layers=3,
        num_attention_heads=3, intermediate_size=384, num_labels=20,
    )
    return Workload(
        VideoMAEForVideoClassification(config),
        {"pixel_values": torch.randn(4, 8, 3, 64, 64),
         "labels": torch.randint(0, 20, (4,))},
        lambda out: out.loss,
        "video tubelet embedding; spatiotemporal attention; mean pooling",
    )


def build_diffusers_unet() -> Workload:
    from diffusers import UNet2DModel

    model = UNet2DModel(
        sample_size=64, in_channels=3, out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(64, 128), layers_per_block=1,
        attention_head_dim=8, norm_num_groups=8,
    )
    target = torch.randn(4, 3, 64, 64)
    model.register_buffer("_rgpu_training_target", target)
    return Workload(
        model,
        {"sample": torch.randn(4, 3, 64, 64), "timestep": torch.randint(0, 1000, (4,))},
        lambda out: F.mse_loss(out.sample, model._rgpu_training_target),
        "diffusion U-Net; residual convolution; spatial attention; up/downsampling",
    )


def build_diffusers_vae() -> Workload:
    from diffusers import AutoencoderKL

    model = AutoencoderKL(
        sample_size=64, in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128), layers_per_block=1, latent_channels=4,
        norm_num_groups=8, mid_block_add_attention=True,
    )
    target = torch.randn(4, 3, 64, 64)
    model.register_buffer("_rgpu_training_target", target)
    return Workload(
        model,
        {"sample": target.clone(), "sample_posterior": True},
        lambda out: F.mse_loss(out.sample, model._rgpu_training_target),
        "variational encode/reparameterize/decode; convolution; spatial attention",
    )


BUILDERS = {
    "gpt2": build_gpt2,
    "llama": build_llama,
    "t5": build_t5,
    "mixtral": build_mixtral,
    "mamba2": build_mamba2,
    "vit": build_vit,
    "convnext": build_convnext,
    "swin": build_swin,
    "segformer": build_segformer,
    "wav2vec2": build_wav2vec2,
    "whisper": build_whisper,
    "clip": build_clip,
    "detr": build_detr,
    "videomae": build_videomae,
    "diffusers_unet": build_diffusers_unet,
    "diffusers_vae": build_diffusers_vae,
}


def move(value, device: torch.device, dtype: torch.dtype | None = None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else None)
    if isinstance(value, dict):
        return {key: move(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device, dtype) for item in value)
    return value


def pin_static_host_tensor_attributes(model: torch.nn.Module) -> int:
    """Pin direct tensor attributes that are not parameters or buffers."""
    pinned = 0
    for module in model.modules():
        parameters = module.__dict__.get("_parameters", {})
        buffers = module.__dict__.get("_buffers", {})
        for name, value in list(module.__dict__.items()):
            if name in parameters or name in buffers:
                continue
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and not value.is_pinned()
            ):
                setattr(module, name, value.pin_memory())
                pinned += 1
    return pinned


def cache_static_host_tensor_attributes(
    model: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> int:
    """Materialize non-buffer CPU tensor constants once on the selected GPU."""
    cached = 0
    for module in model.modules():
        parameters = module.__dict__.get("_parameters", {})
        buffers = module.__dict__.get("_buffers", {})
        for name, value in list(module.__dict__.items()):
            if name in parameters or name in buffers:
                continue
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                target_dtype = dtype if value.is_floating_point() else None
                setattr(module, name, value.to(device=device, dtype=target_dtype))
                cached += 1
    return cached


def enable_detr_bulk_matcher_indices() -> None:
    """Pack SciPy's CPU assignments into one host-to-device transfer.

    Upstream DETR returns two CPU index tensors per batch element.  PyTorch then
    copies the same tiny indices to CUDA repeatedly as each loss indexes remote
    tensors.  Packing them once preserves the assignments while exposing the
    transfer-boundary optimization a framework integration can apply.
    """
    from transformers.loss.loss_for_object_detection import HungarianMatcher

    original_forward = HungarianMatcher.forward

    def bulk_forward(self, outputs, targets):
        indices = original_forward(self, outputs, targets)
        lengths = [source.numel() for source, _ in indices]
        if not lengths or sum(lengths) == 0:
            return indices
        packed_cpu = torch.stack(
            (
                torch.cat([source for source, _ in indices]),
                torch.cat([target for _, target in indices]),
            ),
            dim=0,
        )
        packed = packed_cpu.to(device=outputs["logits"].device)
        sources = packed[0].split(lengths)
        targets = packed[1].split(lengths)
        return list(zip(sources, targets))

    HungarianMatcher.forward = bulk_forward


def enable_detr_host_control_regions(model: torch.nn.Module) -> None:
    """Reduce exact, single-device DETR host-control boundaries.

    This adapter keeps SciPy and Python on the client.  It caches the otherwise
    per-forward criterion, keeps the already-known box count as a Python value,
    and combines each pair of GIoU validation predicates into one transfer.
    Distributed reduction deliberately falls back to upstream semantics.
    """
    from transformers.loss import loss_for_object_detection as detr_loss
    from transformers.models.detr.modeling_detr import DetrEncoderLayer

    # Express the encoder's data-dependent finite-value recovery as tensor
    # selection.  This is value-equivalent to the Python branch (including its
    # NaN behavior) but lets AOT compilation keep the region on the GPU.
    def tensorized_encoder_forward(
        self,
        hidden_states,
        attention_mask,
        spatial_position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=spatial_position_embeddings,
            **kwargs,
        )
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)
        if self.training:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            recovered = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
            hidden_states = torch.where(
                torch.isfinite(hidden_states).all(), hidden_states, recovered
            )
        return hidden_states

    DetrEncoderLayer.forward = tensorized_encoder_forward

    def combined_validation_giou(boxes1, boxes2):
        valid = torch.stack(
            (
                (boxes1[:, 2:] >= boxes1[:, :2]).all(),
                (boxes2[:, 2:] >= boxes2[:, :2]).all(),
            )
        ).cpu()
        if not bool(valid[0]):
            raise ValueError(
                f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}"
            )
        if not bool(valid[1]):
            raise ValueError(
                f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}"
            )
        iou, union = detr_loss.box_iou(boxes1, boxes2)
        top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        width_height = (bottom_right - top_left).clamp(min=0)
        area = width_height[:, :, 0] * width_height[:, :, 1]
        return iou - (area - union) / area

    detr_loss.generalized_box_iou = combined_validation_giou

    matcher = detr_loss.HungarianMatcher(
        class_cost=model.config.class_cost,
        bbox_cost=model.config.bbox_cost,
        giou_cost=model.config.giou_cost,
    )
    criterion = detr_loss.ImageLoss(
        matcher=matcher,
        num_classes=model.config.num_labels,
        eos_coef=model.config.eos_coefficient,
        losses=["labels", "boxes", "cardinality"],
    )
    original_criterion_forward = criterion.forward

    def single_device_forward(self, outputs, targets):
        if torch.distributed.is_initialized():
            return original_criterion_forward(outputs, targets)
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "auxiliary_outputs"
        }
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = max(float(sum(len(t["class_labels"]) for t in targets)), 1.0)
        losses = {}
        for loss_name in self.losses:
            losses.update(
                self.get_loss(loss_name, outputs, targets, indices, num_boxes)
            )
        if "auxiliary_outputs" in outputs:
            for index, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                auxiliary_indices = self.matcher(auxiliary_outputs, targets)
                for loss_name in self.losses:
                    if loss_name == "masks":
                        continue
                    values = self.get_loss(
                        loss_name,
                        auxiliary_outputs,
                        targets,
                        auxiliary_indices,
                        num_boxes,
                    )
                    losses.update(
                        {key + f"_{index}": value for key, value in values.items()}
                    )
        return losses

    criterion.forward = types.MethodType(single_device_forward, criterion)
    criterion_ready = False

    def cached_loss(
        logits,
        labels,
        device,
        pred_boxes,
        config,
        outputs_class=None,
        outputs_coord=None,
        **kwargs,
    ):
        nonlocal criterion_ready
        if not criterion_ready:
            criterion.to(device)
            criterion_ready = True
        outputs_loss = {"logits": logits, "pred_boxes": pred_boxes}
        auxiliary_outputs = None
        if config.auxiliary_loss:
            auxiliary_outputs = detr_loss._set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        loss_dict = criterion(outputs_loss, labels)
        weights = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
        }
        if config.auxiliary_loss:
            for index in range(config.decoder_layers - 1):
                weights.update(
                    {
                        key + f"_{index}": value
                        for key, value in list(weights.items())
                        if "_" not in key.rsplit("_", 1)[-1]
                    }
                )
        loss = sum(loss_dict[key] * weights[key] for key in loss_dict if key in weights)
        return loss, loss_dict, auxiliary_outputs

    model.loss_function = cached_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA ordinal to train on (use 1 for the first remote GPU in mixed mode)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-optimizer",
        action="store_true",
        help="AOT-compile the optimizer step in addition to the selected model region",
    )
    parser.add_argument(
        "--whole-step-graph",
        action="store_true",
        help="capture forward, backward, and optimizer as one static CUDA graph",
    )
    parser.add_argument(
        "--pin-static-host-tensors",
        action="store_true",
        help="pin non-buffer CPU tensor attributes so static transfers can be captured",
    )
    parser.add_argument(
        "--cache-static-host-tensors",
        action="store_true",
        help="materialize non-buffer CPU tensor constants once on the GPU",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--compile-region",
        choices=("auto", "full", "core", "dense"),
        default="auto",
        help=(
            "compile the full model or safe tensor-only regions; auto isolates "
            "DETR's host matcher and Mixtral's dynamic expert routing"
        ),
    )
    parser.add_argument("--cuda-events", action="store_true")
    parser.add_argument(
        "--reset-rpc-stats-after-warmup",
        action="store_true",
        help="reset LUPINE's optional RPC counters immediately before measured iterations",
    )
    parser.add_argument(
        "--profile-trace",
        help="write a CPU-side PyTorch profiler trace for the first measured step",
    )
    parser.add_argument(
        "--detr-bulk-matcher-indices",
        action="store_true",
        help="batch DETR's CPU Hungarian indices into one CUDA transfer",
    )
    parser.add_argument(
        "--detr-host-control-regions",
        action="store_true",
        help="apply exact single-device DETR host-control region optimizations",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--blas",
        choices=("default", "cublas", "cublaslt"),
        default="default",
        help="diagnostically select PyTorch's CUDA BLAS backend",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=("default", "grouped_mm", "batched_mm", "eager"),
        default="default",
        help="select a registered Transformers MoE implementation",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.cuda.set_device(args.device)
    if args.blas != "default":
        torch.backends.cuda.preferred_blas_library(args.blas)
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.detr_bulk_matcher_indices or args.detr_host_control_regions:
        if args.model != "detr":
            raise ValueError("--detr-bulk-matcher-indices only applies to DETR")
        enable_detr_bulk_matcher_indices()
    workload = BUILDERS[args.model]()
    if (
        args.whole_step_graph
        and args.model == "mixtral"
        and args.experts_implementation == "default"
    ):
        # The default grouped-MM path rebuilds CPU offsets every forward.
        # Batched-MM is the upstream graph-safe implementation and preserves
        # the same routed-expert computation without host transfers.
        args.experts_implementation = "batched_mm"
    if args.experts_implementation != "default":
        if args.model != "mixtral":
            raise ValueError("--experts-implementation currently applies to Mixtral")
        workload.model.set_experts_implementation(args.experts_implementation)
    pinned_static_tensors = (
        pin_static_host_tensor_attributes(workload.model)
        if args.pin_static_host_tensors
        else 0
    )
    if args.detr_host_control_regions:
        enable_detr_host_control_regions(workload.model)
    workload.model.train().to(device=device, dtype=dtype)
    cached_static_tensors = (
        cache_static_host_tensor_attributes(workload.model, device, dtype)
        if args.cache_static_host_tensors or args.whole_step_graph
        else 0
    )
    inputs = move(workload.inputs, device, dtype)
    compile_region = None
    model = workload.model
    if args.compile:
        compile_region = (
            "core"
            if args.compile_region == "auto" and args.model == "detr"
            else "dense"
            if args.compile_region == "auto" and args.model == "mixtral"
            else "full" if args.compile_region == "auto" else args.compile_region
        )
        compile_options = {
            "dynamic": False,
            "mode": None if args.compile_mode == "default" else args.compile_mode,
        }
        if compile_region == "core":
            if args.model != "detr":
                raise ValueError("--compile-region core is currently defined for DETR")
            # DETR's backbone/transformer is a pure tensor region.  Its outer
            # training wrapper includes SciPy's CPU Hungarian matcher and
            # variable-length Python label structures, which must stay eager.
            workload.model.model = torch.compile(
                workload.model.model, **compile_options
            )
        elif compile_region == "dense":
            if args.model != "mixtral":
                raise ValueError(
                    "--compile-region dense is currently defined for Mixtral"
                )
            # Keep data-dependent sparse expert dispatch eager. Compiling that
            # path in fp32 selects Transformers' fallback grouped-MM operator,
            # whose offset conversion synchronizes GPU data to the CPU. The
            # attention regions are static and safe to capture independently.
            for layer in workload.model.model.layers:
                layer.self_attn = torch.compile(layer.self_attn, **compile_options)
        else:
            model = torch.compile(workload.model, **compile_options)
    optimizer = torch.optim.AdamW(
        workload.model.parameters(),
        lr=1e-4,
        foreach=True,
        capturable=args.whole_step_graph,
    )
    optimizer_step = optimizer.step
    if args.compile_optimizer:
        if not args.compile:
            raise ValueError("--compile-optimizer requires --compile")
        optimizer_step = torch.compile(
            optimizer.step,
            dynamic=False,
            mode=None if args.compile_mode == "default" else args.compile_mode,
        )
    if args.whole_step_graph and args.compile_mode != "default":
        raise ValueError(
            "--whole-step-graph requires --compile-mode default"
        )

    samples = []
    losses = []
    profiler = None
    whole_step_graph = None
    graph_loss = None
    process_started = time.perf_counter()
    for iteration in range(args.warmup + args.iterations):
        if args.whole_step_graph and iteration < args.warmup:
            # Whole-step capture owns its warm-up stream below. Running the
            # ordinary eager path first creates persistent AccumulateGrad
            # nodes on the legacy stream, which CUDA correctly refuses to
            # make dependent on a blocking capture stream.
            continue
        if iteration == args.warmup and args.reset_rpc_stats_after_warmup:
            try:
                ctypes.CDLL("libcuda.so.1").lupine_rpc_stats_reset()
            except (OSError, AttributeError) as error:
                raise RuntimeError(
                    "the active CUDA library does not expose lupine_rpc_stats_reset"
                ) from error
        if iteration == args.warmup and args.profile_trace:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
            )
            profiler.start()
        if iteration == args.warmup and args.whole_step_graph:
            # Gradients must already exist so zeroing them is a captured device
            # operation on every replay, rather than a one-time Python pointer
            # update during capture.
            if "output" in locals():
                del output
            if "loss" in locals():
                del loss
            gc.collect()
            optimizer.zero_grad(set_to_none=True)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=False)
                    capture_output = model(**inputs)
                    capture_loss = workload.loss(capture_output)
                    capture_loss.backward()
                    optimizer_step()
                    del capture_output, capture_loss
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            # Release the final warm-up autograd frame so its temporary blocks
            # are available to the graph-private allocator pool. This is
            # especially important when the CUDA driver is routed remotely,
            # where a cache miss would otherwise become a capture-illegal
            # legacy allocation on the server.
            gc.collect()
            optimizer.zero_grad(set_to_none=False)
            whole_step_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(whole_step_graph, stream=capture_stream):
                graph_output = model(**inputs)
                graph_loss = workload.loss(graph_output)
                graph_loss.backward()
                optimizer_step()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
        if whole_step_graph is None:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True) if args.cuda_events else None
        end_event = torch.cuda.Event(enable_timing=True) if args.cuda_events else None
        if start_event is not None:
            start_event.record()
        if args.compile and args.compile_mode == "reduce-overhead":
            # This training harness synchronizes every iteration, so its step
            # boundary is explicit. Tell CUDAGraph Trees that outputs retained
            # by the previous backward pass no longer belong to this step.
            torch.compiler.cudagraph_mark_step_begin()
        if whole_step_graph is not None:
            whole_step_graph.replay()
            loss = graph_loss
        else:
            output = model(**inputs)
            loss = workload.loss(output)
            loss.backward()
            optimizer_step()
        if end_event is not None:
            end_event.record()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        gpu_ms = (
            start_event.elapsed_time(end_event)
            if start_event is not None and end_event is not None else None
        )
        if iteration >= args.warmup:
            sample = {"wall_ms": wall_seconds * 1000}
            if gpu_ms is not None:
                sample["gpu_ms"] = gpu_ms
            samples.append(sample)
            losses.append(float(loss.detach().cpu()))
        if iteration == args.warmup and profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(args.profile_trace)

    payload = {
        "model": args.model,
        "feature": workload.feature,
        "status": "pass",
        "compiled": args.compile,
        "compiled_optimizer": args.compile_optimizer,
        "whole_step_graph": args.whole_step_graph,
        "pinned_static_host_tensors": pinned_static_tensors,
        "cached_static_host_tensors": cached_static_tensors,
        "compile_mode": args.compile_mode if args.compile else None,
        "compile_region": compile_region,
        "cuda_events": args.cuda_events,
        "dtype": args.dtype,
        "experts_implementation": args.experts_implementation,
        "detr_bulk_matcher_indices": args.detr_bulk_matcher_indices,
        "detr_host_control_regions": args.detr_host_control_regions,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": sum(parameter.numel() for parameter in workload.model.parameters()),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "diffusers": (
            __import__("diffusers").__version__
            if args.model.startswith("diffusers_") else None
        ),
        "device": torch.cuda.get_device_name(),
        "median_wall_ms": statistics.median(sample["wall_ms"] for sample in samples),
        "median_gpu_ms": (
            statistics.median(sample["gpu_ms"] for sample in samples)
            if args.cuda_events else None
        ),
        "process_training_seconds": time.perf_counter() - process_started,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "samples": samples,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
