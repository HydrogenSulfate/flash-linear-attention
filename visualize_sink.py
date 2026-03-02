"""
可视化脚本：检测 Qwen3_5 模型中的三类 Outlier 现象及 Norm Weight 与 Residual Sink 的关联
  1. Attention Sink      —— 少数 token 持续接收大量注意力权重（全注意力层）
  2. Massive Activation  —— 少数 token 的残差流 L2 范数异常大
  3. Residual Sink       —— 少数固定维度在大多数 token 上持续拥有极大激活值
  4. Norm Weight 分析    —— RMSNorm 可学习权重的 outlier 分布，及其与 Residual Sink 的对齐关系
     ④-A  norm_weight_heatmap_input.png / norm_weight_heatmap_post_attn.png
     ④-B  norm_weight_distribution.png
     ④-C  norm_weight_alignment.png
     ④-D  norm_weight_rescaling.png

依赖：pip install scipy

用法：
    python visualize_outliers.py --model_path /path/to/your/model [--seq_len 512] [--top_k_dims 16]

输出图片（保存在 --output_dir，默认当前目录）：
    attention_sink.png
    massive_activation.png
    residual_sink.png
    norm_weight_heatmap_input.png
    norm_weight_heatmap_post_attn.png
    norm_weight_distribution.png
    norm_weight_alignment.png
    norm_weight_rescaling.png
"""

from fla.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from fla.models.qwen3_5 import Qwen3_5ForCausalLM
from transformers import AutoTokenizer
from torch import nn
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

# ------------------------------------------------------------------ #
#  导入模型                                                            #
# ------------------------------------------------------------------ #

# ================================================================== #
#  1. 全局存储                                                         #
# ================================================================== #

# attn_weights_by_layer[layer_idx] = Tensor(n_heads, seq_len, seq_len)  float32 cpu
attn_weights_by_layer: dict = {}

# residual_stream_by_layer[layer_idx] = Tensor(seq_len, hidden_size)    float32 cpu
#   捕获每个 DecoderLayer「输出」的完整残差流（已含两次残差加法）
residual_stream_by_layer: dict = {}

# ================================================================== #
#  2. Hook 函数                                                        #
# ================================================================== #


def make_attn_output_hook(layer_idx: int):
    """
    挂在 Qwen3_5Attention 上，通过 forward_hook 的 output 拿 attn_weights。
    Qwen3_5Attention.forward 返回 (attn_output, attn_weights)。

    ⚠️  必须以 attn_implementation="eager" 加载模型，
        flash / sdpa 路径下 attn_weights 为 None。

    ⚠️  原脚本用 monkey-patch 替换模块级 eager_attention_forward 变量，
        但 Qwen3_5Attention.forward 内部是通过
        ALL_ATTENTION_FUNCTIONS.get_interface(...) 动态查找函数，
        并非直接引用该变量，因此 monkey-patch 对它完全无效。
        正确做法是直接在 Qwen3_5Attention 上注册 forward_hook。
    """
    def hook(module: nn.Module, inputs, output):
        attn_weights = output[1]   # (batch, heads, q_len, kv_len) or None
        if attn_weights is not None:
            attn_weights_by_layer[layer_idx] = attn_weights[0].detach().cpu().float()
    return hook


def make_decoder_output_hook(layer_idx: int):
    """
    挂在 Qwen3_5DecoderLayer 上，捕获加完两次残差后的完整 hidden states。
    DecoderLayer.forward 直接返回 Tensor(batch, seq_len, hidden_size)。

    ⚠️  原脚本把 hook 挂在 post_attention_layernorm 上，捕获的只是
        "第一次残差加法后、进入 MLP 前" 的状态，并不是完整残差流，
        也不适合做 Residual Sink 分析（需要看完整残差流的维度分布）。
    """
    def hook(module: nn.Module, inputs, output):
        hs = output if isinstance(output, torch.Tensor) else output[0]
        residual_stream_by_layer[layer_idx] = hs[0].detach().cpu().float()
    return hook


# ================================================================== #
#  3. 模型加载与 Hook 注入                                             #
# ================================================================== #

def load_model_and_register_hooks(model_path: str):
    print(f"[1/3] 加载模型（强制 eager attention）：{model_path}")

    # ⚠️  原脚本在这里直接 from_pretrained，没有指定 attn_implementation，
    #     默认会用 flash/sdpa，导致 attn_weights 全部为 None，
    #     Attention Sink 图完全空白。
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",   # ← 关键：必须 eager
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   已加载至 {device}，参数量 ≈ {n_params:.1f}M")

    print("[2/3] 注入 Forward Hooks …")
    for layer_idx, decoder_layer in enumerate(model.model.layers):

        # ── Attention Sink：只给全注意力层挂 ──────────────────────
        if hasattr(decoder_layer, "self_attn") and isinstance(
            decoder_layer.self_attn, Qwen3_5Attention
        ):
            decoder_layer.self_attn.register_forward_hook(
                make_attn_output_hook(layer_idx)
            )

        # ── Massive Activation / Residual Sink：给所有层挂 ─────────
        decoder_layer.register_forward_hook(
            make_decoder_output_hook(layer_idx)
        )

    return model, device


# ================================================================== #
#  4. 推理                                                             #
# ================================================================== #

SAMPLE_TEXT = """The first thing people often misunderstand about complexity is that it rarely announces itself with noise. More often, it emerges quietly, through the accumulation of small decisions that appear harmless in isolation but become deeply influential when composed together. A single assumption, left unexamined, forms the seed of a system; repeated often enough, that system begins to feel natural, inevitable, and even necessary. Yet nothing about it was inevitable. It was assembled, piece by piece, by individuals acting within constraints they only partially understood.

Consider how any structured environment evolves. At the beginning there is openness, a wide field of possible actions. Early participants improvise. They solve immediate problems using whatever tools are available, guided more by intuition than by doctrine. Their solutions are pragmatic rather than elegant, local rather than global. However, each solution leaves a trace. That trace becomes a precedent, and precedent gradually transforms into expectation. What was once an improvisation becomes a rule.

Over time, rules interact. Interactions create patterns. Patterns produce feedback. Feedback stabilizes some behaviors while suppressing others. Eventually, the environment acquires a form of memory—not a conscious memory, but a structural one. It “remembers” through reinforcement. Paths that are used frequently become easier to traverse; paths that are neglected disappear. In this way, history reshapes possibility.

This phenomenon is not limited to institutions or technologies. It also characterizes intellectual work. When a researcher approaches a difficult question, the first framework they adopt is rarely perfect. It is simply the one that makes the problem tractable. Yet once the framework is chosen, it begins to guide perception itself. Certain variables appear important because the framework can measure them. Others fade into the background because they resist quantification. The tools of analysis quietly influence what is considered real.

There is a subtle tension between clarity and reduction. To understand something, we must simplify it. But every simplification conceals relationships that may later prove essential. The challenge, therefore, is not to avoid simplification—that would make understanding impossible—but to remain aware that every model is provisional. A model is a lens, not a mirror.

When systems grow large enough, they exhibit behaviors that no individual explicitly designed. These emergent properties are often mistaken for intentional architecture. Observers search for a central planner, a hidden logic, or a unifying blueprint. In reality, the order they see is frequently the byproduct of distributed adaptation. Countless local optimizations, each rational within its own context, aggregate into a global structure that appears deliberate only in retrospect.

This retrospective illusion is powerful. Humans are natural storytellers, and stories prefer coherence. Faced with a complex outcome, we reconstruct a narrative in which every step leads naturally to the next. Uncertainty, hesitation, and accident are edited out. The resulting account is easier to remember, easier to teach, and easier to defend. It is also, inevitably, less accurate.

A more faithful perspective accepts that progress is rarely linear. Advancement proceeds through cycles of expansion and consolidation. During expansion, experimentation flourishes. Many approaches are tried, most of them failing. During consolidation, successful patterns are standardized, documented, and scaled. These phases alternate, and each is necessary. Without expansion, innovation stagnates. Without consolidation, knowledge dissolves into fragmentation.

An important but often overlooked aspect of this cycle is the role of constraints. Constraints are sometimes perceived as obstacles, yet they serve as catalysts for creativity. When resources are unlimited, choices lose structure; when limits are present, priorities must be established. The act of prioritization forces clarity about what truly matters. In this way, constraint shapes direction.

Another dimension of complexity arises from time. Short-term optimization and long-term resilience are not always aligned. Decisions that yield immediate efficiency may reduce adaptability later. Conversely, maintaining flexibility can appear wasteful in the present while proving invaluable in the future. Balancing these horizons requires judgment rather than calculation, because the future context cannot be fully specified.

Communication introduces further layers. Ideas must travel between individuals with different experiences, vocabularies, and assumptions. Each transmission involves translation, and translation inevitably alters meaning. Even when participants believe they agree, subtle divergences can accumulate. Over long periods, these divergences lead to entirely distinct interpretations emerging from what was once a shared premise.

To navigate such environments effectively, one must cultivate a tolerance for ambiguity. This does not mean abandoning rigor; rather, it means recognizing that rigor operates within boundaries. Clear reasoning can coexist with incomplete information. Indeed, the ability to act responsibly despite uncertainty is one of the defining features of mature systems.

Reflection plays a critical role here. Without periodic reassessment, processes that were once adaptive become rigid. Reflection functions as a kind of reset mechanism, allowing participants to distinguish between principles that remain valid and practices that persist only through inertia. It is an opportunity to ask not just whether something works, but why it works, and whether those conditions still apply.

Yet reflection must be paired with action. Endless analysis can be as limiting as unquestioned habit. The objective is not to achieve perfect understanding before moving forward, because such understanding is unattainable. Instead, the goal is iterative refinement: act, observe, adjust, and repeat. Through iteration, knowledge becomes embodied rather than merely abstract.

An interesting paradox emerges from this iterative view. Stability is not the absence of change; it is the capacity to absorb change without losing coherence. Systems that endure are not static. They evolve continuously, but they do so in ways that preserve certain invariants. These invariants provide identity across transformation, much like a melody remains recognizable even when played in a different key.

At the human level, engagement with complexity often produces a mixture of frustration and fascination. Frustration arises because outcomes resist prediction. Fascination persists because each layer of understanding reveals further depth. The experience is less like solving a puzzle and more like exploring a landscape—there is no final vantage point from which everything becomes visible, only a succession of perspectives, each broader than the last.

Learning, therefore, is not a process of accumulating isolated facts. It is the gradual reorganization of perception. Concepts that once seemed unrelated begin to connect. Patterns that were previously invisible become obvious. What changes is not only what we know, but how we interpret what we encounter.

This transformation is subtle and often difficult to measure. External indicators capture only part of the story. The deeper shift occurs internally, in the formation of mental models that enable more effective reasoning. These models are rarely explicit. They operate beneath conscious articulation, guiding intuition.

Collaboration amplifies both the benefits and challenges of complex work. Diverse perspectives increase the range of available insights, but they also increase the need for alignment. Successful collaboration depends less on unanimous agreement than on shared orientation.

Trust becomes essential in such settings. Trust allows individuals to rely on partial information contributed by others, making progress possible without requiring everyone to verify everything independently. It reduces redundancy while enabling specialization. However, trust must be maintained through transparency and accountability, or it erodes.

As systems mature, they often face a dilemma between accessibility and sophistication. Making tools easier to use broadens participation but can obscure underlying mechanisms. Preserving depth supports expert control but may discourage newcomers. Resolving this tension requires thoughtful layering.

In the end, the study of complex processes reveals as much about our modes of inquiry as about the phenomena themselves. We bring frameworks, metaphors, and expectations to every investigation. These cognitive instruments shape discovery just as surely as experimental apparatus shapes measurement.

Recognizing this mutual influence does not weaken objectivity; rather, it enriches it. Knowledge advances through dialogue between perspectives.

What remains constant is the interplay between structure and adaptation. Structure provides continuity. Adaptation provides relevance. Maintain both, and a system gains the capacity to persist amid uncertainty.

Thus, complexity should not be regarded as an adversary to be eliminated, but as an environment to be navigated. Mastery lies not in reducing the world to something simple, but in developing the intellectual flexibility to engage with what cannot be fully simplified. The aim is not final answers, but better questions—questions that open pathways rather than close them."""

# SAMPLE_TEXT = (
#     "The first thing people often misunderstand about complexity is that it rarely "
#     "announces itself with noise. More often, it emerges quietly, through the accumulation "
#     "of small decisions that appear harmless in isolation but become deeply influential when "
#     "composed together. A single assumption, left unexamined, forms the seed of a system; "
#     "repeated often enough, that system begins to feel natural, inevitable, and even necessary. "
#     "Yet nothing about it was inevitable. It was assembled, piece by piece, by individuals "
#     "acting within constraints they only partially understood.\n\n"
#     "Consider how any structured environment evolves. At the beginning there is openness, a "
#     "wide field of possible actions. Early participants improvise. They solve immediate problems "
#     "using whatever tools are available, guided more by intuition than by doctrine. Their "
#     "solutions are pragmatic rather than elegant, local rather than global. However, each "
#     "solution leaves a trace. That trace becomes a precedent, and precedent gradually transforms "
#     "into expectation. What was once an improvisation becomes a rule.\n\n"
#     "Over time, rules interact. Interactions create patterns. Patterns produce feedback. "
#     "Feedback stabilizes some behaviors while suppressing others. Eventually, the environment "
#     "acquires a form of memory—not a conscious memory, but a structural one. It remembers "
#     "through reinforcement. Paths that are used frequently become easier to traverse; paths "
#     "that are neglected disappear. In this way, history reshapes possibility."
# )


def run_inference(model, device, model_path: str, max_seq_len: int = 512):
    print("[3/3] Tokenize 并前向传播 …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    # ⚠️  原脚本写 inputs.shape，但 tokenizer 返回 BatchEncoding（dict-like），
    #     没有 .shape 属性，应使用 inputs.input_ids.shape。
    seq_len = inputs.input_ids.shape[1]
    print(f"   输入序列长度：{seq_len} tokens")

    with torch.no_grad():
        model(**inputs, use_cache=False)

    print(f"   前向传播完成，捕获 {len(attn_weights_by_layer)} 个全注意力层，"
          f"{len(residual_stream_by_layer)} 个 decoder 层。")
    return seq_len


# ================================================================== #
#  5. 可视化                                                           #
# ================================================================== #

def _savefig(fname: str):
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   已保存 → {fname}")


# ------------------------------------------------------------------ #
#  5a. Attention Sink                                                  #
# ------------------------------------------------------------------ #
def plot_attention_sink(seq_len: int, out_path: str = "attention_sink.png"):
    """
    对每个全注意力层画两列子图：
      左列：注意力热图（所有 head 平均），x 轴 = kv token，y 轴 = query token。
            可以直观看出 sink token 是否对应"整列高权重"的竖条。
      右列：每个 kv token 被所有 query 平均接收的权重（列均值），
            放大展示前 min(seq_len, 64) 个 token，便于定位 sink 位置。

    ⚠️  原脚本只画最后一个 query token 对前 100 个 kv token 的一条折线，
        既看不出全局分布，也无法区分"sink token 被所有 query 一致关注"
        与"最后一个 token 恰好关注了某 token"这两种情况。
    """
    if not attn_weights_by_layer:
        print("   [跳过] 未捕获到任何注意力权重（请确认模型以 eager 模式加载）")
        return

    layer_ids = sorted(attn_weights_by_layer.keys())
    n_layers = len(layer_ids)
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 4.5 * n_layers))
    if n_layers == 1:
        axes = [axes]   # 统一为二维列表

    for row, lid in enumerate(layer_ids):
        w = attn_weights_by_layer[lid]   # (heads, q, kv)
        w_mean = w.mean(dim=0).numpy()        # (q, kv)

        # ── 左：热图 ──────────────────────────────────────────────
        ax_heat = axes[row][0]
        im = ax_heat.imshow(
            w_mean, aspect="auto", cmap="hot", origin="upper",
            vmin=0, vmax=np.percentile(w_mean, 99),
        )
        ax_heat.set_title(f"Layer {lid}  Attention Heatmap (head-avg)", fontsize=9)
        ax_heat.set_xlabel("KV Token Index")
        ax_heat.set_ylabel("Query Token Index")
        plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04)

        # ── 右：列均值柱状图（前 64 个 kv token）──────────────────
        ax_line = axes[row][1]
        col_mean = w_mean.mean(axis=0)        # (kv,)  每个 kv token 被所有 query 平均关注量
        show_len = min(seq_len, 64)
        ax_line.bar(range(show_len), col_mean[:show_len],
                    color="steelblue", alpha=0.85)
        # 标注 top-3 sink token（红色虚线）
        top3 = np.argsort(col_mean[:show_len])[::-1][:3]
        for t in top3:
            ax_line.axvline(t, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
                            label=f"sink@{t}")
        ax_line.set_title(
            f"Layer {lid}  Avg Attention Received  (first {show_len} tokens)", fontsize=9
        )
        ax_line.set_xlabel("KV Token Index")
        ax_line.set_ylabel("Avg Weight")
        ax_line.legend(fontsize=7)
        ax_line.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("① Attention Sink Check", fontsize=13, y=1.005)
    _savefig(out_path)


# ------------------------------------------------------------------ #
#  5b. Massive Activation                                              #
# ------------------------------------------------------------------ #
def plot_massive_activation(out_path: str = "massive_activation.png"):
    """
    Massive Activation：残差流中少数 token 的 L2 范数在某些层异常大
    （通常是 BOS / 分隔符等特殊 token，从中间层开始"拉开"）。

    画三张子图：
      上：逐层折线 —— 每层每个 token 的残差流 L2 范数
      中：热图 —— (层 × token) 的 L2 范数矩阵
      下：每层 L2 范数的 max / 99th percentile / mean 三条统计曲线
    """
    if not residual_stream_by_layer:
        print("   [跳过] 未捕获到残差流")
        return

    layer_ids = sorted(residual_stream_by_layer.keys())
    norm_matrix = np.stack(
        [residual_stream_by_layer[lid].norm(dim=-1).numpy() for lid in layer_ids],
        axis=0,
    )  # (n_layers, seq_len)

    # seq_len = norm_matrix.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # ── 上：逐层折线 ──────────────────────────────────────────────
    ax0 = axes[0]
    cmap = plt.get_cmap("tab20")
    nl = len(layer_ids)
    for i, lid in enumerate(layer_ids):
        ax0.plot(norm_matrix[i], alpha=0.65,
                 color=cmap(i / max(nl - 1, 1)), label=f"L{lid}")
    ax0.set_title("Residual Stream L2 Norm per Token (all layers)", fontsize=10)
    ax0.set_xlabel("Token Index")
    ax0.set_ylabel("L2 Norm")
    ax0.legend(fontsize=6, ncol=8, loc="upper right")
    ax0.grid(True, linestyle="--", alpha=0.4)

    # ── 中：热图 (层 × token) ─────────────────────────────────────
    ax1 = axes[1]
    im = ax1.imshow(norm_matrix, aspect="auto", cmap="YlOrRd", origin="upper")
    ax1.set_title("L2 Norm Heatmap  (row = layer, col = token)", fontsize=10)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("Layer Index")
    ax1.set_yticks(range(nl))
    ax1.set_yticklabels([str(l) for l in layer_ids], fontsize=6)
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.04)

    # ── 下：每层统计曲线 ──────────────────────────────────────────
    ax2 = axes[2]
    ax2.plot(layer_ids, norm_matrix.max(axis=1),
             "r-o",  markersize=4, label="max")
    ax2.plot(layer_ids, np.percentile(norm_matrix, 99, axis=1),
             "b--s", markersize=4, label="99th pct")
    ax2.plot(layer_ids, norm_matrix.mean(axis=1),
             "g-^",  markersize=4, label="mean")
    ax2.set_title(
        "Per-layer L2 Norm Statistics\n"
        "(max/99th pct far above mean → Massive Activation exists)", fontsize=9
    )
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("L2 Norm")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("② Massive Activation Check", fontsize=13)
    _savefig(out_path)


# ------------------------------------------------------------------ #
#  5c. Residual Sink                                                   #
# ------------------------------------------------------------------ #
def plot_residual_sink(out_path: str = "residual_sink.png", top_k_dims: int = 16):
    """
    Residual Sink：少数固定的 hidden_size 维度（feature dimension）在几乎
    所有 token 上持续具有极大绝对值激活，且随层加深越来越显著。

    ⚠️  这是原脚本完全缺失的分析。原脚本只看了每个 token 的 L2 范数
        （Massive Activation 视角），却没有按「维度」分析哪些固定维度
        在所有 token 上都很大（Residual Sink 视角）。

    可视化：
      ① 热图：最后一层 (token × top_k_dims) 的激活值
             ——若某行（维度）在几乎所有 token 上都很大 → Residual Sink
      ② 折线：top-K 维度的「跨 token 绝对值最大值」随层的演化
             ——若某些维度从某层起持续"拉开差距" → Residual Sink
      ③ 分布对比直方图：top-K sink 维度 vs bottom-K 普通维度的激活分布
             ——sink 维度分布尾巴更重 → Residual Sink 确认
    """
    if not residual_stream_by_layer:
        print("   [跳过] 未捕获到残差流")
        return

    layer_ids = sorted(residual_stream_by_layer.keys())

    # 每层每维度的「跨 token 绝对值最大值」: (n_layers, hidden_size)
    dim_max_matrix = np.stack(
        [residual_stream_by_layer[lid].abs().max(dim=0).values.numpy()
         for lid in layer_ids],
        axis=0,
    )

    # 用最后一层确定全局 top-K sink 维度
    # .copy() 消除 [::-1] 产生的负步长，避免 torch 索引报错
    top_k_idx = np.argsort(dim_max_matrix[-1])[::-1][:top_k_dims].copy()
    bot_k_idx = np.argsort(dim_max_matrix[-1])[:top_k_dims].copy()   # 最小的 top_k 维度（对比用）

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # ── ① 热图：最后一层 (top_k_dims × seq_len) ──────────────────
    ax0 = axes[0]
    last_lid = layer_ids[-1]
    hs_last = residual_stream_by_layer[last_lid].numpy()   # (seq_len, hidden_size)
    hs_topk = hs_last[:, top_k_idx].T                      # (top_k_dims, seq_len)
    im0 = ax0.imshow(
        hs_topk, aspect="auto", cmap="bwr",
        norm=mcolors.CenteredNorm(), origin="upper",
    )
    ax0.set_title(
        f"Layer {last_lid} | Top-{top_k_dims} Sink Dims × All Tokens\n"
        "(each row = one feature dim; strong color across most cols → Residual Sink)",
        fontsize=9,
    )
    ax0.set_xlabel("Token Index")
    ax0.set_ylabel("Sink Dim (sorted by |max|)")
    ax0.set_yticks(range(top_k_dims))
    ax0.set_yticklabels([str(d) for d in top_k_idx], fontsize=7)
    plt.colorbar(im0, ax=ax0, fraction=0.02, pad=0.04)

    # ── ② 折线：top-K 维度的 dim_max 随层演化 ────────────────────
    ax1 = axes[1]
    cmap2 = plt.get_cmap("tab20")
    nl = max(top_k_dims - 1, 1)
    for i, dim_i in enumerate(top_k_idx):
        ax1.plot(layer_ids, dim_max_matrix[:, dim_i],
                 label=f"dim {dim_i}", alpha=0.85,
                 color=cmap2(i / nl))
    ax1.set_title(
        f"Top-{top_k_dims} Dim |max| across Layers\n"
        "(dims that diverge from mid-layer onwards → Residual Sink)",
        fontsize=9,
    )
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("|max| Activation Value")
    ax1.legend(fontsize=7, ncol=4, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── ③ 分布对比直方图 ──────────────────────────────────────────
    ax2 = axes[2]
    sink_vals = hs_last[:, top_k_idx].flatten()
    norm_vals = hs_last[:, bot_k_idx].flatten()
    vmin = min(float(sink_vals.min()), float(norm_vals.min()))
    vmax = max(float(sink_vals.max()), float(norm_vals.max()))
    bins = np.linspace(vmin, vmax, 80)
    ax2.hist(sink_vals, bins=bins, alpha=0.6, color="red",
             label=f"Top-{top_k_dims} sink dims")
    ax2.hist(norm_vals, bins=bins, alpha=0.6, color="blue",
             label=f"Bottom-{top_k_dims} normal dims")
    ax2.set_title(
        f"Layer {last_lid} | Activation Distribution: Sink dims vs Normal dims\n"
        "(heavier tails in sink dims → Residual Sink confirmed)",
        fontsize=9,
    )
    ax2.set_xlabel("Activation Value")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("③ Residual Sink Check", fontsize=13)
    _savefig(out_path)


# ================================================================== #
#  5d. Norm Weight 分布分析（新增）                                    #
# ================================================================== #

def _collect_norm_weights(model) -> dict:
    """
    遍历所有 DecoderLayer，收集两类 RMSNorm 的可学习权重：
      - input_layernorm     : 位于 Token Mixer（Attention/Linear Attn）之前
      - post_attention_layernorm : 位于 MLP 之前

    论文核心观点：RMSNorm 的可学习缩放权重 w 中，少数维度会被训练成异常大的值，
    这些 outlier weight 维度与 Residual Sink 维度高度重合，
    共同实现 outlier-driven rescaling：
        - 残差流中 sink 维度激活值 x_d 异常大
        - LayerNorm 归一化时分母 RMS(x) ≈ x_d / sqrt(D)，即由 sink 决定
        - 归一化后非 sink 维度被压缩到接近 0，sink 维度被 norm weight 再次放大
        → 达到对其他分量的动态重缩放效果

    返回：
        {
          layer_idx: {
            "input":    Tensor(hidden_size,)  float32,  # input_layernorm.weight
            "post_attn": Tensor(hidden_size,) float32,  # post_attention_layernorm.weight
          },
          ...
        }
    """
    norm_weights = {}
    for layer_idx, layer in enumerate(model.model.layers):
        entry = {}
        # RMSNorm weight 在 Qwen3_5RMSNorm 中存储为 (1 + weight)，
        # 但 forward 中实际缩放因子 = 1 + weight，因此直接取 .weight 即可看分布
        if hasattr(layer, "input_layernorm"):
            w = layer.input_layernorm.weight.detach().cpu().float()
            entry["input"] = w
        if hasattr(layer, "post_attention_layernorm"):
            w = layer.post_attention_layernorm.weight.detach().cpu().float()
            entry["post_attn"] = w
        norm_weights[layer_idx] = entry
    return norm_weights


def plot_norm_weight_distribution(
    model,
    out_path_prefix: str = "norm_weight",
    top_k_dims: int = 16,
):
    """
    四张图，分析 Norm Weight 分布及其与 Residual Sink 的关联。

    ④-A  norm_weight_heatmap.png
         跨层 norm weight 绝对值热图：
           行 = 层，列 = 维度，颜色 = |weight|
         → 若某些固定列（维度）在多数层中持续高亮，说明这些维度是
           稳定的 weight outlier，与 Residual Sink 维度候选一致。

    ④-B  norm_weight_distribution.png
         每层 input_layernorm 与 post_attention_layernorm 的 weight 分布直方图：
           左列 = input_layernorm，右列 = post_attention_layernorm
         → 若分布有重尾或双峰，说明少数维度被显著放大。

    ④-C  norm_weight_vs_residual_sink.png
         Norm Weight Outlier 维度 × Residual Sink 维度 对齐分析：
           子图1：逐层 norm weight top-K 维度排名 vs residual stream dim_max top-K
                  用 Spearman 秩相关系数量化二者对齐程度
           子图2：各层两者的 top-K 维度 Overlap（Jaccard 相似度）折线图
         → Overlap 高 / Spearman ρ 高 → norm weight outlier 与 residual sink 强耦合，
           验证 outlier-driven rescaling 机制。

    ④-D  norm_weight_outlier_driven_rescaling.png
         Outlier-Driven Rescaling 效应直接验证：
           对每个有残差流数据的层，取 sink 维度激活值的绝对值均值（跨 token），
           与对应 norm weight 值做散点图，验证二者正相关；
           同时画归一化后非 sink 分量的平均 norm，验证被压缩。
    """
    if not residual_stream_by_layer:
        print("   [跳过] 未捕获到残差流，无法做 Norm-Residual Sink 对齐分析")
        return

    norm_weights = _collect_norm_weights(model)
    layer_ids = sorted(residual_stream_by_layer.keys())
    hidden_size = next(iter(residual_stream_by_layer.values())).shape[1]

    # ── 预计算 Residual Sink 各层维度 dim_max ─────────────────────────
    # dim_max_matrix[i, d] = 第 layer_ids[i] 层第 d 维的跨 token |max|
    dim_max_rs = np.stack(
        [residual_stream_by_layer[lid].abs().max(dim=0).values.numpy()
         for lid in layer_ids],
        axis=0,
    )  # (n_layers, hidden_size)

    # 全局 sink 维度（用最后一层）
    # 注意：[::-1] 会产生负步长 numpy 数组，直接用于 torch 索引会报错，
    # 必须调用 .copy() 消除负步长。
    global_sink_idx = np.argsort(dim_max_rs[-1])[::-1][:top_k_dims].copy()

    # ── ④-A: 跨层 norm weight 绝对值热图 ─────────────────────────────
    print("   绘制 ④-A: Norm Weight 热图 …")
    for norm_key, norm_label in [("input", "input_layernorm"),
                                 ("post_attn", "post_attention_layernorm")]:
        w_matrix = np.stack(
            [norm_weights[lid][norm_key].numpy()
             for lid in layer_ids if norm_key in norm_weights[lid]],
            axis=0,
        )  # (n_layers, hidden_size)
        valid_layer_ids = [lid for lid in layer_ids if norm_key in norm_weights[lid]]

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        # 热图
        ax0 = axes[0]
        # 只展示前 min(hidden_size, 512) 维避免图像过宽
        show_dims = min(hidden_size, 512)
        im = ax0.imshow(
            np.abs(w_matrix[:, :show_dims]),
            aspect="auto", cmap="YlOrRd", origin="upper",
        )
        ax0.set_title(
            f"{norm_label} | Weight |value| Heatmap  (row=layer, col=dim)\n"
            f"Persistent bright columns → weight outlier dims (potential Residual Sink dims)",
            fontsize=9,
        )
        ax0.set_xlabel(f"Hidden Dim Index  (showing first {show_dims} of {hidden_size})")
        ax0.set_ylabel("Layer Index")
        ax0.set_yticks(range(len(valid_layer_ids)))
        ax0.set_yticklabels([str(l) for l in valid_layer_ids], fontsize=6)
        plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.03)

        # 每层 weight 的 max / 99th / mean 统计曲线
        ax1 = axes[1]
        ax1.plot(valid_layer_ids, np.abs(w_matrix).max(axis=1),
                 "r-o", markersize=4, label="|max|")
        ax1.plot(valid_layer_ids, np.percentile(np.abs(w_matrix), 99, axis=1),
                 "b--s", markersize=4, label="|99th pct|")
        ax1.plot(valid_layer_ids, np.abs(w_matrix).mean(axis=1),
                 "g-^", markersize=4, label="|mean|")
        ax1.set_title(
            f"{norm_label} | Per-layer Weight Statistics\n"
            "(max >> mean → outlier weights present)",
            fontsize=9,
        )
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("|Weight| Value")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"④-A  {norm_label} Weight Distribution", fontsize=12)
        _savefig(f"{out_path_prefix}_heatmap_{norm_key}.png")

    # ── ④-B: 逐层 weight 分布直方图（采样若干层） ───────────────────
    print("   绘制 ④-B: 逐层 Norm Weight 分布直方图 …")
    sample_layer_ids = layer_ids[::max(1, len(layer_ids) // 8)]  # 均匀采样最多 8 层
    n_sample = len(sample_layer_ids)
    fig, axes = plt.subplots(n_sample, 2, figsize=(14, 3 * n_sample))
    if n_sample == 1:
        axes = [axes]

    for row, lid in enumerate(sample_layer_ids):
        for col, (norm_key, norm_label) in enumerate([
            ("input",     "input_layernorm"),
            ("post_attn", "post_attention_layernorm"),
        ]):
            ax = axes[row][col]
            if norm_key not in norm_weights[lid]:
                ax.axis("off")
                continue
            w = norm_weights[lid][norm_key].numpy()
            # 实际缩放因子 = 1 + w（Qwen3_5RMSNorm 的设计）
            scale = 1.0 + w
            ax.hist(scale, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
            # 标注 top-3 outlier 维度
            top3 = np.argsort(np.abs(scale))[::-1][:3]
            for t in top3:
                ax.axvline(scale[t], color="red", linestyle="--",
                           linewidth=1.0, label=f"dim {t}={scale[t]:.2f}")
            ax.set_title(f"L{lid} {norm_label}  (scale = 1+w)", fontsize=8)
            ax.set_xlabel("Scale Value (1 + weight)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=6)
            ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("④-B  Per-layer Norm Weight Scale Distribution\n"
                 "(heavy right tail / outlier spikes → learned amplification on specific dims)",
                 fontsize=11)
    _savefig(f"{out_path_prefix}_distribution.png")

    # ── ④-C: Norm Weight Outlier 与 Residual Sink 维度对齐分析 ───────
    print("   绘制 ④-C: Norm Weight Outlier vs Residual Sink 对齐分析 …")

    spearman_rho_input = []
    spearman_rho_post_attn = []
    jaccard_input = []
    jaccard_post_attn = []

    for i, lid in enumerate(layer_ids):
        rs_rank = np.argsort(dim_max_rs[i])[::-1][:top_k_dims].copy()   # residual sink top-K 维

        for norm_key, rho_list, jac_list in [
            ("input",     spearman_rho_input,     jaccard_input),
            ("post_attn", spearman_rho_post_attn, jaccard_post_attn),
        ]:
            if norm_key not in norm_weights[lid]:
                rho_list.append(np.nan)
                jac_list.append(np.nan)
                continue

            w_abs = np.abs(norm_weights[lid][norm_key].numpy())
            nw_rank = np.argsort(w_abs)[::-1][:top_k_dims].copy()       # norm weight top-K 维

            # Spearman 秩相关：用全维度的排名
            rho, _ = spearmanr(dim_max_rs[i], w_abs)
            rho_list.append(rho)

            # Jaccard：top-K 集合重叠率
            inter = len(set(rs_rank) & set(nw_rank))
            union = len(set(rs_rank) | set(nw_rank))
            jac_list.append(inter / union if union > 0 else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax0 = axes[0]
    ax0.plot(layer_ids, spearman_rho_input,
             "b-o", markersize=5, label="input_layernorm  Spearman ρ")
    ax0.plot(layer_ids, spearman_rho_post_attn,
             "r--s", markersize=5, label="post_attn_layernorm  Spearman ρ")
    ax0.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax0.set_title(
        "Spearman Rank Correlation: Norm Weight |value| vs Residual Sink dim_max\n"
        "(ρ → 1 means norm weight outlier dims strongly align with residual sink dims\n"
        " → validates outlier-driven rescaling hypothesis)",
        fontsize=9,
    )
    ax0.set_xlabel("Layer Index")
    ax0.set_ylabel("Spearman ρ")
    ax0.set_ylim(-0.2, 1.05)
    ax0.legend()
    ax0.grid(True, linestyle="--", alpha=0.4)

    ax1 = axes[1]
    ax1.plot(layer_ids, jaccard_input,
             "b-o", markersize=5, label=f"input_layernorm  Jaccard (top-{top_k_dims})")
    ax1.plot(layer_ids, jaccard_post_attn,
             "r--s", markersize=5, label=f"post_attn_layernorm  Jaccard (top-{top_k_dims})")
    ax1.set_title(
        f"Top-{top_k_dims} Dim Jaccard Overlap: Norm Weight Outliers vs Residual Sink Dims\n"
        "(Jaccard > 0.5 → >50% overlap, strong structural coupling)",
        fontsize=9,
    )
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Jaccard Similarity")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("④-C  Norm Weight Outlier ↔ Residual Sink Alignment", fontsize=12)
    _savefig(f"{out_path_prefix}_alignment.png")

    # ── ④-D: Outlier-Driven Rescaling 效应直接验证 ───────────────────
    print("   绘制 ④-D: Outlier-Driven Rescaling 效应验证 …")
    # 机制：sink 维度绝对值越大 → RMS(x) 越大 → LayerNorm 后非 sink 分量 norm 越小
    # 验证：对每层，画 sink 维度平均 |activation| vs 该层 input_layernorm 之后
    #       非 sink 分量 norm 的散点，期望负相关
    #
    # 用 input_layernorm 之前的残差流（= 上一层输出）作为 x
    # input_layernorm(x) = x / RMS(x) * (1 + w)
    # 非 sink 分量 norm after norm = ||x_{non_sink}|| / RMS(x)  （近似）

    sink_act_mean = []   # 每层：sink 维度跨 token 平均 |activation|（来自上层输出）
    nonsink_ratio = []   # 每层：非 sink 分量 norm / RMS(x)（量化压缩程度）
    plot_layer_ids = []

    for i, lid in enumerate(layer_ids):
        hs = residual_stream_by_layer[lid]   # (seq_len, hidden_size)，当前层输出

        # sink 维度的平均绝对值（跨 token）
        sink_mean = hs[:, global_sink_idx].abs().mean().item()
        sink_act_mean.append(sink_mean)

        # RMS(x) 跨 token 平均
        rms_x = hs.pow(2).mean(dim=-1).sqrt().mean().item()  # scalar

        # 非 sink 维度的 norm
        non_sink_mask = np.ones(hidden_size, dtype=bool)
        non_sink_mask[global_sink_idx] = False
        non_sink_norm = hs[:, non_sink_mask].norm(dim=-1).mean().item()

        # 非 sink 分量在 LayerNorm 后的压缩比 ≈ non_sink_norm / (sqrt(hidden_size) * rms_x)
        ratio = non_sink_norm / (rms_x * (hidden_size ** 0.5) + 1e-8)
        nonsink_ratio.append(ratio)
        plot_layer_ids.append(lid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：两条折线随层的变化
    ax0 = axes[0]
    ax0_twin = ax0.twinx()
    l1, = ax0.plot(plot_layer_ids, sink_act_mean,
                   "r-o", markersize=5, label="Sink dim avg |activation|")
    l2, = ax0_twin.plot(plot_layer_ids, nonsink_ratio,
                        "b--s", markersize=5, label="Non-sink norm / RMS(x)·√D")
    ax0.set_xlabel("Layer Index")
    ax0.set_ylabel("Sink |activation| (mean over tokens & dims)", color="red")
    ax0_twin.set_ylabel("Non-sink compression ratio", color="blue")
    ax0.set_title(
        "Outlier-Driven Rescaling: Per-layer Trend\n"
        "(sink grows → non-sink compressed → rescaling active)",
        fontsize=9,
    )
    ax0.legend(handles=[l1, l2], loc="upper left", fontsize=8)
    ax0.grid(True, linestyle="--", alpha=0.3)

    # 右：散点图（每层一个点），期望负相关
    ax1 = axes[1]
    sc = ax1.scatter(sink_act_mean, nonsink_ratio,
                     c=plot_layer_ids, cmap="viridis", s=60, zorder=3)
    # 添加趋势线
    if len(sink_act_mean) > 2:
        z = np.polyfit(sink_act_mean, nonsink_ratio, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(sink_act_mean), max(sink_act_mean), 100)
        ax1.plot(xs, p(xs), "r--", linewidth=1.5, label="trend")
    plt.colorbar(sc, ax=ax1, label="Layer Index")
    ax1.set_xlabel("Sink dim avg |activation|")
    ax1.set_ylabel("Non-sink compression ratio")
    ax1.set_title(
        "Outlier-Driven Rescaling: Scatter Plot\n"
        "(negative slope → larger sink → stronger suppression of non-sink → rescaling confirmed)",
        fontsize=9,
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("④-D  Outlier-Driven Rescaling Validation", fontsize=12)
    _savefig(f"{out_path_prefix}_rescaling.png")


# ================================================================== #
#  6. 主函数                                                           #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Outlier 可视化脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练完成的模型路径")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="输入序列截断长度（默认 512）")
    parser.add_argument("--top_k_dims", type=int, default=16,
                        help="Residual Sink / Norm Weight 分析的 top-K 维度数（默认 16）")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="图片输出目录（默认当前目录）")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在：{args.model_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型 + 注入 hooks
    model, device = load_model_and_register_hooks(args.model_path)

    # 推理（一次前向传播即可）
    seq_len = run_inference(model, device, args.model_path, max_seq_len=args.seq_len)

    n_layers = len(model.model.layers)
    print(f"\n模型共 {n_layers} 层，开始绘图 …")

    plot_attention_sink(
        seq_len,
        out_path=os.path.join(args.output_dir, "attention_sink.png"),
    )
    plot_massive_activation(
        out_path=os.path.join(args.output_dir, "massive_activation.png"),
    )
    plot_residual_sink(
        out_path=os.path.join(args.output_dir, "residual_sink.png"),
        top_k_dims=args.top_k_dims,
    )
    plot_norm_weight_distribution(
        model=model,
        out_path_prefix=os.path.join(args.output_dir, "norm_weight"),
        top_k_dims=args.top_k_dims,
    )

    print("\n✅ 全部完成！")
    print("\n输出文件列表：")
    outputs = [
        "attention_sink.png",
        "massive_activation.png",
        "residual_sink.png",
        "norm_weight_heatmap_input.png",
        "norm_weight_heatmap_post_attn.png",
        "norm_weight_distribution.png",
        "norm_weight_alignment.png",
        "norm_weight_rescaling.png",
    ]
    for f in outputs:
        full = os.path.join(args.output_dir, f)
        status = "✓" if os.path.exists(full) else "✗ (未生成)"
        print(f"   {status}  {full}")


if __name__ == "__main__":
    main()
