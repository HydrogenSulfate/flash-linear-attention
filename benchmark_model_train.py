#!/usr/bin/env python3
"""
benchmark_model.py — 训练峰值显存 & Python 推理性能测试
适用于 flame + flash-linear-attention + torchtitan 训练的模型

用法示例:
  # 基本用法（全部测试，默认参数）
  python benchmark_model.py --model_path /path/to/model_dir

  # 只测训练显存
  python benchmark_model.py --model_path /path/to/model_dir --mode train_memory

  # 只测推理
  python benchmark_model.py --model_path /path/to/model_dir --mode inference

  # 自定义参数
  python benchmark_model.py --model_path /path/to/model_dir \
      --train_batch_size 4 --train_seq_len 4096 \
      --infer_batch_sizes 1 4 8 \
      --infer_seq_lens 256 1024 4096 \
      --gen_len 128 --dtype bf16

  # 保存结果为 JSON
  python benchmark_model.py --model_path /path/to/model_dir --output result.json
"""

import argparse
import contextlib
import gc
import json
import time
import traceback
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────


def bytes_to_gib(n: int) -> float:
    return n / (1024 ** 3)


def reset_peak(device: torch.device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def peak_gib(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return bytes_to_gib(torch.cuda.max_memory_allocated(device))


def current_gib(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return bytes_to_gib(torch.cuda.memory_allocated(device))


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def sep(char="─", width=72):
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, device: torch.device):
    """
    加载模型与 tokenizer。
    flash-linear-attention 的自定义架构需要先 import fla 完成注册。
    """
    try:
        import fla  # noqa: F401  — 注册自定义模型类到 transformers
        print("  [✓] fla 已导入，自定义架构已注册")
    except ImportError:
        print("  [!] fla 未安装，若模型为标准架构可忽略此警告")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,   # 手动管理设备，保证显存统计准确
        attn_implementation="flash_attention_3",
    )
    model = model.to(device)
    return model, tokenizer


def model_info(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total_params": total}


# ─────────────────────────────────────────────────────────────────────────────
# 训练峰值显存测试
# ─────────────────────────────────────────────────────────────────────────────

def bench_train_memory_once(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    with_optimizer: bool,
) -> dict:
    """
    执行一步训练（forward + backward [+ optimizer.step]），记录峰值显存。
    每次调用独立加载 / 卸载模型，保证显存统计干净。
    """
    label = "含 AdamW 优化器" if with_optimizer else "仅前向+反向（无优化器）"
    result = {
        "with_optimizer": with_optimizer,
        "label": label,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    try:
        clear_cache()
        model, _ = load_model_and_tokenizer(model_path, dtype, device)
        model.train()
        vocab_size = model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if with_optimizer else None

        # 确保权重/优化器状态都在 GPU 上再清零计量起点
        torch.cuda.synchronize(device)
        clear_cache()
        reset_peak(device)

        if optimizer:
            optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if optimizer:
            optimizer.step()

        peak = peak_gib(device)
        result.update({
            "status": "ok",
            "peak_memory_gib": round(peak, 3),
            "loss": round(loss.item(), 6),
        })

    except torch.cuda.OutOfMemoryError:
        result.update({"status": "OOM", "peak_memory_gib": None})
    except Exception as e:
        result.update({"status": f"ERROR: {e}", "peak_memory_gib": None})
        traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            del model, optimizer
        clear_cache()

    return result


def bench_train_memory(model_path, device, dtype, batch_size, seq_len) -> list:
    results = []
    for with_opt in [True, False]:
        tag = "含优化器" if with_opt else "无优化器"
        print(f"  [{tag}] batch={batch_size}, seq_len={seq_len} ...", end=" ", flush=True)
        r = bench_train_memory_once(model_path, device, dtype, batch_size, seq_len, with_opt)
        if r["status"] == "ok":
            print(f"峰值显存 = {r['peak_memory_gib']:.3f} GiB  (loss={r['loss']})")
        else:
            print(r["status"])
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 推理性能测试
# ─────────────────────────────────────────────────────────────────────────────

def bench_inference_single(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup_runs: int,
    measure_runs: int,
) -> dict:
    """
    针对一组 (batch_size, prompt_len, gen_len) 测量：
      - TTFT  : 仅 prefill（单次 model forward）的耗时，avg/min/max
      - 吞吐量: 完整 generate 的 new_tokens / 总耗时
      - 峰值显存
    """
    result = {"batch_size": batch_size, "prompt_len": prompt_len, "gen_len": gen_len}

    try:
        vocab_size = model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

        # attention_mask: 全 1（随机 input 没有 padding，全部有效）
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = model.config.eos_token_id
        # eos_token_id 可能是 int 或 list，统一取第一个
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = pad_token_id[0]

        # ── 预热 ──────────────────────────────────────────────────────
        with torch.inference_mode():
            for _ in range(warmup_runs):
                model.generate(input_ids, attention_mask=attention_mask,
                               pad_token_id=pad_token_id,
                               max_new_tokens=gen_len,
                               do_sample=False, use_cache=True)
        clear_cache()

        # ── 正式计时 ──────────────────────────────────────────────────
        ttft_ms_list = []
        tps_list = []
        prefill_peak_list = []
        decode_peak_list = []

        for _ in range(measure_runs):
            # TTFT + prefill 峰值显存
            clear_cache()
            reset_peak(device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize(device)
            ttft_ms_list.append((time.perf_counter() - t0) * 1000)
            prefill_peak_list.append(peak_gib(device))

            # 完整生成: prefill + decode，decode 峰值 = generate 峰值（含 KV cache）
            clear_cache()
            reset_peak(device)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(input_ids, attention_mask=attention_mask,
                                     pad_token_id=pad_token_id,
                                     max_new_tokens=gen_len,
                                     do_sample=False, use_cache=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t1
            decode_peak_list.append(peak_gib(device))

            new_tokens = (out.shape[1] - prompt_len) * batch_size
            tps_list.append(new_tokens / elapsed if elapsed > 0 else 0.0)

        def _avg(lst):
            return round(sum(lst) / len(lst), 3)

        result.update({
            "status": "ok",
            "ttft_ms":                        _avg(ttft_ms_list),
            "ttft_ms_min":                    round(min(ttft_ms_list), 3),
            "ttft_ms_max":                    round(max(ttft_ms_list), 3),
            "throughput_tokens_per_sec":      round(_avg(tps_list), 2),
            "prefill_peak_memory_gib":        _avg(prefill_peak_list),
            "decode_peak_memory_gib":         _avg(decode_peak_list),
        })

    except torch.cuda.OutOfMemoryError:
        result.update({"status": "OOM"})
    except Exception as e:
        result.update({"status": f"ERROR: {e}"})
        traceback.print_exc()

    return result


def bench_inference(
    model_path, device, dtype,
    batch_sizes, seq_lens, gen_len,
    warmup_runs, measure_runs,
) -> tuple:
    """加载一次模型，遍历所有 (batch, seq_len) 组合。"""
    meta = {}
    results = []

    try:
        clear_cache()
        model, _ = load_model_and_tokenizer(model_path, dtype, device)
        model.eval()

        info = model_info(model)
        load_mem = current_gib(device)
        meta = {
            "total_params_M": round(info["total_params"] / 1e6, 2),
            "model_load_memory_gib": round(load_mem, 3),
        }
        print(f"  参数量: {meta['total_params_M']:.1f}M  "
              f"加载显存: {meta['model_load_memory_gib']:.3f} GiB")

        for bs in batch_sizes:
            for sl in seq_lens:
                print(f"  batch={bs}, prompt_len={sl}, gen_len={gen_len} ...",
                      end=" ", flush=True)
                r = bench_inference_single(
                    model, device, bs, sl, gen_len, warmup_runs, measure_runs
                )
                results.append(r)
                if r["status"] == "ok":
                    print(f"TTFT={r['ttft_ms']:.1f}ms  "
                          f"吞吐={r['throughput_tokens_per_sec']:.1f} tok/s  "
                          f"峰值显存={r['peak_memory_gib']:.3f} GiB")
                else:
                    print(r["status"])

    except Exception as e:
        print(f"  [ERROR] 推理测试失败: {e}")
        traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            del model
        clear_cache()

    return meta, results


# ─────────────────────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────────────────────

def print_train_summary(train_results: list):
    sep()
    print("训练峰值显存汇总")
    sep()
    print(f"  {'模式':<32} {'batch':<7} {'seq_len':<9} {'峰值显存 (GiB)':<18} {'状态'}")
    print("  " + "─" * 68)
    for r in train_results:
        peak = f"{r['peak_memory_gib']:.3f}" if r.get("peak_memory_gib") is not None else "N/A"
        print(f"  {r['label']:<32} {r['batch_size']:<7} {r['seq_len']:<9} "
              f"{peak:<18} {r['status']}")


def print_inference_summary(infer_results: list):
    sep()
    print("推理性能汇总")
    sep()
    print(f"  {'bs':<5} {'plen':<7} {'glen':<6} "
          f"{'TTFT avg(ms)':<14} {'min':<9} {'max':<9} "
          f"{'吞吐(tok/s)':<14} {'prefill峰值显存(GiB)':<22} {'decode峰值显存(GiB)':<22} {'状态'}")
    print("  " + "─" * 110)
    for r in infer_results:
        if r["status"] == "ok":
            print(f"  {r['batch_size']:<5} {r['prompt_len']:<7} {r['gen_len']:<6} "
                  f"{r['ttft_ms']:<14.1f} {r['ttft_ms_min']:<9.1f} {r['ttft_ms_max']:<9.1f} "
                  f"{r['throughput_tokens_per_sec']:<14.1f} "
                  f"{r['prefill_peak_memory_gib']:<22.3f} {r['decode_peak_memory_gib']:<22.3f} ok")
        else:
            print(f"  {r['batch_size']:<5} {r['prompt_len']:<7} {r['gen_len']:<6} "
                  f"{'—':<14} {'—':<9} {'—':<9} {'—':<14} {'—':<22} {'—':<22} {r['status']}")


# ─────────────────────────────────────────────────────────────────────────────
# 多模型对比绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmarks(json_paths: list[str], out_dir: str = ".") -> None:
    """
    读取多个 benchmark JSON 文件，绘制多模型对比图。

    生成文件（保存到 out_dir）：
      train_memory.png        — 训练峰值显存对比（含/不含优化器，分组柱状图）
      infer_ttft.png          — TTFT 对比（每个 bs×plen 组合为一组，折线图）
      infer_throughput.png    — 吞吐量对比（折线图）
      infer_memory.png        — 推理峰值显存对比（折线图）

    参数:
      json_paths  — benchmark JSON 文件路径列表（每个文件对应一个模型）
      out_dir     — 图片输出目录（默认当前目录）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        # 设置中文字体（按优先级查找）
        candidate_fonts = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei"
        ]

        for font in candidate_fonts:
            if any(font in f.name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = font
                break

        # 解决负号显示问题
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[ERROR] 绘图需要 matplotlib 和 numpy，请先安装：pip install matplotlib numpy")
        return

    import os
    os.makedirs(out_dir, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    reports: list[dict] = []
    for p in json_paths:
        with open(p, encoding="utf-8") as f:
            reports.append(json.load(f))

    if not reports:
        print("[WARN] 未加载到任何数据，跳过绘图。")
        return

    model_names = [r["model_name"] for r in reports]

    # 公共样式
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    def _save(fig: "plt.Figure", name: str):
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] 已保存: {path}")

    def _bar_label(ax, bars, fmt="{:.2f}"):
        """在柱子顶部标注数值。"""
        for bar in bars:
            h = bar.get_height()
            if h and h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h,
                    fmt.format(h), ha="center", va="bottom", fontsize=8,
                )

    # ── 1. 训练峰值显存对比 ───────────────────────────────────────────────────
    # 结构：每个模型两个 bar（含优化器 / 不含优化器），分组排列
    train_data: list[dict] = []
    for r in reports:
        row = {"model": r["model_name"]}
        for tm in r.get("train_memory", []):
            key = "with_opt" if tm.get("with_optimizer") else "no_opt"
            row[key] = tm.get("peak_memory_gib")  # None → OOM
        train_data.append(row)

    if train_data and any("with_opt" in d or "no_opt" in d for d in train_data):
        n = len(train_data)
        x = np.arange(n)
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(6, n * 1.6), 5))
        bars1 = ax.bar(x - w / 2,
                       [d.get("with_opt") or 0 for d in train_data],
                       w, label="含 AdamW 优化器", color=COLORS[0], alpha=0.85)
        bars2 = ax.bar(x + w / 2,
                       [d.get("no_opt") or 0 for d in train_data],
                       w, label="仅前向+反向", color=COLORS[1], alpha=0.85)
        _bar_label(ax, bars1)
        _bar_label(ax, bars2)

        # OOM 标记
        for i, d in enumerate(train_data):
            for offset, key in [(-w / 2, "with_opt"), (w / 2, "no_opt")]:
                if d.get(key) is None:
                    ax.text(i + offset, 0.1, "OOM", ha="center",
                            va="bottom", color="red", fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([d["model"] for d in train_data], rotation=15, ha="right")
        ax.set_ylabel("峰值显存 (GiB)")
        ax.set_title("训练峰值显存对比")
        ax.legend()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        _save(fig, "train_memory.png")

    # ── 2 / 3 / 4. 推理指标折线图 ────────────────────────────────────────────
    # X 轴：bs×plen 组合标签；每条折线代表一个模型
    # 先收集所有出现过的 (batch_size, prompt_len) 组合，保持顺序且去重
    all_combos: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in reports:
        for inf in r.get("inference", []):
            if inf.get("status") == "ok":
                key = (inf["batch_size"], inf["prompt_len"])
                if key not in seen:
                    all_combos.append(key)
                    seen.add(key)
    all_combos.sort(key=lambda t: (t[0], t[1]))
    x_labels = [f"bs{bs}\nplen{pl}" for bs, pl in all_combos]

    def _lookup(report: dict, combo: tuple[int, int], field: str):
        """从单个 report 的 inference 列表中查找对应组合的指标值。"""
        for inf in report.get("inference", []):
            if (inf.get("batch_size"), inf.get("prompt_len")) == combo:
                if inf.get("status") == "ok":
                    return inf.get(field)
        return None  # OOM 或缺失

    infer_specs = [
        ("ttft_ms", "TTFT Comparison", "TTFT avg (ms)", "infer_ttft.png"),
        ("throughput_tokens_per_sec", "Throughput Comparison", "Throughput (tokens/sec)", "infer_throughput.png"),
        ("peak_memory_gib", "Peak Memory Comparison", "Peak Memory (GiB)", "infer_memory.png"),
    ]

    if all_combos:
        x_pos = np.arange(len(all_combos))

        for field, title, ylabel, fname in infer_specs:
            fig, ax = plt.subplots(figsize=(max(7, len(all_combos) * 1.1), 5))

            for idx, (r, name) in enumerate(zip(reports, model_names)):
                ys = [_lookup(r, combo, field) for combo in all_combos]
                # 有效点连线，None（OOM/缺失）用空心红叉标注
                valid_x = [x_pos[i] for i, v in enumerate(ys) if v is not None]
                valid_y = [v for v in ys if v is not None]
                oom_x = [x_pos[i] for i, v in enumerate(ys) if v is None]

                color = COLORS[idx % len(COLORS)]
                if valid_x:
                    ax.plot(valid_x, valid_y, marker="o", label=name,
                            color=color, linewidth=1.8, markersize=5)
                    # 数值标注（避免过密时重叠，仅标有效点）
                    for xi, yi in zip(valid_x, valid_y):
                        ax.annotate(f"{yi:.1f}", (xi, yi),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha="center", fontsize=7, color=color)
                if oom_x:
                    ax.scatter(oom_x, [0] * len(oom_x), marker="x",
                               color="red", s=60, zorder=5)
                    for xi in oom_x:
                        ax.text(xi, 0, "OOM", ha="center", va="bottom",
                                color="red", fontsize=7)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            _save(fig, fname)

    print(f"绘图完成，共生成 {1 + len(infer_specs)} 张图（保存到 {os.path.abspath(out_dir)}）")


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="训练峰值显存 & Python 推理性能测试",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model_path", required=False, default=None,
                   help="模型目录（含 config.json 和权重文件）")
    p.add_argument("--mode", choices=["all", "train_memory", "inference"],
                   default="all", help="测试模式（默认: all）")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                   default="bf16", help="模型精度（默认: bf16）")
    p.add_argument("--device", default="cuda:0", help="目标设备（默认: cuda:0）")

    # 训练显存
    p.add_argument("--train_batch_size", type=int, default=2,
                   help="训练测试 batch size（默认: 2）")
    p.add_argument("--train_seq_len", type=int, default=2048,
                   help="训练测试序列长度（默认: 2048）")

    # 推理
    p.add_argument("--infer_batch_sizes", type=int, nargs="+", default=[1, 4, 8],
                   help="推理 batch size 列表（默认: 1 4 8）")
    p.add_argument("--infer_seq_lens", type=int, nargs="+", default=[256, 1024, 4096],
                   help="推理 prompt 长度列表（默认: 256 1024 4096）")
    p.add_argument("--gen_len", type=int, default=64,
                   help="推理生成 token 数（默认: 64）")
    p.add_argument("--warmup_runs", type=int, default=10,
                   help="推理预热轮数（默认: 10")
    p.add_argument("--measure_runs", type=int, default=50,
                   help="推理计时轮数（默认: 50）")

    # 输出
    p.add_argument("--output", type=str, default=None,
                   help="保存 JSON 结果的路径（可选）")

    # 绘图（独立模式：传入多个已有 JSON，不跑 benchmark）
    p.add_argument("--plot", type=str, nargs="+", default=None,
                   metavar="JSON",
                   help="绘图模式：传入一或多个 benchmark JSON 路径，生成对比图后退出\n"
                        "例: --plot a.json b.json c.json")
    p.add_argument("--plot_dir", type=str, default=".",
                   help="绘图输出目录（默认: 当前目录）")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 绘图独立模式：直接读 JSON，不跑 benchmark ─────────────────────────────
    if args.plot:
        plot_benchmarks(args.plot, out_dir=args.plot_dir)
        return

    if not args.model_path:
        print("[ERROR] 请通过 --model_path 指定模型路径，或通过 --plot 指定 JSON 进行绘图。")
        return

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sep("═")
    print("benchmark_model.py")
    sep("═")
    print(f"  模型路径  : {args.model_path}")
    print(f"  设备      : {device}  |  精度: {args.dtype}  |  模式: {args.mode}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU       : {props.name}  "
              f"({bytes_to_gib(props.total_memory):.1f} GiB 总显存)")
    sep("═")

    report = {
        "model_path": args.model_path,
        "model_name": Path(args.model_path).name,
        "dtype": args.dtype,
        "device": str(device),
        "train_memory": [],
        "inference_meta": {},
        "inference": [],
    }

    # ── 训练峰值显存 ──────────────────────────────────────────────────────────
    if args.mode in ("all", "train_memory"):
        sep()
        print("【1/2】训练峰值显存测试（含优化器 & 仅前向+反向 各一次）")
        sep()
        report["train_memory"] = bench_train_memory(
            args.model_path, device, dtype,
            args.train_batch_size, args.train_seq_len,
        )

    # ── 推理性能 ──────────────────────────────────────────────────────────────
    if args.mode in ("all", "inference"):
        sep()
        print("【2/2】推理性能测试（torch.inference_mode，无推理框架优化）")
        sep()
        meta, infer_results = bench_inference(
            args.model_path, device, dtype,
            args.infer_batch_sizes, args.infer_seq_lens, args.gen_len,
            args.warmup_runs, args.measure_runs,
        )
        report["inference_meta"] = meta
        report["inference"] = infer_results

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print()
    if report["train_memory"]:
        print_train_summary(report["train_memory"])
    if report["inference"]:
        print_inference_summary(report["inference"])
    sep("═")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
