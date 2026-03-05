#!/usr/bin/env python3
"""
benchmark_prefill_decode.py — Prefilling & Decoding 专项性能测试 v2

【三种测试模式】

  ① Prefilling
     输入长度 = seq_len，只做一次 model forward，不生成任何新 token。
     指标：latency(ms)、输入侧吞吐(in-tok/s)、峰值显存(GiB)。
     峰值显存 = 权重 + 激活值（full-attention 的 attention 矩阵 ∝ seq²）。

  ② Decoding — snapshot 模式
     先 prefill kv_len 长度建好 KV cache，然后固定该 cache 不累积，
     每步只喂 1 个 token，重复计时 measure_runs × decode_steps 次取均值。
     反映"在某个固定上下文长度下"的单步 decode 速度。
     指标：latency(ms/step)、gen-tok/s、峰值显存(GiB)。
     峰值显存 = 权重 + KV cache(kv_len) + 单步激活（极小）。

  ③ Decoding — degradation 模式  ★
     从 init_kv_len 个 token 的 prompt 开始，真实地逐步累积 KV cache，
     每生成 sample_every 个 token 记录一次单步耗时和当前显存，
     直到总长度达到 max_kv_len 为止。
     专门用于体现含 full-attention 层的模型"解码速度随上下文增长而衰减"
     的现象，生成 latency-vs-kv_len 衰减曲线。
     指标：每个采样点的 (kv_len, latency_ms, current_mem_gib)，
           以及整个过程的全局峰值显存。

用法示例：
  # 测两个模型，全部三项
  python benchmark_prefill_decode.py \\
      --model_paths /path/to/340M /path/to/1B3 \\
      --batch_sizes 1 4 8 \\
      --seq_lens 2048 8192 16384 32768 \\
      --decode_steps 100 \\
      --warmup_runs 3 --measure_runs 5 \\
      --degrade_init_kv 512 \\
      --degrade_max_kv 32768 \\
      --degrade_sample_every 512 \\
      --output_dir benchmark_output

  # 只测衰减曲线
  python benchmark_prefill_decode.py \\
      --model_paths /path/to/340M \\
      --mode degrade \\
      --batch_sizes 1 \\
      --degrade_init_kv 512 --degrade_max_kv 32768 --degrade_sample_every 512
"""

import argparse
import contextlib
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def bytes_to_gib(n): return n / (1024 ** 3)
def reset_peak(dev): torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
def peak_gib(dev): torch.cuda.synchronize(dev); return bytes_to_gib(torch.cuda.max_memory_allocated(dev))
def current_gib(dev): torch.cuda.synchronize(dev); return bytes_to_gib(torch.cuda.memory_allocated(dev))
def clear_cache(): gc.collect(); torch.cuda.empty_cache()
def sep(c="─", w=76): print(c * w)


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, dtype: torch.dtype, device: torch.device):
    try:
        import fla  # noqa — 注册 flash-linear-attention 自定义架构
        print("  [✓] fla 已导入")
    except ImportError:
        print("  [!] fla 未安装，若为标准架构可忽略")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
        # attn_implementation="flash_attention_3",
    ).to(device).eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ① Prefilling benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_prefill_single(model, device, batch_size, seq_len, warmup, measure) -> dict:
    """
    纯 prefill：输入长度 = seq_len，只跑 forward（不生成任何新 token）。

    峰值显存 = 权重显存 + 单次 forward 激活值峰值。
    对于含 full-attention 层的混合架构，attention 矩阵 ∝ seq²，
    因此峰值显存会随 seq_len 增大而显著增长。
    """
    result = {"batch_size": batch_size, "seq_len": seq_len}
    try:
        vocab = model.config.vocab_size
        ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        mask = torch.ones_like(ids)

        # 预热
        with torch.inference_mode():
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=mask)
        clear_cache()

        # reset_peak 在预热后调用，只统计正式计时阶段的峰值
        reset_peak(device)
        lat_list = []
        with torch.inference_mode():
            for _ in range(measure):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                model(input_ids=ids, attention_mask=mask)
                torch.cuda.synchronize(device)
                lat_list.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(lat_list) / len(lat_list)
        result.update({
            "status": "ok",
            "latency_ms_avg": round(avg_ms, 3),
            "latency_ms_min": round(min(lat_list), 3),
            "latency_ms_max": round(max(lat_list), 3),
            "throughput_input_tok_per_sec": round(batch_size * seq_len / (avg_ms / 1000), 1),
            "peak_mem_gib": round(peak_gib(device), 3),
        })
    except torch.cuda.OutOfMemoryError:
        result.update({"status": "OOM"})
    except Exception as e:
        result.update({"status": f"ERROR: {e}"})
        traceback.print_exc()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ② Decoding — snapshot 模式
# ─────────────────────────────────────────────────────────────────────────────

def bench_decode_snapshot(model, device, batch_size, kv_len, decode_steps, warmup, measure) -> dict:
    """
    固定 KV cache 长度，测在该上下文长度下的稳定单步 decode 速度。

    峰值显存 = 权重显存 + KV cache（kv_len × layers × heads × head_dim × 2 × dtype_bytes）
               + 单步 forward 激活值（只有 1 个 token，非常小）。
    这是该 kv_len 下的"稳态"显存占用，随 kv_len 线性增长。
    """
    result = {"batch_size": batch_size, "kv_len": kv_len, "decode_steps": decode_steps}
    past = None
    prefill_out = None
    try:
        vocab = model.config.vocab_size
        prompt_ids = torch.randint(0, vocab, (batch_size, kv_len), device=device)
        prompt_mask = torch.ones(batch_size, kv_len, device=device, dtype=torch.long)

        # prefill — 建立 KV cache，不计入 decode 计时
        with torch.inference_mode():
            prefill_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
        past = prefill_out.past_key_values
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_mask = torch.ones(batch_size, kv_len + 1, device=device, dtype=torch.long)

        # 预热 decode
        with torch.inference_mode():
            for _ in range(warmup):
                model(input_ids=next_token, attention_mask=step_mask,
                      past_key_values=past, use_cache=True)
        clear_cache()

        # 正式计时：每步复用同一份 past，KV 长度始终钉在 kv_len
        reset_peak(device)
        lat_list = []
        with torch.inference_mode():
            for _ in range(measure * decode_steps):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                model(input_ids=next_token, attention_mask=step_mask,
                      past_key_values=past, use_cache=True)
                torch.cuda.synchronize(device)
                lat_list.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(lat_list) / len(lat_list)
        result.update({
            "status": "ok",
            "latency_ms_avg": round(avg_ms, 3),
            "latency_ms_min": round(min(lat_list), 3),
            "latency_ms_max": round(max(lat_list), 3),
            "throughput_gen_tok_per_sec": round(batch_size / (avg_ms / 1000), 1),
            # 峰值显存 = 权重 + KV cache(kv_len) + 单步激活
            "peak_mem_gib": round(peak_gib(device), 3),
        })

    except torch.cuda.OutOfMemoryError:
        result.update({"status": "OOM"})
    except Exception as e:
        result.update({"status": f"ERROR: {e}"})
        traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            del past, prefill_out
            clear_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ③ Decoding — degradation 模式
# ─────────────────────────────────────────────────────────────────────────────

def bench_decode_degradation(
    model, device, batch_size,
    init_kv_len: int,
    max_kv_len: int,
    sample_every: int,
    warmup_steps: int = 10,
) -> dict:
    """
    真实累积 KV cache，每 sample_every 步记录一次单步耗时和当前显存。
    专用于画 full-attention 模型的 latency 衰减曲线。

    采样点列表结构：
      [{"kv_len": int, "latency_ms": float, "current_mem_gib": float}, ...]

    注意：
      - latency 是该时刻单步 decode 的瞬时值（单次计时），不做多次平均，
        目的是看趋势曲线而非精确均值。
      - current_mem_gib 是该步结束后的已分配显存（权重 + 已累积 KV cache），
        随 kv_len 线性增长。
      - peak_mem_gib 是整个过程的全局峰值（在 max_kv_len 附近取到）。
    """
    result = {
        "batch_size": batch_size,
        "init_kv_len": init_kv_len,
        "max_kv_len": max_kv_len,
        "sample_every": sample_every,
        "samples": [],
    }
    past = None
    cur_len = init_kv_len
    try:
        vocab = model.config.vocab_size
        prompt_ids = torch.randint(0, vocab, (batch_size, init_kv_len), device=device)
        prompt_mask = torch.ones(batch_size, init_kv_len, device=device, dtype=torch.long)

        # prefill — 建立初始 KV cache
        with torch.inference_mode():
            out = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out, prompt_ids, prompt_mask
        clear_cache()

        # 预热若干步（让 CUDA kernel JIT 编译稳定）
        with torch.inference_mode():
            for _ in range(warmup_steps):
                step_mask = torch.ones(batch_size, cur_len + 1, device=device, dtype=torch.long)
                step_out = model(input_ids=next_token, attention_mask=step_mask,
                                 past_key_values=past, use_cache=True)
                past = step_out.past_key_values
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur_len += 1
                del step_out

        reset_peak(device)
        step_counter = 0

        # 真实累积 decode 循环
        with torch.inference_mode():
            while cur_len < max_kv_len:
                step_mask = torch.ones(batch_size, cur_len + 1, device=device, dtype=torch.long)

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                step_out = model(input_ids=next_token, attention_mask=step_mask,
                                 past_key_values=past, use_cache=True)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                past = step_out.past_key_values
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del step_out
                cur_len += 1
                step_counter += 1

                if step_counter % sample_every == 0:
                    result["samples"].append({
                        "kv_len": cur_len,
                        "latency_ms": round(elapsed_ms, 3),
                        "current_mem_gib": round(current_gib(device), 3),
                    })

        result.update({
            "status": "ok",
            "total_steps": step_counter,
            "peak_mem_gib": round(peak_gib(device), 3),
        })

    except torch.cuda.OutOfMemoryError:
        result.update({
            "status": "OOM",
            "oom_at_kv_len": cur_len,
            # 保留 OOM 前已采集的样本，peak 取 OOM 前的最大值
            "peak_mem_gib": round(peak_gib(device), 3),
        })
        with contextlib.suppress(Exception):
            del past
            clear_cache()
    except Exception as e:
        result.update({"status": f"ERROR: {e}"})
        traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            del past
            clear_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 打印汇总
# ─────────────────────────────────────────────────────────────────────────────

def print_prefill_table(results):
    sep()
    print("【Prefilling 汇总】")
    sep()
    print(f"  {'bs':<5} {'seq_len':<9} {'avg(ms)':<10} {'min':<9} {'max':<9} "
          f"{'in-tok/s':<13} {'peak_mem(GiB)':<15} 状态")
    print("  " + "─" * 80)
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['batch_size']:<5} {r['seq_len']:<9} "
                  f"{r['latency_ms_avg']:<10.2f} {r['latency_ms_min']:<9.2f} {r['latency_ms_max']:<9.2f} "
                  f"{r['throughput_input_tok_per_sec']:<13.0f} {r['peak_mem_gib']:<15.3f} ok")
        else:
            print(f"  {r['batch_size']:<5} {r['seq_len']:<9} — {r['status']}")


def print_decode_snapshot_table(results):
    sep()
    print("【Decoding 汇总 — snapshot 模式】")
    sep()
    print(f"  {'bs':<5} {'kv_len':<9} {'avg(ms)':<10} {'min':<9} {'max':<9} "
          f"{'gen-tok/s':<13} {'peak_mem(GiB)':<15} 状态")
    print("  " + "─" * 80)
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['batch_size']:<5} {r['kv_len']:<9} "
                  f"{r['latency_ms_avg']:<10.2f} {r['latency_ms_min']:<9.2f} {r['latency_ms_max']:<9.2f} "
                  f"{r['throughput_gen_tok_per_sec']:<13.1f} {r['peak_mem_gib']:<15.3f} ok")
        else:
            print(f"  {r['batch_size']:<5} {r['kv_len']:<9} — {r['status']}")


def print_degrade_summary(results):
    sep()
    print("【Decoding 汇总 — degradation 模式】")
    sep()
    for r in results:
        samples = r.get("samples", [])
        status = r["status"]
        tag = "" if status == "ok" else f"  [{status}]"
        if samples:
            first, last = samples[0], samples[-1]
            slowdown = last['latency_ms'] / first['latency_ms'] if first['latency_ms'] > 0 else 0
            mem_growth = last['current_mem_gib'] - first['current_mem_gib']
            print(f"  bs={r['batch_size']}{tag}  采样={len(samples)}点  "
                  f"peak_mem={r.get('peak_mem_gib','N/A')}GiB")
            print(f"    latency : {first['latency_ms']:>8.2f} ms → {last['latency_ms']:>8.2f} ms"
                  f"  ({slowdown:.2f}x 衰减)")
            print(f"    mem     : {first['current_mem_gib']:>7.3f} GiB → "
                  f"{last['current_mem_gib']:>7.3f} GiB  (+{mem_growth:.3f} GiB)")
            if status == "OOM":
                print(f"    OOM @ kv_len = {r.get('oom_at_kv_len')}")
        else:
            print(f"  bs={r['batch_size']}{tag}  无采样点")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_reports: list, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[WARN] 未安装 matplotlib/numpy，跳过绘图")
        return

    os.makedirs(out_dir, exist_ok=True)
    for font in ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]:
        if any(font in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] {p}")

    def _annot(ax, xs, ys, color, fmt="{:.1f}"):
        for x, y in zip(xs, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color=color)

    def _xfmt(lens):
        return [f"{v//1024}k" if v >= 1024 else str(v) for v in lens]

    # ── ① Prefill：3 列子图（latency / throughput / peak_mem）─────────────────
    for report in all_reports:
        pf = report.get("prefill", [])
        if not pf:
            continue
        name = report["model_name"]
        batch_sizes = sorted(set(r["batch_size"] for r in pf))
        seq_lens = sorted(set(r["seq_len"] for r in pf if r["status"] == "ok"))

        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        for idx, bs in enumerate(batch_sizes):
            color = COLORS[idx % len(COLORS)]
            rows = {r["seq_len"]: r for r in pf if r["batch_size"] == bs and r["status"] == "ok"}
            xs = [sl for sl in seq_lens if sl in rows]
            if not xs:
                continue
            lats = [rows[sl]["latency_ms_avg"] for sl in xs]
            thrs = [rows[sl]["throughput_input_tok_per_sec"] for sl in xs]
            mems = [rows[sl]["peak_mem_gib"] for sl in xs]
            lbl = f"bs={bs}"
            axes[0].plot(xs, lats, marker="o", label=lbl, color=color)
            axes[1].plot(xs, thrs, marker="o", label=lbl, color=color)
            axes[2].plot(xs, mems, marker="o", label=lbl, color=color)
            _annot(axes[0], xs, lats, color, "{:.0f}")
            _annot(axes[1], xs, thrs, color, "{:.0f}")
            _annot(axes[2], xs, mems, color, "{:.2f}")

        for ax, title, ylabel in [
            (axes[0], f"Prefill Latency — {name}", "Latency (ms)"),
            (axes[1], f"Prefill Throughput — {name}", "Input tokens/sec"),
            (axes[2], f"Prefill Peak Memory — {name}", "Peak Memory (GiB)"),
        ]:
            ax.set_xlabel("seq_len")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(seq_lens)
            ax.set_xticklabels(_xfmt(seq_lens), rotation=30)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        fig.tight_layout()
        _save(fig, f"prefill_{name}.png")

    # ── ② Decode snapshot：3 列子图 ──────────────────────────────────────────
    for report in all_reports:
        dc = report.get("decode_snapshot", [])
        if not dc:
            continue
        name = report["model_name"]
        batch_sizes = sorted(set(r["batch_size"] for r in dc))
        kv_lens = sorted(set(r["kv_len"] for r in dc if r["status"] == "ok"))

        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        for idx, bs in enumerate(batch_sizes):
            color = COLORS[idx % len(COLORS)]
            rows = {r["kv_len"]: r for r in dc if r["batch_size"] == bs and r["status"] == "ok"}
            xs = [kl for kl in kv_lens if kl in rows]
            if not xs:
                continue
            lats = [rows[kl]["latency_ms_avg"] for kl in xs]
            thrs = [rows[kl]["throughput_gen_tok_per_sec"] for kl in xs]
            mems = [rows[kl]["peak_mem_gib"] for kl in xs]
            lbl = f"bs={bs}"
            axes[0].plot(xs, lats, marker="o", label=lbl, color=color)
            axes[1].plot(xs, thrs, marker="o", label=lbl, color=color)
            axes[2].plot(xs, mems, marker="o", label=lbl, color=color)
            _annot(axes[0], xs, lats, color, "{:.1f}")
            _annot(axes[1], xs, thrs, color, "{:.0f}")
            _annot(axes[2], xs, mems, color, "{:.2f}")

        for ax, title, ylabel in [
            (axes[0], f"Decode Latency/step (snapshot) — {name}", "Latency (ms)"),
            (axes[1], f"Decode Throughput (snapshot) — {name}", "Gen tokens/sec"),
            (axes[2], f"Decode Peak Memory (snapshot) — {name}", "Peak Memory (GiB)"),
        ]:
            ax.set_xlabel("kv_len")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(kv_lens)
            ax.set_xticklabels(_xfmt(kv_lens), rotation=30)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        fig.tight_layout()
        _save(fig, f"decode_snapshot_{name}.png")

    # ── ③ Degradation 曲线：latency & mem vs kv_len ───────────────────────────
    for report in all_reports:
        dg = report.get("decode_degrade", [])
        if not dg:
            continue
        name = report["model_name"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for idx, r in enumerate(dg):
            samples = r.get("samples", [])
            if not samples:
                continue
            bs = r["batch_size"]
            color = COLORS[idx % len(COLORS)]
            xs = [s["kv_len"] for s in samples]
            lat_ys = [s["latency_ms"] for s in samples]
            mem_ys = [s["current_mem_gib"] for s in samples]
            lbl = f"bs={bs}" + (" (OOM)" if r["status"] == "OOM" else "")
            # latency 用散点+趋势线，凸显衰减趋势
            axes[0].scatter(xs, lat_ys, s=8, color=color, alpha=0.6, label=lbl)
            # 叠加滑动平均趋势线（窗口 = 10% 的采样点数，至少 5）
            win = max(5, len(lat_ys) // 10)
            if len(lat_ys) >= win:
                smoothed = np.convolve(lat_ys, np.ones(win) / win, mode="valid")
                smooth_xs = xs[win // 2: win // 2 + len(smoothed)]
                axes[0].plot(smooth_xs, smoothed, linewidth=2, color=color, alpha=0.9)
            axes[1].plot(xs, mem_ys, linewidth=1.8, color=color, alpha=0.9, label=lbl)

        for ax, title, ylabel in [
            (axes[0],
             f"Decode Latency Degradation — {name}\n"
             f"(scatter=raw, line=smoothed moving avg)",
             "Latency per step (ms)"),
            (axes[1],
             f"Decode Memory Growth — {name}\n(current allocated mem vs kv_len)",
             "Current Memory (GiB)"),
        ]:
            ax.set_xlabel("KV cache length (tokens)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(linestyle="--", alpha=0.4)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        fig.tight_layout()
        _save(fig, f"decode_degrade_{name}.png")

    # ── 多模型对比图 ─────────────────────────────────────────────────────────
    if len(all_reports) > 1:
        # snapshot 对比（bs=1）：latency + peak_mem
        for key, xk, yk, title, ylabel, fname in [
            ("prefill",         "seq_len", "latency_ms_avg",
             "Prefill Latency Comparison (bs=1)", "ms", "cmp_prefill_lat.png"),
            ("prefill",         "seq_len", "peak_mem_gib",
             "Prefill Peak Memory Comparison (bs=1)", "GiB", "cmp_prefill_mem.png"),
            ("decode_snapshot", "kv_len",  "latency_ms_avg",
             "Decode Latency Comparison (bs=1, snapshot)", "ms/step", "cmp_decode_lat.png"),
            ("decode_snapshot", "kv_len",  "peak_mem_gib",
             "Decode Peak Memory Comparison (bs=1, snapshot)", "GiB", "cmp_decode_mem.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            for idx, report in enumerate(all_reports):
                rows = [r for r in report.get(key, [])
                        if r.get("batch_size") == 1 and r.get("status") == "ok"]
                rows.sort(key=lambda r: r[xk])
                if not rows:
                    continue
                xs = [r[xk] for r in rows]
                ys = [r[yk] for r in rows]
                color = COLORS[idx % len(COLORS)]
                ax.plot(xs, ys, marker="o", label=report["model_name"], color=color)
                _annot(ax, xs, ys, color)
            ax.set_xlabel(xk)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            _save(fig, fname)

        # degradation 对比（bs=1）：latency 衰减 + 显存增长
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for idx, report in enumerate(all_reports):
            dg = report.get("decode_degrade", [])
            r1 = next((r for r in dg if r["batch_size"] == 1), None)
            if not r1 or not r1.get("samples"):
                continue
            color = COLORS[idx % len(COLORS)]
            lbl = report["model_name"] + ("" if r1["status"] == "ok" else " (OOM)")
            xs = [s["kv_len"] for s in r1["samples"]]
            lys = [s["latency_ms"] for s in r1["samples"]]
            mys = [s["current_mem_gib"] for s in r1["samples"]]
            axes[0].scatter(xs, lys, s=6, color=color, alpha=0.5)
            win = max(5, len(lys) // 10)
            if len(lys) >= win:
                smoothed = np.convolve(lys, np.ones(win) / win, mode="valid")
                sxs = xs[win // 2: win // 2 + len(smoothed)]
                axes[0].plot(sxs, smoothed, linewidth=2, color=color, alpha=0.9, label=lbl)
            axes[1].plot(xs, mys, linewidth=1.8, color=color, alpha=0.9, label=lbl)
        for ax, title, ylabel in [
            (axes[0], "Decode Latency Degradation Comparison (bs=1)", "ms/step"),
            (axes[1], "Decode Memory Growth Comparison (bs=1)", "GiB"),
        ]:
            ax.set_xlabel("KV cache length")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(linestyle="--", alpha=0.4)
        fig.tight_layout()
        _save(fig, "cmp_degrade.png")

    print(f"\n绘图完成，图片保存至: {os.path.abspath(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_one_model(model_path, dtype, device, args) -> dict:
    model_name = Path(model_path).name
    sep("═")
    print(f"模型: {model_name}")
    sep("═")

    report = {
        "model_path": model_path,
        "model_name": model_name,
        "dtype": args.dtype,
        "device": str(device),
        "prefill": [],
        "decode_snapshot": [],
        "decode_degrade": [],
    }

    # ── ① Prefilling ──────────────────────────────────────────────────────────
    if args.mode in ("all", "prefill"):
        sep()
        print("【1/3 Prefilling】seq_len = 输入长度，纯 forward，不生成新 token")
        sep()
        try:
            clear_cache()
            model, _ = load_model(model_path, dtype, device)
            total_params = sum(p.numel() for p in model.parameters())
            load_mem = current_gib(device)
            print(f"  参数量: {total_params/1e6:.1f}M  模型加载显存: {load_mem:.3f} GiB")

            for bs in args.batch_sizes:
                for sl in args.seq_lens:
                    print(f"  bs={bs}, seq_len={sl:>6} ...", end=" ", flush=True)
                    r = bench_prefill_single(model, device, bs, sl,
                                             args.warmup_runs, args.measure_runs)
                    report["prefill"].append(r)
                    if r["status"] == "ok":
                        print(f"avg={r['latency_ms_avg']:.1f}ms  "
                              f"{r['throughput_input_tok_per_sec']:.0f} in-tok/s  "
                              f"peak_mem={r['peak_mem_gib']:.3f}GiB")
                    else:
                        print(r["status"])
        finally:
            with contextlib.suppress(Exception):
                del model
            clear_cache()
        print_prefill_table(report["prefill"])

    # ── ② Decode snapshot ─────────────────────────────────────────────────────
    if args.mode in ("all", "decode"):
        sep()
        print("【2/3 Decoding — snapshot】固定 KV 长度，测稳定单步速度与显存")
        sep()
        try:
            clear_cache()
            model, _ = load_model(model_path, dtype, device)
            print(f"  模型加载显存: {current_gib(device):.3f} GiB")

            for bs in args.batch_sizes:
                for kv in args.seq_lens:
                    print(f"  bs={bs}, kv_len={kv:>6} ...", end=" ", flush=True)
                    r = bench_decode_snapshot(model, device, bs, kv,
                                              args.decode_steps,
                                              args.warmup_runs, args.measure_runs)
                    report["decode_snapshot"].append(r)
                    if r["status"] == "ok":
                        print(f"avg={r['latency_ms_avg']:.2f}ms  "
                              f"{r['throughput_gen_tok_per_sec']:.1f} gen-tok/s  "
                              f"peak_mem={r['peak_mem_gib']:.3f}GiB")
                    else:
                        print(r["status"])
        finally:
            with contextlib.suppress(Exception):
                del model
            clear_cache()
        print_decode_snapshot_table(report["decode_snapshot"])

    # ── ③ Decode degradation ──────────────────────────────────────────────────
    if args.mode in ("all", "degrade"):
        sep()
        print("【3/3 Decoding — degradation】真实累积 KV cache，观察速度衰减曲线")
        sep()
        total_decode_steps = args.degrade_max_kv - args.degrade_init_kv - args.degrade_warmup_steps
        n_samples = total_decode_steps // args.degrade_sample_every
        print(f"  kv: {args.degrade_init_kv} → {args.degrade_max_kv}  "
              f"sample_every={args.degrade_sample_every}  "
              f"预计 decode ~{total_decode_steps} 步，采集 ~{n_samples} 点")
        try:
            clear_cache()
            model, _ = load_model(model_path, dtype, device)
            print(f"  模型加载显存: {current_gib(device):.3f} GiB")

            for bs in args.batch_sizes:
                print(f"  bs={bs} ...", flush=True)
                r = bench_decode_degradation(
                    model, device, bs,
                    init_kv_len=args.degrade_init_kv,
                    max_kv_len=args.degrade_max_kv,
                    sample_every=args.degrade_sample_every,
                    warmup_steps=args.degrade_warmup_steps,
                )
                report["decode_degrade"].append(r)
                samples = r.get("samples", [])
                if samples:
                    first, last = samples[0], samples[-1]
                    slowdown = last['latency_ms'] / first['latency_ms'] if first['latency_ms'] > 0 else 0
                    status_tag = f"  [{r['status']}]" if r["status"] != "ok" else ""
                    print(f"  完成{status_tag}  steps={r.get('total_steps', len(samples)*args.degrade_sample_every)}  "
                          f"采样={len(samples)}点  peak_mem={r.get('peak_mem_gib','N/A')}GiB")
                    print(f"    latency: {first['latency_ms']:.2f}ms → "
                          f"{last['latency_ms']:.2f}ms  ({slowdown:.2f}x 衰减)")
                elif r["status"] == "OOM":
                    print(f"  OOM @ kv_len={r.get('oom_at_kv_len')}，无有效采样点")
                else:
                    print(f"  {r['status']}")
        finally:
            with contextlib.suppress(Exception):
                del model
            clear_cache()
        print_degrade_summary(report["decode_degrade"])

    return report


def parse_args():
    p = argparse.ArgumentParser(
        description="Prefilling & Decoding 专项性能测试（含峰值显存 & 衰减曲线）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model_paths", nargs="+", required=True,
                   help="一个或多个模型目录")
    p.add_argument("--mode",
                   choices=["all", "prefill", "decode", "degrade"],
                   default="all",
                   help="all=全部三项  prefill=仅预填充  decode=仅snapshot  degrade=仅衰减曲线")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda:0")

    # prefill & decode snapshot 共用
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--seq_lens", type=int, nargs="+",
                   default=[2048, 8192, 16384, 32768],
                   help="prefill = 输入长度；decode snapshot = KV cache 长度")
    p.add_argument("--decode_steps", type=int, default=100,
                   help="snapshot 模式每组合的计时步数（实际样本数 = decode_steps × measure_runs）")
    p.add_argument("--warmup_runs", type=int, default=3)
    p.add_argument("--measure_runs", type=int, default=5)

    # degradation 专用
    p.add_argument("--degrade_init_kv", type=int, default=512,
                   help="degradation 起始 KV 长度（默认 512）")
    p.add_argument("--degrade_max_kv", type=int, default=32768,
                   help="degradation 最大 KV 长度（默认 32768）")
    p.add_argument("--degrade_sample_every", type=int, default=512,
                   help="每生成多少 token 采样一次 latency & mem（默认 512）\n"
                        "建议：想要更平滑的曲线可设为 128 或 256")
    p.add_argument("--degrade_warmup_steps", type=int, default=10,
                   help="degradation 正式计时前的预热 decode 步数（默认 10）")

    p.add_argument("--output_dir", default="benchmark_output")
    p.add_argument("--no_plot", action="store_true", help="不生成图片")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    sep("═")
    print("benchmark_prefill_decode.py  v2")
    sep("═")
    print(f"  设备: {device}  精度: {args.dtype}  模式: {args.mode}")
    print(f"  seq_lens    : {args.seq_lens}")
    print(f"  batch_sizes : {args.batch_sizes}")
    print(f"  decode snapshot: steps={args.decode_steps} × runs={args.measure_runs} "
          f"= {args.decode_steps * args.measure_runs} 样本/组合")
    print(f"  degradation : kv {args.degrade_init_kv}→{args.degrade_max_kv}, "
          f"sample_every={args.degrade_sample_every}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name}  ({bytes_to_gib(props.total_memory):.1f} GiB)")

    all_reports = []
    for mp in args.model_paths:
        report = run_one_model(mp, dtype, device, args)
        all_reports.append(report)
        json_path = os.path.join(args.output_dir, f"{report['model_name']}_bench.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON 已保存: {json_path}")

    if not args.no_plot:
        sep("═")
        print("生成图表 ...")
        plot_results(all_reports, args.output_dir)

    sep("═")
    print("全部完成。")


if __name__ == "__main__":
    main()
