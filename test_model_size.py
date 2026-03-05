import argparse

import torch
from flame.tools.utils import get_nparams_and_flops
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Compute model params and FLOPs")

    # config 文件路径
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to model config json file",
    )

    # 序列长度
    parser.add_argument(
        "-l", "--seq_len",
        type=int,
        default=4096,
        help="Sequence length used for FLOPs calculation",
    )

    # 是否打印模型结构
    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Show model structure",
        default=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_config = AutoConfig.from_pretrained(args.config)

    # meta device 创建模型（不占显存）
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model,
        model_config,
        args.seq_len,
    )
    if args.show:
        print("\033[32m" + str(model) + "\033[0m")
    print(
        f"model_param_count = {model_param_count:,}, "
        f"num_flops_per_token = {num_flops_per_token:,}"
    )


if __name__ == "__main__":
    main()
