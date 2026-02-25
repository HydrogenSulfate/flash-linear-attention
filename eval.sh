set -ex

# MODEL='/root/paddlejob/workspace/env_run/output/hesensen/flame/exp/qwen3_5_kda-340M-15B/kda.batch8.seqlen4096.warmup1024.update4.steps28672.lr1e-3'
MODEL='/root/paddlejob/workspace/env_run/output/hesensen/flame/exp/qwen3_5-340M-15B/batch8.seqlen4096.warmup1024.update4.steps28672.lr1e-3'

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,dtype=bfloat16,max_length=4096,trust_remote_code=True  \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,social_iqa \
    --device cuda \
    --batch_size auto \
    --trust_remote_code

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,dtype=bfloat16,max_length=4096,trust_remote_code=True  \
    --tasks niah_single_1,niah_single_2,niah_single_3 \
    --metadata='{"max_seq_lengths":[2048,4096]}' \
    --device cuda \
    --batch_size auto \
    --trust_remote_code
