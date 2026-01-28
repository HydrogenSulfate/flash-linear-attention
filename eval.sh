# 设置预训练模型路径
MODEL='/root/paddlejob/workspace/env_run/output/hesensen/flame/exp/gdn-1B-100B/batch16.seqlen4096.warmup2048.update1.steps204800.lr4e-4'

# 单卡测试
CUDA_VISIBLE_DEVICES=0 \
  python -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,trust_remote_code=True  \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,copa,openbookqa,social_iqa,sciq \
    --batch_size 128  \
    --num_fewshot 0  \
    --device cuda \
    --trust_remote_code

# 多卡测试
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True  \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,copa,openbookqa,social_iqa,sciq \
    --batch_size 128  \
    --num_fewshot 0  \
    --device cuda \
    --trust_remote_code
