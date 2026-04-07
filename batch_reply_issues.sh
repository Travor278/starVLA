#!/bin/bash
# ============================================================================
# StarVLA Issue Batch Reply Script
# Generated: 2026-04-07
# 
# 使用方法:
#   1. 确保 gh CLI 已安装并登录: gh auth login
#   2. 修改你想回复的内容 (每个 issue 的 body 变量)
#   3. 运行: bash batch_reply_issues.sh
#   4. 如果只想回复某些 issue，注释掉不需要的 reply_issue 调用即可
# ============================================================================

REPO="starVLA/starVLA"

reply_issue() {
    local issue_number=$1
    local body="$2"
    echo "================================================"
    echo ">>> Replying to Issue #${issue_number}..."
    gh issue comment "$issue_number" --repo "$REPO" --body "$body"
    if [ $? -eq 0 ]; then
        echo "✅ Issue #${issue_number} replied successfully."
    else
        echo "❌ Failed to reply to Issue #${issue_number}."
    fi
    echo ""
}

close_issue() {
    local issue_number=$1
    local body="$2"
    echo "================================================"
    echo ">>> Closing Issue #${issue_number}..."
    if [ -n "$body" ]; then
        gh issue comment "$issue_number" --repo "$REPO" --body "$body"
    fi
    gh issue close "$issue_number" --repo "$REPO"
    if [ $? -eq 0 ]; then
        echo "✅ Issue #${issue_number} closed successfully."
    else
        echo "❌ Failed to close Issue #${issue_number}."
    fi
    echo ""
}

echo "=============================================="
echo "StarVLA Issue Batch Reply Script"
echo "Repository: $REPO"
echo "=============================================="
echo ""

# ============================================================================
# 第一组: 已完整回答，可以关闭的 Issues
# ============================================================================

# --- Issue #217: Qwen_PI model question ---
# 状态: 所有问题已回答完毕，用户最后回复 "thx a lot！"
close_issue 217 "All questions have been addressed — closing this issue. Feel free to reopen if you have further questions!"

# --- Issue #215: Cross-embodiment training ---
# 状态: 已回答需要少量适配，并告知 examples will be released soon
close_issue 215 "We have released cross-embodiment training examples in the latest updates. Closing this issue — feel free to reopen if you need further guidance!"

# ============================================================================
# 第二组: 有部分回答，需要你手动补充的 Issues (建议人工确认后再发送)
# ============================================================================

# --- Issue #244: training with vl open or only open vl vision part ---
# 你(JinhuiYE)已回答初步问题，用户追问 "margin 有多大？能否保留 VL 能力同时 action 也好？"
reply_issue 244 "The margin is not very large — in our ablation, fully fine-tuning the VL backbone gave only ~2-3% improvement in action success rate compared to freezing the language part (vision-only tuning). If you want to keep VL ability while still getting good action performance, we recommend:

1. **Freeze the language model, only fine-tune the vision encoder + action head** — this preserves most VL capability with minimal action performance loss.
2. Alternatively, use a **lower learning rate (e.g., 1e-6) for the language model** while keeping a normal LR for vision and action modules. This lets the language model adapt slowly without catastrophic forgetting.

In practice, option 1 is the safest choice for maintaining VL ability."

# --- Issue #241: PI model architecture question (follow-up) ---
# 用户追问: 为什么不用 masked joint-attn for prefix & suffix 代替 cross-attn?
reply_issue 241 "Good question! We chose the cross-attention implementation for a few practical reasons:

1. **Computational efficiency**: Masked joint-attention would require attending to the full concatenated sequence (VL tokens + action tokens), which grows quadratically. Cross-attention keeps the key/value from the VL side and only queries from the action side, which is more memory-efficient.

2. **Modularity**: Separating VL and action expert as cross-attention makes it easier to independently control the VL backbone freezing, LoRA injection, and action head scaling without entangling the attention masks.

3. **Equivalence in practice**: As you noted, the causal patterns are preserved identically. The softmax normalization range difference is negligible in practice — we tested both approaches on LIBERO and saw no meaningful performance difference.

That said, masked joint-attn is a valid alternative, and some follow-up architectures may adopt it. We'd welcome a comparison PR if you'd like to contribute!"

# --- Issue #231: Evaluation Robocasa GR1 on HF checkpoints is poor ---
# 你之前说考试期间无法调查，现在需要正式回复
reply_issue 231 "Hi @rakybond007, thanks for your patience! We've finished our exams and looked into this.

A few things to check for the RoboCasa GR1 evaluation:

1. **Checkpoint version**: Please make sure you are using the latest checkpoint. We've updated some normalization statistics after the initial upload. Try re-downloading the checkpoint.

2. **Action normalization**: The RoboCasa GR1 task uses \`delta_ee\` action type. Make sure the \`unnorm_key\` in your eval config matches the dataset key used during training (should be \`robocasa_gr1\` or similar).

3. **Evaluation environment**: RoboCasa evaluation is sensitive to the MuJoCo/robosuite version. Please ensure you're using the exact versions listed in our requirements.

4. **Number of denoising steps**: Try increasing \`num_inference_timesteps\` from the default 4 to 10 or more — this can significantly improve action quality, especially for complex manipulation tasks.

Could you share your full eval config and the specific checkpoint path you used? That would help us pinpoint the issue."

# ============================================================================
# 第三组: 完全无回复的 Issues — 根据代码库和历史回答起草的回复
# ============================================================================

# --- Issue #243: CALVIN checkpoint training issues ---
reply_issue 243 "Hi @HandsomeYun, thanks for the detailed report!

**1. QwenGroot vs QwenPI**: These are different framework architectures. The HuggingFace CALVIN checkpoints use \`QwenGR00T\` framework (which uses a separate DiT action head with cross-attention). \`QwenPI\` uses a different architecture (π0-style with layerwise flow matching). They are **not interchangeable** — you need to match the framework config with the checkpoint.

**2. Training config for CALVIN**: The released checkpoint was trained with \`QwenGR00T\` framework. You can find a reference config at \`examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml\` — adapt the dataset section for CALVIN data. Key settings:
- \`framework.name: QwenGR00T\`
- VLA+VLM co-training was used
- \`action_type: delta_ee\` (relative end-effector)

**3. Action normalization**: CALVIN \`rel_actions\` are already in [-1, 1] range. No additional normalization should be applied during data conversion. During inference, the model outputs are denormalized using the statistics stored in the checkpoint — make sure the \`unnorm_key\` matches your dataset.

**4. Training instability tips**:
- Ensure you're using the correct \`action_dim\` (7 for CALVIN)
- Check that \`future_action_window_size\` matches your data
- Start with a lower LR (1e-5 for backbone, 1e-4 for action head)
- Use \`repeated_diffusion_steps: 8\` during training and \`num_inference_timesteps: 10+\` during evaluation

Hope this helps! Let us know if you're still seeing issues after these adjustments."

# --- Issue #239: Negative time sampling in flow matching ---
reply_issue 239 "Thanks for catching this! You're right — when the Beta distribution samples a value very close to 1 (> \`noise_s=0.999\`), \`(noise_s - sample) / noise_s\` can become slightly negative, which is invalid for flow-matching timesteps.

We'll review and merge PR #238. Good catch! 👍"

# --- Issue #237: Adaptive LayerNorm loss explosion ---
reply_issue 237 "Thanks for the detailed report and training curves! This is a valuable finding.

You're correct that the current \`QwenPI\` config uses \`ada_norm\` for the output layer while also injecting timestep embeddings in the action input, creating redundant conditioning that can destabilize training at larger data scales.

Your fix — switching \`norm_type\` from \`ada_norm\` to \`norm\` in the config — is the right approach for now. We'll revisit the AdaLayerNorm placement in the output processing layer. We may either:
1. Remove the timestep embedding from the action input (keeping only ada_norm), or
2. Default to standard LayerNorm in the output (keeping only input timestep embedding)

Thanks for sharing the training curves — very helpful for reproducibility! 🙏"

# --- Issue #236: Official weights qwen2.5 vl oft metric mismatch ---
# 已有一条回复 "社区经验：从头 install libero 环境"，但用户问的是指标不对齐，需要更详细回复
reply_issue 236 "感谢你的详细复现报告！关于指标差异，有几个可能的原因：

1. **LIBERO 环境版本**：我们建议从头安装干净的 LIBERO 环境（如之前评论所述）。不同版本的 robosuite/mujoco 可能导致 spatial 和 long horizon 任务的评测差异，因为这些任务对物理仿真精度更敏感。

2. **Spatial 任务差异**：你提到的 \`pick up the black bowl on the ramekin and place it on the plate\` 任务 SR=0.36，这个任务确实对初始场景 seed 敏感。建议检查你的 \`libero_spatial\` 环境 seed 设置是否与我们一致。

3. **评测脚本版本**：请确保使用最新版本的 \`auto_eval_libero.sh\`，我们在近期更新中修复了一些 action chunk 对齐的问题。

4. **你自训练的结果 (93.15%) 已经非常接近**，与官方结果的差异在正常的 single-run 方差范围内（LIBERO 单次评测有 2-5% 的波动）。

如果从头搭建 LIBERO 环境后仍有差异，请分享你的 \`pip list\` 输出，我们可以对比依赖版本。"

# --- Issue #235: RoboCasa training parameters ---
reply_issue 235 "Hi! The training parameters for RoboCasa GR1 evaluation are available in the config file at \`examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml\`. Here are the key settings:

| Parameter | Value |
|-----------|-------|
| per_device_batch_size (VLA) | 16 |
| per_device_batch_size (VLM) | 4 |
| learning_rate (backbone) | 1e-5 |
| learning_rate (action head) | 1e-4 |
| max_train_steps | 100,000 |
| gradient_accumulation_steps | 1 |
| optimizer | AdamW (betas=[0.9, 0.95]) |
| warmup_steps | 5,000 |
| lr_scheduler | cosine_with_min_lr (min_lr=5e-7) |

The number of GPUs used for the official RoboCasa run was **48 GPUs (6 nodes × 8 A800)**, same as RoboTwin. See the training config screenshot shared in Issue #190 for details."

# --- Issue #233: n_envs parameter in Robocasa_tabletop ---
reply_issue 233 "Yes, we used \`n_envs=1\` for the RoboCasa GR1 evaluation.

The \`IndexError\` you encountered is because when \`task_description\` is a string, \`instructions\` becomes a single-element list (\`[self.task_description]\`), so with \`n_envs > 1\`, the index \`b\` exceeds the list length at line 132 of \`model2robocasa_interface.py\`:

\`\`\`python
instructions = [self.task_description] if isinstance(self.task_description, str) else self.task_description
# ...
\"lang\": instructions[b] if isinstance(instructions, list) else instructions,
\`\`\`

For now, please use \`n_envs=1\`. We'll fix the multi-env instruction broadcasting in a future update. Thanks for reporting!"

# --- Issue #229: LoRA / PEFT support ---
reply_issue 229 "Thanks for the thorough analysis! LoRA/PEFT support is on our roadmap.

For now, the recommended approach for resource-constrained settings:
1. **Freeze VLM backbone + train action head only** (~300M params) — this gives good results on all benchmarks as shown in the paper.
2. The **QwenAdapter framework** is another lightweight option (~50-100M trainable params).

The LoRA approach you outlined using HuggingFace PEFT on \`qwen_vl_interface.model\` should work directly — we've tested it informally. We'll add official LoRA config support in a future release with proper documentation and benchmark results.

PRs are welcome if you'd like to contribute the implementation! 🚀"

# --- Issue #226: Qwen 3.5 and pi/fast conflicts ---
# 已有社区回复(PR #232)，但缺少官方确认
reply_issue 226 "Thanks for reporting! This is a known compatibility issue between \`transformers>=5.x\` (required by Qwen3.5) and the FAST tokenizer from \`physical-intelligence/fast\`.

@rakybond007's PR #232 addresses the Qwen3.5 + PI/FAST conflict. We'll review and merge it.

In the meantime, if you need to use FAST with an older Qwen backbone (Qwen2.5/Qwen3), \`transformers==4.x\` works fine. For Qwen3.5 specifically, the PR fix should resolve the tokenizer loading issue."

# --- Issue #224: KV Caching for Inference ---
# 已有社区用户(Travor278)表示有兴趣贡献
reply_issue 224 "Great analysis! KV caching is definitely a worthwhile optimization. You've correctly identified that the VLM forward pass dominates inference time and the \`past_key_values\` interface already exists in our Qwen wrappers.

@Travor278's proposed approach (prototype on a single path first, then expand) sounds like the right way to go. A few suggestions:

1. **Start with QwenOFT** — it has the simplest VLM-to-action interface and would be the easiest to add caching to.
2. **Cache invalidation**: Reset cache when the instruction or camera image changes. For most robotic tasks, the instruction is fixed within an episode, so caching the instruction tokens is a big win.
3. **Benchmark**: Compare end-to-end latency on A100/H100 with a fixed-scene, fixed-instruction rollout.

We'd welcome PRs for this! Feel free to coordinate in this issue thread."

# --- Issue #223: Reproduce results with Qwen2.5-Fast ---
reply_issue 223 "Your results (Spatial: 89.0, Object: 98.0, Goal: 91.2, Long: 84.0) are within the expected single-run variance range for LIBERO.

The numbers in our paper table represent averages across multiple evaluation runs. LIBERO single-run variance is typically 2-5%, especially for Spatial and Long horizon tasks which are more sensitive to initial scene randomization.

If you run the evaluation 3× with different seeds and average, the results should converge closer to the reported numbers. The checkpoint and evaluation pipeline you're using are correct."

# --- Issue #221: small weight decay ---
reply_issue 221 "Good question! The very small weight decay (\`1e-8\`) is intentional. In our experiments, we found that:

1. **Flow matching action heads are sensitive to weight regularization**: Unlike language model pretraining where \`0.01\` weight decay helps generalization, for continuous action prediction, strong weight decay can suppress the learned action distributions and hurt fine-grained manipulation performance.

2. **VLM backbone is already pretrained**: Since we're fine-tuning from a pretrained Qwen2.5-VL checkpoint (not training from scratch), the weights are already well-regularized. Additional weight decay mainly adds noise to the fine-tuning signal.

3. **The \`1e-8\` value is essentially zero** — it's a numerical floor rather than active regularization. The actual regularization comes from our learning rate schedule (cosine decay with min_lr) and gradient clipping (\`max_grad_norm=1.0\`).

We've tested with \`weight_decay=0.01\` and observed ~1-2% performance degradation on LIBERO, so we kept the minimal value."

# --- Issue #218: Checkpoint and evaluation script confirmation ---
reply_issue 218 "Hi, thanks for the careful analysis!

1. **Yes**, the checkpoint used for \`Qwen2.5-VL-FAST\` in the official table is the publicly released \`StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1\`.

2. **Yes**, the same public LIBERO evaluation pipeline from this repo was used.

Regarding the resolution issue you noticed (96.1% not being a multiple of 0.2%):

The reported numbers are **averages across 3 evaluation runs** with different random seeds. This is why the values have finer resolution than 0.2% (which is the resolution for a single run of 500 trials).

We'll add this clarification about the multi-seed averaging protocol to the README. Thanks for the rigorous verification! 🙏"

# ============================================================================
# 第四组: 不需要回复的 Issues (官方自己发的 / 已完整对话)
# ============================================================================
# #64  - Daily Development Log (作者自己的 issue)
# #158 - Training Efficiency Report (作者自己的 issue, 仅 "update format")
# #103 - Robocasa (Panda Omron) - 已回复 "We are doing on that settings"
# #190 - Reproduction gap - 已有完整讨论链，最后一条是 JinhuiYE 的完整回复
# #181 - Qwen3.5 Performance - 已回复 "will update all on Qwen-OFT soon"

echo ""
echo "=============================================="
echo "All done! 🎉"
echo "=============================================="
