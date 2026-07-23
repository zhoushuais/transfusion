# TransFusion KITTI Single-H800 Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固化 TransFusion KITTI 三阶段单 H800 梯度累积训练协议，并用独立的正式产物完成 run1 训练、checkpoint 衔接和完整 KITTI AP40 验证。

**Architecture:** 只在三个项目配置中增加 MMEngine `OptimWrapper.accumulative_counts`，用 micro-batch 4/6/2 和累积 4/8/8 保持名义等效 batch 16/48/16；模型、数据划分、冻结策略、学习率和调度曲线保持不变。README 作为服务器操作入口，明确 FP32、seed、独立 work dir、恢复规则、spconv2 metadata 验收和完整评估流程；配置测试防止训练协议漂移。

**Tech Stack:** Python 3.8、PyTorch 2.1.2+cu118、CUDA 11.8、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.2.0、MMDetection3D 1.4.0、spconv 2.3.6、pytest、单张 NVIDIA H800。

---

## File Map

```text
projects/TransFusionKITTI/
├── configs/
│   ├── r50_fpn_kitti_2d.py       # Stage 0：micro-batch 4、累积 4
│   ├── transfusion_l_kitti.py     # Stage 1：micro-batch 6、累积 8
│   └── transfusion_lc_kitti.py    # Stage 2：micro-batch 2、累积 8、关闭 auto LR scale
├── tests/
│   └── test_configs.py            # 三阶段 batch、LR、schedule、epoch、best metric 契约
└── README.md                       # accumulation smoke、正式 run1、恢复、合并和评估命令
```

正式产物固定使用以下路径，不能复用已有 smoke 目录：

```text
work_dirs/r50_fpn_kitti_2d_formal_run1/
work_dirs/transfusion_l_kitti_formal_run1/
checkpoints/transfusion_kitti_stage2_formal_run1_init.pth
work_dirs/transfusion_lc_kitti_formal_run1/
```

### Task 1: Add Failing Formal-Protocol Config Tests

**Files:**
- Modify: `projects/TransFusionKITTI/tests/test_configs.py`
- Test: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Add a shared nominal-batch assertion helper**

在 imports 后加入：

```python
def assert_batch_protocol(cfg, micro_batch, accumulation, nominal_batch):
    assert cfg.train_dataloader.batch_size == micro_batch
    assert cfg.optim_wrapper.get('accumulative_counts') == accumulation
    assert micro_batch * accumulation == nominal_batch
```

- [ ] **Step 2: Extend the Stage 1 test with the complete training contract**

在 `test_lidar_config_contract()` 末尾加入：

```python
    assert_batch_protocol(cfg, 6, 8, 48)
    assert cfg.train_cfg.max_epochs == 40
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.0018)
    assert cfg.auto_scale_lr.enable is False
    assert [scheduler.type for scheduler in cfg.param_scheduler] == [
        'CosineAnnealingLR',
        'CosineAnnealingLR',
        'CosineAnnealingMomentum',
        'CosineAnnealingMomentum',
    ]
    assert [(scheduler.begin, scheduler.end)
            for scheduler in cfg.param_scheduler] == [
                (0, 16),
                (16, 40),
                (0, 16),
                (16, 40),
            ]
```

- [ ] **Step 3: Extend the Stage 2 test with the complete training contract**

在 `test_lc_config_uses_single_camera_without_geometry_augmentation()` 末尾加入：

```python
    assert_batch_protocol(cfg, 2, 8, 16)
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(1e-4)
    assert cfg.param_scheduler[0].type == 'OneCycleLR'
    assert cfg.param_scheduler[0].total_steps == 6
    assert cfg.param_scheduler[0].eta_max == pytest.approx(1e-3)
    assert cfg.auto_scale_lr.enable is False
    assert cfg.default_hooks.checkpoint.save_best == (
        'Kitti metric/pred_instances_3d/KITTI/'
        'Overall_3D_AP40_moderate')
```

- [ ] **Step 4: Extend the Stage 0 test with the complete training contract**

在 `test_2d_pretraining_config_contract()` 末尾加入：

```python
    assert_batch_protocol(cfg, 4, 4, 16)
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.02)
    assert cfg.auto_scale_lr.enable is False
    assert cfg.default_hooks.checkpoint.save_best == 'coco/bbox_mAP'
```

- [ ] **Step 5: Run the focused test and verify RED**

Run in the configured TransFusion server environment:

```bash
python -m pytest projects/TransFusionKITTI/tests/test_configs.py -q
```

Expected: FAIL. Stage 0/1 have no `accumulative_counts`, Stage 2 has no
`accumulative_counts` and still reports `auto_scale_lr.enable=True`. The failure
must come from the new protocol assertions, not from an import or config parse error.

### Task 2: Implement the Three-Stage Accumulation Configuration

**Files:**
- Modify: `projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py`
- Modify: `projects/TransFusionKITTI/configs/transfusion_l_kitti.py`
- Modify: `projects/TransFusionKITTI/configs/transfusion_lc_kitti.py`
- Test: `projects/TransFusionKITTI/tests/test_configs.py`

- [ ] **Step 1: Add Stage 0 accumulation without replacing the inherited optimizer**

在 `r50_fpn_kitti_2d.py` 的 `model` 配置后加入：

```python
optim_wrapper = dict(accumulative_counts=4)
```

不要添加 `_delete_=True`。这样会保留 MMDetection Faster R-CNN 基础配置中的
SGD、`lr=0.02`、momentum、weight decay 和 gradient clipping 行为。

- [ ] **Step 2: Add Stage 1 accumulation without replacing the cyclic optimizer**

在 `transfusion_l_kitti.py` 的 dataloader 配置后加入：

```python
optim_wrapper = dict(accumulative_counts=8)
```

不要添加 `_delete_=True`。这样会保留 `cyclic-40e.py` 中 AdamW
`lr=0.0018`、betas、weight decay、gradient clipping 和四段 scheduler。

- [ ] **Step 3: Add Stage 2 accumulation and disable automatic LR scaling**

把 `transfusion_lc_kitti.py` 中的优化器和自动缩放配置改为：

```python
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=8,
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=16)
```

保留 `lr = 1e-4`、OneCycle `eta_max=lr * 10`、6 epochs 和冻结策略不变。

- [ ] **Step 4: Run the focused test and verify GREEN**

```bash
python -m pytest projects/TransFusionKITTI/tests/test_configs.py -q
```

Expected: `3 passed`。三个配置分别解析为 `(micro-batch, accumulation)` =
`(4, 4)`、`(6, 8)`、`(2, 8)`，且 Stage 2 自动 LR 缩放关闭。

- [ ] **Step 5: Commit the tested configuration contract**

```bash
git add \
  projects/TransFusionKITTI/tests/test_configs.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py
git commit -m "feat: configure single-H800 gradient accumulation"
```

### Task 3: Document the Reproducible Server Workflow

**Files:**
- Modify: `projects/TransFusionKITTI/README.md`

- [ ] **Step 1: Add the formal-training protocol before `## H800 smoke test`**

加入以下完整章节：

````markdown
## Single-H800 formal training protocol

正式实验使用 FP32 和 MMEngine 梯度累积。三个阶段的
`(micro-batch, accumulative_counts, nominal batch)` 分别为
`(4, 4, 16)`、`(6, 8, 48)`、`(2, 8, 16)`。这里的“名义等效 batch”只表示
一次 optimizer update 汇总的样本数；BatchNorm 仍按 micro-batch 统计，
Stage 1 在 epoch 尾部还有约 0.2% 的更新边界差异，不能描述为与八卡 DDP
逐指令等价。

先打印并保存三份展开配置，确认训练协议已经生效：

```bash
mkdir -p work_dirs/transfusion_kitti_formal_config_dumps
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  > work_dirs/transfusion_kitti_formal_config_dumps/stage0.py
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  > work_dirs/transfusion_kitti_formal_config_dumps/stage1.py
python tools/misc/print_config.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  > work_dirs/transfusion_kitti_formal_config_dumps/stage2.py
rg -n "batch_size|accumulative_counts|auto_scale_lr|eta_max|max_epochs" \
  work_dirs/transfusion_kitti_formal_config_dumps
```

正式 run1 固定使用 GPU 2、`seed=0` 和独立目录。初次启动全程不加
`--amp`、`--auto-scale-lr` 或 `--resume`。

Stage 0：

```bash
mkdir -p work_dirs/r50_fpn_kitti_2d_formal_run1
git rev-parse HEAD \
  > work_dirs/r50_fpn_kitti_2d_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_formal_run1 \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False
```

Stage 1：

```bash
mkdir -p work_dirs/transfusion_l_kitti_formal_run1
git rev-parse HEAD \
  > work_dirs/transfusion_l_kitti_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1 \
  --cfg-options randomness.seed=0 randomness.deterministic=False
```

两个阶段都结束后，从真实文件名解析 best checkpoint 并合并：

```bash
IMAGE_BEST=$(find work_dirs/r50_fpn_kitti_2d_formal_run1 \
  -maxdepth 1 -type f -name 'best*.pth' -print -quit)
LIDAR_BEST=$(find work_dirs/transfusion_l_kitti_formal_run1 \
  -maxdepth 1 -type f -name 'best*.pth' -print -quit)
test -n "$IMAGE_BEST" && test -f "$IMAGE_BEST"
test -n "$LIDAR_BEST" && test -f "$LIDAR_BEST"
python projects/TransFusionKITTI/tools/merge_pretrained_weights.py \
  --lidar "$LIDAR_BEST" \
  --image "$IMAGE_BEST" \
  --output checkpoints/transfusion_kitti_stage2_formal_run1_init.pth
sha256sum "$IMAGE_BEST" "$LIDAR_BEST" \
  checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
  > work_dirs/transfusion_kitti_formal_run1_sha256.txt
```

合并后检查来源哈希和 spconv2 leaf metadata：

```bash
python - <<'PY'
import torch

path = 'checkpoints/transfusion_kitti_stage2_formal_run1_init.pth'
checkpoint = torch.load(path, map_location='cpu')
metadata = getattr(checkpoint['state_dict'], '_metadata', {})
leaf = metadata.get('middle_encoder.conv_input.0')
sources = checkpoint.get('meta', {}).get('transfusion_kitti_sources')
print('middle_encoder.conv_input.0:', leaf)
print('sources:', sources)
assert leaf == {'version': 2}
assert sources is not None
assert sources['lidar']['sha256']
assert sources['image']['sha256']
PY
```

Stage 2：

```bash
mkdir -p work_dirs/transfusion_lc_kitti_formal_run1
git rev-parse HEAD \
  > work_dirs/transfusion_lc_kitti_formal_run1/source_commit.txt
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1 \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
    randomness.seed=0 randomness.deterministic=False
```

训练意外中断时，只在原配置和原 work dir 中恢复。恢复命令仍保留首次训练的
`load_from` 与 seed 覆盖，MMEngine 会优先恢复该 work dir 的 latest
checkpoint：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_formal_run1 \
  --resume \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False

CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1 \
  --resume \
  --cfg-options randomness.seed=0 randomness.deterministic=False

CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1 \
  --resume \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
    randomness.seed=0 randomness.deterministic=False
```

不得从任何 `*_smoke/epoch_1.pth` 恢复正式实验。run2/run3 只在 run1 的
loss、AP40、checkpoint 和日志审查通过后启动；目录分别使用 `formal_run2`、
`formal_run3`，seed 分别固定为 1、2，其他训练口径不变。

## Formal run1 evaluation

Stage 1 和 Stage 2 都使用各自真实的 best checkpoint 完整评估：

```bash
LIDAR_BEST=$(find work_dirs/transfusion_l_kitti_formal_run1 \
  -maxdepth 1 -type f -name 'best*.pth' -print -quit)
LC_BEST=$(find work_dirs/transfusion_lc_kitti_formal_run1 \
  -maxdepth 1 -type f -name 'best*.pth' -print -quit)
test -n "$LIDAR_BEST" && test -f "$LIDAR_BEST"
test -n "$LC_BEST" && test -f "$LC_BEST"

CUDA_VISIBLE_DEVICES=2 python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  "$LIDAR_BEST" \
  --work-dir work_dirs/transfusion_l_kitti_formal_run1_eval

CUDA_VISIBLE_DEVICES=2 python tools/test.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  "$LC_BEST" \
  --work-dir work_dirs/transfusion_lc_kitti_formal_run1_eval
```

完整记录 Pedestrian、Cyclist、Car 的 3D AP40 Easy/Moderate/Hard，以及
`Overall_3D_AP40_moderate`。Stage 2 只和同一 seed、同一数据划分的 Stage 1
比较；smoke AP 不进入论文表格。
````

- [ ] **Step 2: Replace the old formal-training boundary statement**

将 README 当前状态末句保持为以下准确口径：

```markdown
因此，现代框架迁移和 KITTI 多模态数据流已经跑通；单 H800 正式训练协议
已经固化，但正式全量训练精度仍待验证，不能把 smoke 结果写成“多模态方法
已验证有效”。
```

- [ ] **Step 3: Check the documented command paths**

```bash
rg -n \
  "formal_run1|accumulative_counts|randomness.seed=0|--resume|stage2_formal_run1_init" \
  projects/TransFusionKITTI/README.md
```

Expected: README 同时包含三个正式 work dir、合并 checkpoint、seed 0、
梯度累积说明和三条恢复命令；正式训练命令中没有 `--amp`。

- [ ] **Step 4: Commit the server runbook**

```bash
git add projects/TransFusionKITTI/README.md
git commit -m "docs: add single-H800 formal training runbook"
```

### Task 4: Run Local Regression and Static Verification

**Files:**
- Verify: `projects/TransFusionKITTI/tests/test_configs.py`
- Verify: `projects/TransFusionKITTI/tests/`
- Verify: `projects/TransFusionKITTI/`

- [ ] **Step 1: Run the focused config contract test**

```bash
python -m pytest projects/TransFusionKITTI/tests/test_configs.py -q
```

Expected: `3 passed`。

- [ ] **Step 2: Run the complete project test suite**

```bash
python -m pytest projects/TransFusionKITTI/tests -q
```

Expected: all available project tests PASS; no new failure is allowed. A dependency-based
skip is acceptable only when the skipped dependency is absent from the local machine, not
on the configured H800 server environment.

- [ ] **Step 3: Compile all project Python files**

```bash
python -m compileall -q projects/TransFusionKITTI
```

Expected: exit code 0 with no syntax error.

- [ ] **Step 4: Check whitespace and inspect the exact diff**

```bash
git diff --check
git diff --stat HEAD~2..HEAD
git status --short
```

Expected: `git diff --check` has no output. Only the three configs, config test, README
and this implementation plan are part of this protocol change.

### Task 5: Run Three-Stage Accumulation Smoke on H800

**Files:**
- Verify: `projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py`
- Verify: `projects/TransFusionKITTI/configs/transfusion_l_kitti.py`
- Verify: `projects/TransFusionKITTI/configs/transfusion_lc_kitti.py`
- Artifact: `work_dirs/r50_fpn_kitti_2d_accum_smoke/`
- Artifact: `work_dirs/transfusion_l_kitti_accum_smoke/`
- Artifact: `work_dirs/transfusion_lc_kitti_accum_smoke/`

- [ ] **Step 1: Confirm the exact source revision and clean smoke directories**

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
test ! -e work_dirs/r50_fpn_kitti_2d_accum_smoke
test ! -e work_dirs/transfusion_l_kitti_accum_smoke
test ! -e work_dirs/transfusion_lc_kitti_accum_smoke
```

Expected: branch is `codex/transfusion-kitti-port`, the revision contains Tasks 1-3,
the worktree is clean, and all three `test ! -e` commands succeed. Do not delete or
reuse an existing directory to force this check to pass; choose a new suffixed directory
when a prior smoke artifact must be retained.

- [ ] **Step 2: Run Stage 0 for one effective optimizer update**

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/r50_fpn_kitti_2d.py \
  --work-dir work_dirs/r50_fpn_kitti_2d_accum_smoke \
  --cfg-options \
    load_from=checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.indices=16
```

Expected: four micro-batch iterations complete, one accumulated optimizer update is
performed, loss and grad norm are finite, and `epoch_1.pth` is saved.

- [ ] **Step 3: Run Stage 1 for one effective optimizer update**

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_l_kitti.py \
  --work-dir work_dirs/transfusion_l_kitti_accum_smoke \
  --cfg-options \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.dataset.indices=24
```

Expected: `RepeatDataset(times=2)` exposes 48 samples, eight micro-batch iterations
complete, one accumulated optimizer update is performed, all numeric losses and grad norm
are finite, and `epoch_1.pth` is saved.

- [ ] **Step 4: Run Stage 2 for one effective optimizer update**

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
  projects/TransFusionKITTI/configs/transfusion_lc_kitti.py \
  --work-dir work_dirs/transfusion_lc_kitti_accum_smoke \
  --cfg-options \
    load_from=checkpoints/transfusion_kitti_stage2_init.pth \
    randomness.seed=0 randomness.deterministic=False \
    train_cfg.max_epochs=1 \
    val_cfg=None val_dataloader=None val_evaluator=None \
    train_dataloader.dataset.dataset.indices=8
```

Expected: `RepeatDataset(times=2)` exposes 16 samples, eight micro-batch iterations
complete, one accumulated optimizer update is performed, `epoch_1.pth` is saved, and the
log has neither automatic LR scaling nor `middle_encoder`/backbone shape mismatch.

- [ ] **Step 5: Verify all smoke artifacts and logs**

```bash
test -f work_dirs/r50_fpn_kitti_2d_accum_smoke/epoch_1.pth
test -f work_dirs/transfusion_l_kitti_accum_smoke/epoch_1.pth
test -f work_dirs/transfusion_lc_kitti_accum_smoke/epoch_1.pth

! rg -n -i \
  -e 'loss[^:]*: *(nan|inf|-inf)' \
  -e 'grad_norm: *(nan|inf|-inf)' \
  work_dirs/r50_fpn_kitti_2d_accum_smoke \
  work_dirs/transfusion_l_kitti_accum_smoke \
  work_dirs/transfusion_lc_kitti_accum_smoke \
  -g '*.log'

! rg -n \
  -e 'LR is set based on batch size' \
  -e 'Scaling the original LR' \
  work_dirs/transfusion_lc_kitti_accum_smoke -g '*.log'

! rg -n \
  -e 'size mismatch.*middle_encoder' \
  -e 'middle_encoder.*size mismatch' \
  -e 'size mismatch.*backbone' \
  -e 'backbone.*size mismatch' \
  work_dirs/transfusion_lc_kitti_accum_smoke -g '*.log'
```

Expected: all checkpoint checks pass and all three negated `rg` commands return no
matching log line. Only after this evidence exists may the gradient-accumulation smoke be
recorded as verified.

### Task 6: Execute Formal Run1 in the Approved Serial Order

**Files:**
- Artifact: `work_dirs/r50_fpn_kitti_2d_formal_run1/`
- Artifact: `work_dirs/transfusion_l_kitti_formal_run1/`
- Artifact: `checkpoints/transfusion_kitti_stage2_formal_run1_init.pth`
- Artifact: `work_dirs/transfusion_lc_kitti_formal_run1/`

- [ ] **Step 1: Train Stage 0 for 12 epochs**

Run the Stage 0 command from README without `--resume`. Expected: 12 epochs complete,
validation runs at each configured interval, `coco/bbox_mAP` is finite, and exactly one
or more `best*.pth` candidate exists in the formal Stage 0 directory.

- [ ] **Step 2: Train Stage 1 for 40 epochs**

Run the Stage 1 command from README after Stage 0 releases GPU 2. Expected: 40 epochs
complete in FP32, losses remain finite, complete KITTI validation runs, and a best
checkpoint is selected by
`Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate`.

- [ ] **Step 3: Merge only the formal Stage 0/1 best checkpoints**

Run README's `IMAGE_BEST`/`LIDAR_BEST`, merge, SHA-256 and metadata commands. Expected:
the output is `checkpoints/transfusion_kitti_stage2_formal_run1_init.pth`, both source
hashes are present, and `middle_encoder.conv_input.0` is exactly `{'version': 2}`.

- [ ] **Step 4: Train Stage 2 for 6 epochs**

Run the Stage 2 command from README without `--resume`. Expected: the initialization log
has no `middle_encoder` or backbone size mismatch, only the intended fusion modules are
trainable, no automatic LR scaling message appears, six epochs complete in FP32, and a
best checkpoint is selected by the same strict overall AP40 Moderate metric as Stage 1.

- [ ] **Step 5: Preserve the reproducibility record**

```bash
git rev-parse HEAD \
  > work_dirs/transfusion_kitti_formal_run1_commit.txt
find work_dirs/r50_fpn_kitti_2d_formal_run1 \
     work_dirs/transfusion_l_kitti_formal_run1 \
     work_dirs/transfusion_lc_kitti_formal_run1 \
  -maxdepth 1 -type f -name 'best*.pth' -print \
  > work_dirs/transfusion_kitti_formal_run1_best_checkpoints.txt
sha256sum $(cat work_dirs/transfusion_kitti_formal_run1_best_checkpoints.txt) \
  checkpoints/transfusion_kitti_stage2_formal_run1_init.pth \
  > work_dirs/transfusion_kitti_formal_run1_artifact_sha256.txt
```

Expected: commit、所有真实 best checkpoint 路径和对应 SHA-256 均落盘。不得把
smoke checkpoint 写入该清单。

### Task 7: Run Complete Stage 1/2 AP40 Evaluation and Gate Later Seeds

**Files:**
- Artifact: `work_dirs/transfusion_l_kitti_formal_run1_eval/`
- Artifact: `work_dirs/transfusion_lc_kitti_formal_run1_eval/`

- [ ] **Step 1: Evaluate the Stage 1 best checkpoint on all 3769 validation samples**

Run README's first formal evaluation command. Expected: progress reaches `3769/3769`
and the output contains KITTI bbox/BEV/3D AP11 and AP40 without NaN, CUDA error or
evaluation interruption.

- [ ] **Step 2: Evaluate the Stage 2 best checkpoint on all 3769 validation samples**

Run README's second formal evaluation command. Expected: progress reaches `3769/3769`
and the same complete metric families are printed without runtime failure.

- [ ] **Step 3: Check the paper-facing metric set**

From the two completed logs, record Pedestrian、Cyclist、Car 的 3D AP40
Easy/Moderate/Hard and `Overall_3D_AP40_moderate`. Compare Stage 2 only with Stage 1
from this run1; do not mix validation/test-server values and do not use smoke AP.

- [ ] **Step 4: Decide whether run2/run3 are authorized by run1 evidence**

Proceed only when all of the following are true: three formal stages reached their
configured epoch count; losses stayed finite; formal checkpoint provenance and metadata
checks passed; Stage 1/2 full evaluations completed; AP trends are numerically plausible.
Then reproduce the same sequence with directories `formal_run2`/`formal_run3` and seeds
1/2. If any condition fails, retain logs and diagnose that failure before starting another
seed; do not hide it by enabling AMP, skipping a batch or changing matching costs.

## Completion Gate

This implementation is complete only after:

1. Config tests prove the 4/4、6/8、2/8 accumulation protocol and unchanged training
   hyperparameters.
2. The complete project test suite and compile check pass.
3. All three H800 accumulation smoke runs save finite FP32 checkpoints.
4. Formal run1 completes Stage 0、Stage 1、merge、Stage 2 in order.
5. Stage 1/2 best checkpoints both complete 3769-sample KITTI AP40 evaluation.
6. Logs, commit, checkpoint paths and SHA-256 records distinguish formal products from
   smoke products.

