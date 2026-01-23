# Question for Cloud Agent: FID Deadlock in Multi-GPU Training

## Quick Summary
Multi-GPU SDXL training hangs at `torchmetrics.FrechetInceptionDistance.compute()` because it requires ALL distributed processes to participate, but only the main process calls it.

## The Deadlock
- **Rank 0**: Calls `fid_metric.compute()` → waits for rank 1 to sync
- **Rank 1**: Already passed `wait_for_everyone()` → not calling `fid_metric.compute()` → deadlock

## Evidence
See debug logs in [`FID_SYNC_ISSUE.md`](./FID_SYNC_ISSUE.md) - rank 1 passes barrier, rank 0 hangs in `.compute()`.

## What We Need
Compute FID **only once** on main process in multi-GPU training without deadlock.

## Proposed Solutions

1. **All processes compute FID** (wastes GPU, but works?)
2. **Temporarily destroy process group** (risky, may break Accelerate)
3. **Disable distributed detection during FID** (monkey-patch?)
4. **Separate subprocess** (overhead, but isolated)

## My Recommendation
I think **Option 1** (all processes compute FID) is safest, even though it wastes computation. We can optimize later.

**Alternative**: Could we set `dist_sync_fn=None` or similar when creating FID metric? Or use `with accelerator.no_sync():`?

## Question
**What's the recommended pattern for computing metrics on only the main process in HuggingFace Accelerate when using torchmetrics with distributed training?**

Please advise on best approach. Files ready for you to review in [`FID_SYNC_ISSUE.md`](./FID_SYNC_ISSUE.md).
