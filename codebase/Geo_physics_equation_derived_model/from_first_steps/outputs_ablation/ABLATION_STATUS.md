# Ablation status (2026-07-31 10:22:45)

| Dataset | Variant | Best val F1 | Status |
|---------|---------|-------------|--------|
| bijie | full | 0.8044 | done |
| bijie | no_cmb | 0.7953 | done |
| bijie | no_fs_gate | 0.8964 | done |
| bijie | no_mao | 0.8697 | done |
| bijie | no_mpef | 0.8118 | done |
| bijie | no_prithvi | 0.7281 | done |
| bijie | no_tteb | 0.8894 | done |
| bijie | path_a_only | 0.7528 | done |
| l4s | full | 0.6930 | done |
| l4s | no_cmb | 0.7146 | done |
| l4s | no_fs_gate | 0.7027 | done |
| l4s | no_mao | 0.7174 | done |
| l4s | no_mpef | 0.6885 | done |
| l4s | no_prithvi | 0.5304 | done |
| l4s | no_tteb | 0.7062 | done |
| l4s | path_a_only | 0.7042 | done |

## Complete (2026-07-31 10:23:03)

Both Bijie and L4S 8-variant matrices finished (compact B0, 40 epochs, GPU1).

### Quick ranking by best val F1

**Bijie:** no_fs_gate 0.896 > no_tteb 0.889 > no_mao 0.870 > no_mpef 0.812 > full 0.804 > no_cmb 0.795 > path_a_only 0.753 > no_prithvi 0.728

**L4S:** no_mao 0.717 > no_cmb 0.715 > no_tteb 0.706 > path_a_only 0.704 > no_fs_gate 0.703 > full 0.693 > no_mpef 0.689 > no_prithvi 0.530

