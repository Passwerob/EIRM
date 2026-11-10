# Bundled checkpoints

These files are small enough to store in git. Third-party source checkouts are
not included.

## Practical-RIFE

`practical_rife/train_log/flownet.pkl` is the released Practical-RIFE checkpoint
from <https://github.com/hzwer/Practical-RIFE> (MIT). Copy it into a local
checkout before use:

```bash
cp weights/practical_rife/train_log/flownet.pkl Practical-RIFE/train_log/flownet.pkl
```

## E2VID

`e2vid/E2VID_lightweight.pth.tar` is the pretrained lightweight checkpoint
released with <https://github.com/uzh-rpg/rpg_e2vid>. That project is GPL-3.
This repository redistributes the checkpoint only, not the E2VID source tree.
Pass it with `--weights weights/e2vid/E2VID_lightweight.pth.tar`.

## Paper controllers

The project-trained policy checkpoints stay in
`KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/`. See that directory's README. The Table 1
controller is `eveyes_table1_m0/policy.pt`.
