# Running the 5-year climate rollout

```bash
git clone git@github.com:rcc-uchicago/pedramh-profiling.git
cd pedramh-profiling
git checkout feat/makani-b-continuation-dryair
cd makani_sfno
```

```bash
ARMS="B22 B24" \
RUNDIR_B22=nf4_prod_b16_r1 \
CKPT_B22=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/nf4_prod_b16_r1/training_checkpoints/ckpt_mp0_e22_stable.tar \
RUNDIR_B24=nf4_prod_b16_r1 \
CKPT_B24=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/nf4_prod_b16_r1/training_checkpoints/ckpt_mp0_e24_stable.tar \
qsub polaris/polaris_climate_run.pbs
```

Output: `/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/climate_protocol_<jobid>/`
