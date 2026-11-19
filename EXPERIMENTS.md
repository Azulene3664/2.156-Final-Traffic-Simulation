Ring-road experiment (main result of the mid-term report)

  python ring/train.py --steps 3000000 --lr 1e-3 --seed 1 --out models/ring_ppo   # trains the policy (PPO is unstable here: results vary between runs)
  python ring/evaluate.py                                                         # Table 1 and figures -> ring/figures

Open-road experiments (first attempt, described in the report)

  python training/train.py        # PPO v2 on the open road
  python evaluation/evaluate.py   # 30-seed comparison of all open-road controllers

Dependencies: evaluation/requirements.txt
