# ppo_test
Custom PPO algorithm trained for Cartpole Swing Up and Reaching Point Task in the MuJoCo environment. In each episode, the mass of the pendulum and the location of the target are randomly sampled.

## Trained Demo
https://github.com/user-attachments/assets/92ed3524-d7fe-4790-9384-34fc8a3be2b8

## Starting
```bash
git clone https://github.com/maribellae/ppo_test.git
cd ppo_test
```

## Change arguments in args.py file
```python
experiment_num = 0  # change it every new experiment
env_name = 'YOUR-ENV-NAME'
checkpoint_path = "YOUR-CHECKPOINT-NAME-FOR-LOADING" # <- can be changed to the path to trained_ppo.pth
directory = "YOUR-DIRECTORY"
```

## For Training
```bash
python trainer.py
```

## For Testing (checkpoint filename in args.py)
```bash
python tester.py
```
