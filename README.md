# ppo_test
Custom PPO algorithm trained for Cartpole Swing Up and Reaching Point Task in the MuJoCo environment. In each episode, the mass of the pendulum and the location of the target are randomly sampled.

## Trained Demo
[Trained Demo](https://github.com/maribellae/ppo_test/blob/main/demo.mp4)

## Starting
```bash
git clone https://github.com/maribellae/ppo_test.git
cd ppo_test
```
Change arguments in args.py file

## For Training
```bash
python trainer.py
```

## For Testing (checkpoint filename in args.py)
```bash
python tester.py
```
