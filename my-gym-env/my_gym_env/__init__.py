from gym.envs.registration import register

register(
    id='IndependentTwoWheeledRobot-v0',
    entry_point='my_gym_env.envs:IndependentTwoWheeledRobot'
)

register(
    id='DoubleSlit1D-v0',
    entry_point='my_gym_env.envs:DoubleSlit1D'
)
