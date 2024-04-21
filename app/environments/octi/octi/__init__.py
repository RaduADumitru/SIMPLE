from gym.envs.registration import register

register(
    id='Octi-v0',
    entry_point='octi.envs:OctiEnv',
)


