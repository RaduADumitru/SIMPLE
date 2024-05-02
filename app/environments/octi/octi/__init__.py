from gym.envs.registration import register
import gym.envs.registration

env_name = 'Octi-v0'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

register(
    id='Octi-v0',
    entry_point='octi.envs:OctiEnv',
)


