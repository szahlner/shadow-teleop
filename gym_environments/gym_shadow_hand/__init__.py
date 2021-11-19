from gym.envs.registration import register

register(
    id="shadow_hand_reach-v0",
    entry_point="gym_shadow_hand.envs:ShadowHandReachEnvV0"
)

register(
    id="shadow_hand_reach-v1",
    entry_point="gym_shadow_hand.envs:ShadowHandReachEnv"
)

register(
    id="shadow_hand_reach_goalenv-v1",
    entry_point="gym_shadow_hand.envs:ShadowHandReachGoalEnv"
)

register(
    id="shadow_hand_block-v1",
    entry_point="gym_shadow_hand.envs:ShadowHandManipulateBlockEnv"
)

register(
    id="shadow_hand_block_goalenv-v1",
    entry_point="gym_shadow_hand.envs:ShadowHandManipulateBlockGoalEnv"
)
