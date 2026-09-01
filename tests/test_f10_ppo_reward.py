from duckie_pomdp.control.ppo_reward import PPORewardTerms


def test_reward_components_sum_exactly():
    terms = PPORewardTerms(1.0, -0.2, -0.3, 0.4, -0.1, 2.0)
    assert terms.total == sum((1.0, -0.2, -0.3, 0.4, -0.1, 2.0))


def test_irrelevant_reward_components_can_be_exact_zero():
    terms = PPORewardTerms(0.1, -0.01, 0.0, 0.0, -0.002, 0.0)
    assert terms.pedestrian == 0.0
    assert terms.stop == 0.0

