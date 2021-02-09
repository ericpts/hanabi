import lib_agent


def test_replay_buffer():
    rb = lib_agent.ReplayBuffer(max_size=5)
    for x in range(5):
        rb.push(x)
    rb.push(5)
    assert 0 not in rb
    rb.push(6)
    assert 1 not in rb

    sample_with_all = rb.sample(100_000)
    for x in range(2, 6 + 1):
        assert x in sample_with_all
