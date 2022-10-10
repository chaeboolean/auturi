from auturi.vector.shm_backend import SHMParallelEnv
import gym




if __name__ == "__main__":
    
    num_envs = 2
    SEED = 64
    # env_fn = lambda: DumbEnv(sleep=0, init_=21)

    env_fn = lambda: gym.make("HalfCheetah-v3")
    
    
    shm_env = SHMParallelEnv([env_fn for _ in range(num_envs)])

    # shm_env.seed(seed_dict={i: SEED + i for i in range(num_envs)})
    shm_env.seed(SEED)
    shm_env.reset()
    shm_env.start_loop()

    actions = np.stack([dummy_env.action_space.sample() for _ in range(num_envs)])

    TEST_ENV = subproc_env

    for TEST_ENV in [ray_env, subproc_env, dummy_env, shm_env]:
        TEST_ENV.reset()

        if isinstance(TEST_ENV, SHMParallelEnv):
            TEST_ENV.start_loop()

        for idx in range(10):
            TEST_ENV.step(actions)

        print(type(TEST_ENV))
        for idx in range(10):
            with timeoutcontext(None):
                TEST_ENV.step(actions)
