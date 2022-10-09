algo = sb3.create_algorithm(configs)
tuner = auturi.Tuner(**uturi_configs)

wrap_sb3(algo, tuner)

algo.learn()
