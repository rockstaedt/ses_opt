from seqsuc import SequentialSampling, Parameter, LShapeMethod

# Create all relevant parameters
params = Parameter()

# Run an exemplary l shape method
lsh = LShapeMethod(
    params=params,
    sample_size=100,
    seed=12,
    sampling_method='MC',
    multiprocessing=True,
    progress_info=True
)
lsh.solve_model()

# Run exemplary sequential sampling method
seq = SequentialSampling(
    params=params,
    sampling_method='AV',
    estimator_method='A2RP',
    multiprocessing=True,
    output=True
)
seq.run_seq_sampling()
