from seqsuc import SequentialSampling, Parameter


params = Parameter()

seq = SequentialSampling(params, 'MC', False, 'SRP')

seq.run_seq_sampling()
