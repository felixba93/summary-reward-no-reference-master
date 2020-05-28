import step2_train_rewarder
import itertools
from datetime import datetime

if __name__ == '__main__':
    epoch_num = 500
    batch_size = 32
    # train_type = 'pairwise' #train type is pairwise or regression, but actually has no effect. it is always pairwise
    train_percent = 0.64
    dev_percent = 0.16
    learn_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    model_types = ['linear', 'deep']
    #    device = 'gpu'
    device = 'cpu'
    seeds = [1]
    file_name = 'majority_intra-topic_w-ties_epoch_500_seed1_balanced.csv'

    # compute all combinations
    for learn_rate, model_type, seed in itertools.product(learn_rates, model_types, seeds):
        # for skipping one of the combinations, do something like
        if learn_rate == 0.5 and model_type == 'linear':
            continue

        args = ['dummy', '--epoch_num', epoch_num, '--batch_size', batch_size, '--train_percent', train_percent,
                '--dev_percent', dev_percent, '--learn_rate', learn_rate, '--model_type', model_type, '--device',
                device, '--seed', seed, '--file_name', file_name]
        args = [str(arg) for arg in args]  # convert everything to string, the arguments parser needs it
        print(datetime.now().strftime("%Y-%M-%d %H:%M:%S"))
        print('starting experiment:', args)
        step2_train_rewarder.main(args)
# --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%
