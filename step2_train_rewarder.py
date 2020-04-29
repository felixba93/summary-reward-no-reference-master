import torch
from torch.autograd import Variable
import numpy as np
import os
import argparse
import random
import copy
from tqdm import tqdm
import pickle
from scorer.data_helper.json_reader import read_sorted_scores, read_articles, read_processed_scores, read_scores
from scipy.stats import spearmanr, pearsonr, kendalltau
import math
from torchvision import models
from resources import MODEL_WEIGHT_DIR
from resources import OUTPUTS_DIR
from matplotlib import pyplot as plt
import csv


def parse_split_data(sorted_scores, train_percent, dev_percent, prompt='overall'):
    train = {}
    dev = {}
    test = {}
    all = {}
    topic_count = 0
    for article_id, scores_list in tqdm(sorted_scores.items()):

        entry = {}
        summ_ids = [s['summ_id'] for s in scores_list]
        for sid in summ_ids:
            entry['sys_summ' + repr(sid)] = [s['scores'][prompt] for s in scores_list if s['summ_id'] == sid][0]

        rand = random.random()
        all[article_id] = entry
        if rand < train_percent:
            train[article_id] = entry
        elif rand < train_percent + dev_percent:
            dev[article_id] = entry
        else:
            test[article_id] = entry

        topic_count += 1
    print("topics in parse_split_data", topic_count)

    return train, dev, test, all


def build_model(model_type, vec_length, learn_rate=None):
    if 'linear' in model_type:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, 1),
        )
    else:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, int(vec_length / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(vec_length / 2), 1),
        )
    if learn_rate is not None:
        optimiser = torch.optim.Adam(deep_model.parameters(), lr=learn_rate)
        return deep_model, optimiser
    else:
        return deep_model


def deep_pair_train(vec_list, target, deep_model, optimiser, device):
    # print(np.array(vec_list).shape)
    input = Variable(torch.from_numpy(np.array(vec_list)).float())
    # print(input)
    if 'gpu' in device:
        input = input.to('cuda')
    value_variables = deep_model(input)
    # print(value_variables)
    softmax_layer = torch.nn.Softmax(dim=1)
    pred = softmax_layer(value_variables)
    # print(pred)
    # print(np.array(target).shape, np.array(target).reshape(-1, 2, 1).shape)
    target_variables = Variable(torch.from_numpy(np.array(target)).float()).view(-1, 2, 1)
    # print(target_variables)

    if 'gpu' in device:
        target_variables = target_variables.to('cuda')

    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(pred, target_variables)
    # print(loss)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.cpu().item()


def deep_pair_train_loss_only(vec_list, target, deep_model, optimiser, device):
    # print(np.array(vec_list).shape)
    input = Variable(torch.from_numpy(np.array(vec_list)).float())
    # print(input)
    if 'gpu' in device:
        input = input.to('cuda')
    value_variables = deep_model(input)
    # print(value_variables)
    softmax_layer = torch.nn.Softmax(dim=1)
    pred = softmax_layer(value_variables)
    # print(pred)
    # print(np.array(target).shape, np.array(target).reshape(-1, 2, 1).shape)
    target_variables = Variable(torch.from_numpy(np.array(target)).float()).view(-1, 2, 1)
    # print(target_variables)

    if 'gpu' in device:
        target_variables = target_variables.to('cuda')

    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(pred, target_variables)
    # print(loss)

    return loss.cpu().item()


def build_pairs(entries):
    pair_list = []
    topic_count = 0
    summ_count = 0
    for article_id in entries:
        entry = entries[article_id]
        summ_ids = list(entry.keys())
        for i in range(len(summ_ids) - 1):
            for j in range(1, len(summ_ids)):
                if entry[summ_ids[i]] > entry[summ_ids[j]]:
                    pref = [1, 0]
                elif entry[summ_ids[i]] < entry[summ_ids[j]]:
                    pref = [0, 1]
                else:
                    pref = [0.5, 0.5]
                pair_list.append((article_id, summ_ids[i], summ_ids[j], pref))
        topic_count += 1
        summ_count = summ_count + len(summ_ids)
    print("topics", topic_count)
    print("summ", summ_count)
    return pair_list


def build_pair_vecs(vecs, pairs):
    pair_vec_list = []
    for aid, sid1, sid2, _ in pairs:
        article_vec = list(vecs[aid]['article'])
        s1_vec = list(vecs[aid][sid1])
        s2_vec = list(vecs[aid][sid2])
        pair_vec_list.append([article_vec + s1_vec, article_vec + s2_vec])
    return pair_vec_list


def pair_train_rewarder(vec_dic, pairs, deep_model, optimiser, loss_only, batch_size=32, device='cpu'):
    loss_list = []
    shuffled_pairs = pairs[:]
    np.random.shuffle(shuffled_pairs)
    vec_pairs = build_pair_vecs(vec_dic, shuffled_pairs)
    # print('total number of pairs built: {}'.format(len(vec_pairs)))

    for pointer in range(int(len(pairs) / batch_size) + 1):
        vec_batch = vec_pairs[pointer * batch_size:(pointer + 1) * batch_size]
        target_batch = shuffled_pairs[pointer * batch_size:(pointer + 1) * batch_size]
        target_batch = [ee[-1] for ee in target_batch]
        if loss_only:
            loss = deep_pair_train_loss_only(vec_batch, target_batch, deep_model, optimiser, device)
        else:
            loss = deep_pair_train(vec_batch, target_batch, deep_model, optimiser, device)
        loss_list.append(loss)

    return np.mean(loss_list)


def test_rewarder(vec_list, human_scores, model, device, plot_file=None):
    results = {'rho': [], 'pcc': [], 'tau': [], 'rho_global': [], 'pcc_global': [], 'tau_global': []}
    true_scores_all = []
    pred_scores_all = np.array([])
    # pred_scores_all = []
    for article_id in human_scores:
        entry = human_scores[article_id]
        summ_ids = list(entry.keys())
        if len(summ_ids) < 2: continue
        concat_vecs = []
        true_scores = []
        for i in range(len(summ_ids)):
            article_vec = list(vec_list[article_id]['article'])
            summ_vec = list(vec_list[article_id][summ_ids[i]])
            # print(np.array(concat_vecs).shape, np.array(article_vec).shape, np.array(summ_vec).shape)
            concat_vecs.append(article_vec + summ_vec)
            # print(np.array(concat_vecs).shape)
            true_scores.append(entry[summ_ids[i]])
        true_scores_all += true_scores  # add scores for topic to list of all scores
        input = Variable(torch.from_numpy(np.array(concat_vecs)).float())
        if 'gpu' in device:
            input = input.to('cuda')
        model.eval()
        with torch.no_grad():
            # print(true_scores)
            # print(np.array(true_scores).shape)
            # print(input)
            # print(input.shape)
            # print(model(input).data.cpu().numpy())
            # print(model(input).data.cpu().numpy().shape)
            pred_scores = model(input).data.cpu().numpy().reshape(1, -1)[0]
            pred_scores_all = np.concatenate((pred_scores_all, pred_scores), axis=0)
            # pred_scores_all += pred_scores.tolist()

        rho = spearmanr(true_scores, pred_scores)[0]
        pcc = pearsonr(true_scores, pred_scores)[0]
        tau = kendalltau(true_scores, pred_scores)[0]
        if not (math.isnan(rho) or math.isnan(pcc) or math.isnan(tau)):
            results['rho'].append(rho)
            results['pcc'].append(pcc)
            results['tau'].append(tau)
    rho = spearmanr(true_scores_all, pred_scores_all)[0]
    pcc = pearsonr(true_scores_all, pred_scores_all)[0]
    tau = kendalltau(true_scores_all, pred_scores_all)[0]
    if not (math.isnan(rho) or math.isnan(pcc) or math.isnan(tau)):
        results['rho_global'].append(rho)
        results['pcc_global'].append(pcc)
        results['tau_global'].append(tau)

    if plot_file is not None:
        fig, ax = plt.subplots()

        #true_scores_all=np.array(true_scores_all)
        #pred_scores_all=np.array(pred_scores_all)

        unique = np.sort(np.unique(true_scores_all))
        data_to_plot = [pred_scores_all[true_score == true_scores_all] for true_score in unique]

        # bw_methods determines how soft the distribution curve will be. lower values are more sharp
        ax.violinplot(data_to_plot, showmeans=True, showmedians=True,bw_method=0.2)
        ax.scatter(true_scores_all + np.random.normal(0, 0.1, pred_scores_all.shape[0]), pred_scores_all, marker=".", s=3, alpha=0.5)
        ax.set_title('Comparison and distributions of true values to predicted score')
        ax.set_xlabel('true scores')
        ax.set_ylabel('predicted scores')

        xticklabels = true_scores_all
        ax.set_xticks(true_scores_all)
        print("violin plot written to: %s"%plot_file)
        plt.savefig(plot_file)

    return results


def parse_args():
    ap = argparse.ArgumentParser("arguments for summary sampler")
    ap.add_argument('-e', '--epoch_num', type=int, default=50)
    ap.add_argument('-b', '--batch_size', type=int, default=32)
    ap.add_argument('-tt', '--train_type', type=str, help='pairwise or regression', default='pairwise')
    ap.add_argument('-tp', '--train_percent', type=float, help='how many data used for training', default=.64)
    ap.add_argument('-dp', '--dev_percent', type=float, help='how many data used for dev', default=.16)
    ap.add_argument('-lr', '--learn_rate', type=float, help='learning rate', default=3e-4)
    ap.add_argument('-mt', '--model_type', type=str, help='deep/linear', default='linear')
    ap.add_argument('-dv', '--device', type=str, help='cpu/gpu', default='gpu')
    ap.add_argument('-se', '--seed', type=int, help='random seed number', default='1')

    args = ap.parse_args()
    return args.epoch_num, args.batch_size, args.train_type, args.train_percent, args.dev_percent, args.learn_rate, args.model_type, args.device, args.seed


if __name__ == '__main__':
    epoch_num, batch_size, train_type, train_percent, dev_percent, learn_rate, model_type, device, seed = parse_args()

    print('\n=====Arguments====')
    print('epoch num {}'.format(epoch_num))
    print('batch size {}'.format(batch_size))
    print('train type {}'.format(train_type))
    print('train percent {}'.format(train_percent))
    print('dev percent {}'.format(dev_percent))
    print('learn rate {}'.format(learn_rate))
    print('model type {}'.format(model_type))
    print('device {}'.format(device))
    print('seed {}'.format(seed))
    print('=====Arguments====\n')

    with open('BetterRewardsStatistics_v4.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)

        np.random.seed(seed=seed)
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        if train_percent + dev_percent >= 1.:
            print('ERROR! Train data percentage plus dev data percentage is {}! Make sure the sum is below 1.0!'.format(
                train_percent + dev_percent))
            exit(1)

        BERT_VEC_LENGTH = 1024  # change this to 768 if you use bert-base
        deep_model, optimiser = build_model(model_type, BERT_VEC_LENGTH * 2, learn_rate)
        if 'gpu' in device:
            deep_model.to('cuda')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # read human scores and vectors for summaries/docs, and split the train/dev/test set
        sorted_scores = read_sorted_scores()
        train, dev, test, all = parse_split_data(sorted_scores, train_percent, dev_percent)

        train_pairs = build_pairs(train)
        dev_pairs = build_pairs(dev)
        test_pairs = build_pairs(test)
        print(len(train_pairs), len(dev_pairs), len(test_pairs))

        # read bert vectors
        with open('data/doc_summ_bert_vectors.pkl', 'rb') as ff:
            all_vec_dic = pickle.load(ff)

        pcc_list = []
        weights_list = []
        for ii in range(epoch_num + 1):
            print('\n=====EPOCH {}====='.format(ii))
            if epoch_num == 0:

                # do not train in epoch 0, just evaluate the performance of the randomly initialized model (sanity check and baseline)
                loss_train = pair_train_rewarder(all_vec_dic, train_pairs, deep_model, optimiser, True, batch_size,
                                                 device)
            else:
                # from epoch 1 on, receive the data and learn from it. the loss is still the loss before fed with the training examples
                loss_train = pair_train_rewarder(all_vec_dic, train_pairs, deep_model, optimiser, False, batch_size,
                                                 device)

            loss_dev = pair_train_rewarder(all_vec_dic, dev_pairs, deep_model, optimiser, True, batch_size, device)

            loss_test = pair_train_rewarder(all_vec_dic, test_pairs, deep_model, optimiser, True, batch_size, device)

            csv_row = [seed, learn_rate, model_type, len(train_pairs), len(dev_pairs), len(test_pairs), ii, loss_train,
                       loss_dev, loss_test]
            print('--> loss', loss_train)

            results = test_rewarder(all_vec_dic, dev, deep_model, device)
            for metric in results:
                print('{}\t{}'.format(metric, np.mean(results[metric])))
                csv_row.append(np.mean(results[metric]))

            # Test-Data only
            results_test = test_rewarder(all_vec_dic, test, deep_model, device)
            for metric in results_test:
                print('{}\t{}'.format(metric, np.mean(results_test[metric])))
                csv_row.append(np.mean(results_test[metric]))

            # Train-Data only
            results_train = test_rewarder(all_vec_dic, train, deep_model, device)
            for metric in results_train:
                print('{}\t{}'.format(metric, np.mean(results_train[metric])))
                csv_row.append(np.mean(results_train[metric]))

            writer.writerow(csv_row)
            pcc_list.append(np.mean(results['pcc']))
            weights_list.append(copy.deepcopy(deep_model.state_dict()))

        idx = np.argmax(pcc_list)
        best_result = pcc_list[idx]
        print('\n======Best results come from epoch no. {}====='.format(idx))

        deep_model.load_state_dict(weights_list[idx])
        output_pattern='batch{}_{}_trainPercent{}_seed{}_lrate{}_{}_epoch{}'.format(
            batch_size, train_type, train_percent, seed, learn_rate, model_type,epoch_num
        )
        test_results = test_rewarder(all_vec_dic, test, deep_model, device, os.path.join(OUTPUTS_DIR,output_pattern+'_onTest.pdf'))
        test_rewarder(all_vec_dic, train, deep_model, device, os.path.join(OUTPUTS_DIR,output_pattern+'_onTrain.pdf'))
        test_rewarder(all_vec_dic, dev, deep_model, device, os.path.join(OUTPUTS_DIR,output_pattern+'_onDev.pdf'))
        print('Its performance on the test set is:')
        for metric in test_results:
            print('{}\t{}'.format(metric, np.mean(test_results[metric])))
        model_weight_name = 'pcc{0:.4f}_'.format(np.mean(test_results['pcc']))
        model_weight_name += 'seed{}_epoch{}_batch{}_{}_trainPercent{}_lrate{}_{}.model'.format(
            seed, epoch_num, batch_size, train_type, train_percent, learn_rate, model_type
        )

        torch.save(weights_list[idx], os.path.join(MODEL_WEIGHT_DIR, model_weight_name))
        print('\nbest model weight saved to: {}'.format(os.path.join(MODEL_WEIGHT_DIR, model_weight_name)))
