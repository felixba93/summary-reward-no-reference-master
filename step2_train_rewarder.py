import sys
import torch
from torch.autograd import Variable
import numpy as np
import os
from os import path
import argparse
import random
import copy
from tqdm import tqdm
import pickle
from scorer.data_helper.json_reader import read_sorted_scores, read_pair_anno_scores, read_articles, \
    read_processed_scores, read_scores
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
            entry['sys_summ' + repr(sid)] = [s['scores'][prompt] for s in scores_list if s['summ_id'] == sid][
                0]  # that can be done more efficiently, but who cares...

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


def parse_split_data_balanced(sorted_scores, train_percent, dev_percent, prompt='overall'):
    train = {}
    dev = {}
    test = {}
    all = {}
    topic_count = 0

    article_ids = list(sorted_scores.keys())
    random.shuffle(article_ids)
    num_articles = len(article_ids)
    train_ids = article_ids[0:int(train_percent * num_articles)]
    dev_ids = article_ids[int(train_percent * num_articles):int((train_percent + dev_percent) * num_articles)]
    # test_ids=article_ids[int((train_percent+dev_percent)*num_articles):]

    for article_id, scores_list in tqdm(sorted_scores.items()):

        entry = {}
        summ_ids = [s['summ_id'] for s in scores_list]
        for sid in summ_ids:
            entry['sys_summ' + repr(sid)] = [s['scores'][prompt] for s in scores_list if s['summ_id'] == sid][
                0]  # that can be done more efficiently, but who cares...

        #        rand = random.random()
        all[article_id] = entry
        if article_id in train_ids:
            train[article_id] = entry
        elif article_id in dev_ids:
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
        # really iterate over all pairs. there was an error here before since j started from 1, to prevent i,j=0,0. but this also lead to i,j=x,0 never be chosen the situation i=j is solved otherwise
        for i in range(len(summ_ids)):
            for j in range(len(summ_ids)):
                if i == j: continue
                if entry[summ_ids[i]] > entry[summ_ids[j]]:
                    pref = [1, 0]
                elif entry[summ_ids[i]] < entry[summ_ids[j]]:
                    pref = [0, 1]
                else:
                    pref = [0.5, 0.5]
                pair_list.append((article_id, summ_ids[i], summ_ids[j], pref))
                print(pair_list)
        topic_count += 1
        summ_count = summ_count + len(summ_ids)
    print("topics", topic_count)
    print("summ", summ_count)
    return pair_list


# randomize_pref_order and double_prefs are only relevant if the learning function learns f(s0,s1)=pref. in our case, we learn f(s0)=pref[0] and f(s1)=pref[1], so this should be set to False
# noinspection DuplicatedCode
def build_anno_pairs_majority_preferences(entries, sorted_scores, pair_anno_scores, target_type='graded',
                                          ignore_ties=False,
                                          randomize_pref_order=False, double_prefs=False):
    pair_list = []
    topic_count = 0
    anno_count = 0
    summ_count = 0
    entries_text = {}
    # get summary text and matching id
    for article_id, scores_list in tqdm(sorted_scores.items()):
        temp_entry = {}
        summ_ids = [s['summ_id'] for s in scores_list]
        for sid in summ_ids:
            # get summary text
            s_text = [s['sys_summ'] for s in scores_list if s['summ_id'] == sid][0]
            temp_entry['sys_summ' + repr(sid)] = s_text
        # save in dictionary
        entries_text[article_id] = temp_entry

    for article_id in entries:
        entry = entries[article_id]
        summ_ids = list(entry.keys())

        # mapping from summary text to last summary id with that text. that's the one we will use
        summ2id = {entries_text[article_id][summ_id]: summ_id for summ_id in summ_ids}

        # put here the prefs for this article
        article_prefs = {}

        # still run through all pairs
        # really iterate over all pairs. there was an error here before since j started from 1, to prevent i,j=0,0. but this also lead to i,j=x,0 never be chosen the situation i=j is solved otherwise
        for i in range(len(summ_ids)):
            for j in range(len(summ_ids)):

                text_i = entries_text[article_id][summ_ids[i]]
                text_j = entries_text[article_id][summ_ids[j]]
                # check if text is identical, if yes skip
                if i == j or text_i == text_j:
                    # print("DUPLICATE FOUND: TEXT i", text_i, "TEXT j", text_i)
                    continue
                # get the unique summ ids
                unique_summ_id_pair = [summ2id[text_i], summ2id[text_j]]

                # some debug output
                # noinspection PyUnreachableCode
                if False:
                    print("%s vs. %s (IDs %s vs. %s)" % (
                        summ_ids[i], summ_ids[j], unique_summ_id_pair[0], unique_summ_id_pair[1]))
                    full_entry = sorted_scores[article_id]
                    print("  system %s with score %s (%s) vs." % (
                        full_entry[i]['sys_name'], full_entry[i]['scores']['overall'], entry[summ_ids[i]]))
                    print("  system %s with score %s (%s)" % (
                        full_entry[j]['sys_name'], full_entry[j]['scores']['overall'], entry[summ_ids[j]]))
                    print(
                        "  \"%s...\" vs. \"%s...\"" % (full_entry[i]['sys_summ'][:20], full_entry[j]['sys_summ'][:20]))

                # get keys from dictionary
                entry_keys = list(entry.keys())

                # get pair preference from pair_anno_scores
                for pair in pair_anno_scores[article_id]:
                    if pair['summ_id_i'] == int(entry_keys[i][8]) and pair['summ_id_j'] == int(entry_keys[j][8]):

                        if pair['pref'] == 1:
                            pref = [1, 0]
                        else:
                            pref = [0, 1]

                    else:
                        if pair['summ_id_j'] == int(entry_keys[i][8]) and pair['summ_id_i'] == int(entry_keys[j][8]):

                            if pair['pref'] == 1:
                                pref = [0, 1]
                            else:
                                pref = [1, 0]


                # old code
                # if entry[summ_ids[i]] > entry[summ_ids[j]]:
                #    pref = [1, 0]
                # elif entry[summ_ids[i]] < entry[summ_ids[j]]:
                #    pref = [0, 1]
                # else:
                #    pref = [0.5, 0.5]

                # sort the ids so that we get a unique key, so that (sys_summ0,sys_summ1) and (sys_summ1,sys_summ0) are the same
                if unique_summ_id_pair[1] < unique_summ_id_pair[0]:
                    unique_summ_id_pair = unique_summ_id_pair[::-1]
                    pref = pref[::-1]
                # convert to tuple, otherwise its not hashable for the dict
                unique_summ_id_pair = tuple(unique_summ_id_pair)
                # add up the pref to the total pref vector of the specific summary pair. create a new entry if not existing
                article_prefs[unique_summ_id_pair] = article_prefs.get(unique_summ_id_pair,
                                                                       np.array([0, 0])) + np.array(pref)
        # transform to target
        for unique_summ_id_pair, pref in article_prefs.items():
            # depending on the mode, use binary target, or graded one
            pref = (pref / (pref[0] + pref[1])).tolist()
            if target_type == 'binary':
                if pref[0] > pref[1]:
                    pref = [1, 0]
                elif pref[0] < pref[1]:
                    pref = [1, 0]
                else:
                    pref = [0.5, 0.5]
            # skip if it is a tie and you want to ignore ties
            if pref[0] != 0.5 or not ignore_ties:
                # include the pref two times, once in one direction and once in the other direction
                if double_prefs:
                    pair_list.append((article_id, unique_summ_id_pair[1], unique_summ_id_pair[0], pref[::-1]))
                    pair_list.append((article_id, unique_summ_id_pair[0], unique_summ_id_pair[1], pref))
                else:
                    # include the pref in the reverse order by chance. this might be necessary if there is a bias in the distribution of the score, e.g. if they are ordered
                    if randomize_pref_order and bool(random.getrandbits(1)):
                        pair_list.append((article_id, unique_summ_id_pair[1], unique_summ_id_pair[0], pref[::-1]))
                    else:
                        pair_list.append((article_id, unique_summ_id_pair[0], unique_summ_id_pair[1], pref))
        topic_count += 1
        anno_count += len(summ_ids)
        summ_count += len(summ2id)
    print("topics", topic_count)
    print("annotations", anno_count)
    print("summ", summ_count)
    print("summ pairs", len(pair_list))
    return pair_list


# randomize_pref_order and double_prefs are only relevant if the learning function learns f(s0,s1)=pref. in our case, we learn f(s0)=pref[0] and f(s1)=pref[1], so this should be set to False
def build_pairs_majority_preferences(entries, sorted_scores, target_type='graded', ignore_ties=False,
                                     randomize_pref_order=False, double_prefs=False):
    pair_list = []
    topic_count = 0
    anno_count = 0
    summ_count = 0
    entries_text = {}
    # get summary text and matching id
    for article_id, scores_list in tqdm(sorted_scores.items()):
        temp_entry = {}
        summ_ids = [s['summ_id'] for s in scores_list]
        for sid in summ_ids:
            # get summary text
            s_text = [s['sys_summ'] for s in scores_list if s['summ_id'] == sid][0]
            temp_entry['sys_summ' + repr(sid)] = s_text
        # save in dictionary
        entries_text[article_id] = temp_entry

    for article_id in entries:
        entry = entries[article_id]
        summ_ids = list(entry.keys())

        # mapping from summary text to last summary id with that text. that's the one we will use
        summ2id = {entries_text[article_id][summ_id]: summ_id for summ_id in summ_ids}

        # put here the prefs for this article
        article_prefs = {}

        # still run through all pairs
        # really iterate over all pairs. there was an error here before since j started from 1, to prevent i,j=0,0. but this also lead to i,j=x,0 never be chosen the situation i=j is solved otherwise
        for i in range(len(summ_ids)):
            for j in range(len(summ_ids)):
                # run through dictionary containing summ_ids and matching text
                # for key, value in entries_text[article_id].items():
                # get text for current summaries i and j
                #    if key == summ_ids[i]:
                #        text_i = value
                #    elif key == summ_ids[j]:
                #        text_j = value
                text_i = entries_text[article_id][summ_ids[i]]
                text_j = entries_text[article_id][summ_ids[j]]
                # check if text is identical, if yes skip
                if i == j or text_i == text_j:
                    # print("DUPLICATE FOUND: TEXT i", text_i, "TEXT j", text_i)
                    continue
                # get the unique summ ids
                unique_summ_id_pair = [summ2id[text_i], summ2id[text_j]]
                # some debug output
                # noinspection PyUnreachableCode
                if False:
                    print("%s vs. %s (IDs %s vs. %s)" % (
                        summ_ids[i], summ_ids[j], unique_summ_id_pair[0], unique_summ_id_pair[1]))
                    full_entry = sorted_scores[article_id]
                    print("  system %s with score %s (%s) vs." % (
                        full_entry[i]['sys_name'], full_entry[i]['scores']['overall'], entry[summ_ids[i]]))
                    print("  system %s with score %s (%s)" % (
                        full_entry[j]['sys_name'], full_entry[j]['scores']['overall'], entry[summ_ids[j]]))
                    print(
                        "  \"%s...\" vs. \"%s...\"" % (full_entry[i]['sys_summ'][:20], full_entry[j]['sys_summ'][:20]))
                # unique_summ_id_pair.sort()

                if entry[summ_ids[i]] > entry[summ_ids[j]]:
                    pref = [1, 0]
                elif entry[summ_ids[i]] < entry[summ_ids[j]]:
                    pref = [0, 1]
                else:
                    pref = [0.5, 0.5]
                # if entry[unique_summ_id_pair[0]] > entry[unique_summ_id_pair[1]]:
                #    pref = [1, 0]
                # elif entry[unique_summ_id_pair[0]] > entry[unique_summ_id_pair[1]]:
                #    pref = [0, 1]
                # else:
                #    # todo we could completely ignore ties. doesnt change much. low prio
                #    pref = [0.5, 0.5]
                # sort the ids so that we get a unique key, so that (sys_summ0,sys_summ1) and (sys_summ1,sys_summ0) are the same
                if unique_summ_id_pair[1] < unique_summ_id_pair[0]:
                    unique_summ_id_pair = unique_summ_id_pair[::-1]
                    pref = pref[::-1]
                # convert to tuple, otherwise its not hashable for the dict
                unique_summ_id_pair = tuple(unique_summ_id_pair)
                # add up the pref to the total pref vector of the specific summary pair. create a new entry if not existing
                article_prefs[unique_summ_id_pair] = article_prefs.get(unique_summ_id_pair,
                                                                       np.array([0, 0])) + np.array(pref)
        # transform to target
        for unique_summ_id_pair, pref in article_prefs.items():
            # depending on the mode, use binary target, or graded one
            pref = (pref / (pref[0] + pref[1])).tolist()
            if target_type == 'binary':
                if pref[0] > pref[1]:
                    pref = [1, 0]
                elif pref[0] < pref[1]:
                    pref = [1, 0]
                else:
                    pref = [0.5, 0.5]
            # skip if it is a tie and you want to ignore ties
            if pref[0] != 0.5 or not ignore_ties:
                # include the pref two times, once in one direction and once in the other direction
                if double_prefs:
                    pair_list.append((article_id, unique_summ_id_pair[1], unique_summ_id_pair[0], pref[::-1]))
                    pair_list.append((article_id, unique_summ_id_pair[0], unique_summ_id_pair[1], pref))
                else:
                    # include the pref in the reverse order by chance. this might be necessary if there is a bias in the distribution of the score, e.g. if they are ordered
                    if randomize_pref_order and bool(random.getrandbits(1)):
                        pair_list.append((article_id, unique_summ_id_pair[1], unique_summ_id_pair[0], pref[::-1]))
                    else:
                        pair_list.append((article_id, unique_summ_id_pair[0], unique_summ_id_pair[1], pref))
        topic_count += 1
        anno_count += len(summ_ids)
        summ_count += len(summ2id)
    print("topics", topic_count)
    print("annotations", anno_count)
    print("summ", summ_count)
    print("summ pairs", len(pair_list))
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

    for pointer in range(int((len(
            pairs) - 1) / batch_size) + 1):  # there was a bug here. when len(pairs) was a vielfaches of 32, then there was a last batch with [] causing an exception
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
    results = {'rho': [], 'rho_p': [], 'pcc': [], 'pcc_p': [], 'tau': [], 'tau_p': [], 'rho_global': [],
               'pcc_global': [], 'tau_global': []}
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

        rho, rho_p = spearmanr(true_scores, pred_scores)
        pcc, pcc_p = pearsonr(true_scores, pred_scores)
        tau, tau_p = kendalltau(true_scores, pred_scores)
        if not (math.isnan(rho) or math.isnan(pcc) or math.isnan(tau)):
            results['rho'].append(rho)
            results['rho_p'].append(rho_p)
            results['pcc'].append(pcc)
            results['pcc_p'].append(pcc_p)
            results['tau'].append(tau)
            results['tau_p'].append(tau_p)
    rho = spearmanr(true_scores_all, pred_scores_all)[0]
    pcc = pearsonr(true_scores_all, pred_scores_all)[0]
    tau = kendalltau(true_scores_all, pred_scores_all)[0]
    if not (math.isnan(rho) or math.isnan(pcc) or math.isnan(tau)):
        results['rho_global'].append(rho)
        results['pcc_global'].append(pcc)
        results['tau_global'].append(tau)

    if plot_file is not None:
        fig, ax = plt.subplots()

        # true_scores_all=np.array(true_scores_all)
        # pred_scores_all=np.array(pred_scores_all)

        unique = np.sort(np.unique(true_scores_all))
        data_to_plot = [pred_scores_all[true_score == true_scores_all] for true_score in unique]

        # bw_methods determines how soft the distribution curve will be. lower values are more sharp
        ax.violinplot(data_to_plot, showmeans=True, showmedians=True, bw_method=0.2)
        ax.scatter(true_scores_all + np.random.normal(0, 0.1, pred_scores_all.shape[0]), pred_scores_all, marker=".",
                   s=3, alpha=0.5)
        ax.set_title('Comparison and distributions of true values to predicted score')
        ax.set_xlabel('true scores')
        ax.set_ylabel('predicted scores')

        xticklabels = true_scores_all
        ax.set_xticks(true_scores_all)
        print("violin plot written to: %s" % plot_file)
        plt.savefig(plot_file)

    return results


def parse_args(argv):
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
    ap.add_argument('-fn', '--file_name', type=str, help='file name for csv output',
                    default='BetterRewardsStatistics_test.csv')

    args = ap.parse_args(argv)
    return args.epoch_num, args.batch_size, args.train_type, args.train_percent, args.dev_percent, args.learn_rate, args.model_type, args.device, args.seed, args.file_name


def main(argv):
    epoch_num, batch_size, train_type, train_percent, dev_percent, learn_rate, model_type, device, seed, file_name = parse_args(
        argv[1:])

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
    print('file name {}'.format(file_name))
    print('=====Arguments====\n')

    csv_column_names = ['seed', 'learn_rate', 'model_type', 'train_pairs', 'dev_pairs', 'test_pairs', 'epoch_num',
                        'loss_train', 'loss_dev', 'loss_test', 'rho_train', 'rho_p_train', 'pcc_train', 'pcc_p_train',
                        'tau_train', 'tau_p_train', 'rho_train_global', 'pcc_train_global', 'tau_train_global',
                        'rho_dev', 'rho_p_dev', 'pcc_dev', 'pcc_p_dev', 'tau_dev', 'tau_p_dev',
                        'rho_dev_global', 'pcc_dev_global', 'tau_dev_global', 'rho_test', 'rho_p_test', 'pcc_test',
                        'pcc_p_test', 'tau_test', 'tau_p_test', 'rho_test_global', 'pcc_test_global', 'tau_test_global']

    # check if csv_file exists
    if path.exists(file_name):
        csv_exists = True
    else:
        csv_exists = False

    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # if a new csv_file is generated, write column names
        if csv_exists is False:
            writer.writerow(csv_column_names)
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

        # read pair anno scores

        pair_anno_scores = read_pair_anno_scores()

        # train, dev, test, all = parse_split_data(sorted_scores, train_percent, dev_percent)
        train, dev, test, all = parse_split_data_balanced(sorted_scores, train_percent, dev_percent)

        # without majority preferences
        # train_pairs = build_pairs(train)
        # dev_pairs = build_pairs(dev)
        # test_pairs = build_pairs(test)

        # without majority preferences but with pair anno
        # train_pairs = build_anno_pairs(train)
        # dev_pairs = build_anno_pairs(dev)
        # test_pairs = build_anno_pairs(test)

        # with majority preferences
        # train_pairs = build_pairs_majority_preferences(train, sorted_scores)
        # dev_pairs = build_pairs_majority_preferences(dev, sorted_scores)
        # test_pairs = build_pairs_majority_preferences(test, sorted_scores)

        # with majority preferences and pair anno
        train_pairs = build_anno_pairs_majority_preferences(train, sorted_scores, pair_anno_scores)
        dev_pairs = build_anno_pairs_majority_preferences(dev, sorted_scores, pair_anno_scores)
        test_pairs = build_anno_pairs_majority_preferences(test, sorted_scores, pair_anno_scores)

        print(len(train_pairs), len(dev_pairs), len(test_pairs))

        # read bert vectors
        with open('data/doc_summ_bert_vectors.pkl', 'rb') as ff:
            all_vec_dic = pickle.load(ff)

        pcc_list = []
        weights_list = []

        for ii in range(epoch_num + 1):
            print('\n=====EPOCH {}====='.format(ii))
            if ii == 0:

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
            print('--> losses (train,dev,test)', loss_train, loss_dev, loss_test)

            # Train-Data only
            print("==Train==")
            results_train = test_rewarder(all_vec_dic, train, deep_model, device)
            for metric in results_train:
                print('{}\t{}'.format(metric, np.mean(results_train[metric])))
                csv_row.append(np.mean(results_train[metric]))

            print("==Dev==")
            results = test_rewarder(all_vec_dic, dev, deep_model, device)
            for metric in results:
                print('{}\t{}'.format(metric, np.mean(results[metric])))
                csv_row.append(np.mean(results[metric]))

            # Test-Data only
            print("==Test==")
            results_test = test_rewarder(all_vec_dic, test, deep_model, device)
            for metric in results_test:
                print('{}\t{}'.format(metric, np.mean(results_test[metric])))
                csv_row.append(np.mean(results_test[metric]))

            writer.writerow(csv_row)
            pcc_list.append(np.mean(results['pcc']))
            weights_list.append(copy.deepcopy(deep_model.state_dict()))

        idx = np.argmax(pcc_list)
        best_result = pcc_list[idx]
        print('\n======Best results come from epoch no. {}====='.format(idx))

        deep_model.load_state_dict(weights_list[idx])
        output_pattern = 'batch{}_{}_trainPercent{}_seed{}_lrate{}_{}_epoch{}'.format(
            batch_size, train_type, train_percent, seed, learn_rate, model_type, epoch_num
        )
        test_results = test_rewarder(all_vec_dic, test, deep_model, device,
                                     os.path.join(OUTPUTS_DIR, output_pattern + '_onTest.pdf'))
        test_rewarder(all_vec_dic, train, deep_model, device,
                      os.path.join(OUTPUTS_DIR, output_pattern + '_onTrain.pdf'))
        test_rewarder(all_vec_dic, dev, deep_model, device, os.path.join(OUTPUTS_DIR, output_pattern + '_onDev.pdf'))
        print('Its performance on the test set is:')
        for metric in test_results:
            print('{}\t{}'.format(metric, np.mean(test_results[metric])))
        model_weight_name = 'pcc{0:.4f}_'.format(np.mean(test_results['pcc']))
        model_weight_name += 'seed{}_epoch{}_batch{}_{}_trainPercent{}_lrate{}_{}.model'.format(
            seed, epoch_num, batch_size, train_type, train_percent, learn_rate, model_type
        )

        torch.save(weights_list[idx], os.path.join(MODEL_WEIGHT_DIR, model_weight_name))
        print('\nbest model weight saved to: {}'.format(os.path.join(MODEL_WEIGHT_DIR, model_weight_name)))


if __name__ == '__main__':
    main(sys.argv)
