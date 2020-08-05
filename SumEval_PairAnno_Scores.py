import pandas as pd
import json
import statistics

# set criterion for which you want to create the list of pairs: Referential Clarity, Non-Redundancy, Structure, Readability, Information Content,
# Overall Quality
target_criterion = 'Overall Quality'


# function to check for duplicate entries

def check_duplicate(json_data, topic, summ_id_i, summ_id_j, pref):
    duplicate = False
    for entry in json_data[topic]:
        if entry['topic'] == str(topic) and entry['summ_id_i'] == summ_id_i and entry['summ_id_j'] == summ_id_j:
            entry['pref'].append(pref)
            duplicate = True

    return duplicate


# read PairAnno
df = pd.read_csv("SumEval/data/PairAnno.csv")
json_data = {}

# read sorted scores
with open("SumEval/data/sorted_scores_pair_anno_sumEval.json", 'r') as file:
    sorted_scores = json.load(file)

# loop through csv
for index, row in df.iterrows():
    pair_list = []
    entry = {}
    # get values for one row
    method_i = row['method_i']
    method_j = row['method_j']
    topic = row['topic']
    criterion = row['criterion']
    i_greater_j = row['i greater j?']

    # check if target criterion is matched:
    if str(criterion) == target_criterion:

        # get summary IDs from sorted_scores
        for sorted_scores_entry in sorted_scores[str(topic)]:
            if sorted_scores_entry['sys_name'] == str(method_i):
                summ_id_i = sorted_scores_entry['summ_id']
            if sorted_scores_entry['sys_name'] == method_j:
                summ_id_j = sorted_scores_entry['summ_id']

        # create entry
        entry['topic'] = str(topic)
        entry['summ_id_i'] = summ_id_i
        entry['summ_id_j'] = summ_id_j
        entry['pref'] = [i_greater_j]

        # fill json structure

        # check if topic exists
        if topic in json_data:

            # check if this entry has already been made
            duplicate = check_duplicate(json_data, topic, summ_id_i, summ_id_j, i_greater_j)

            if duplicate:
                continue
            else:
                json_data[topic].append(entry)

        else:
            pair_list.append(entry)
            json_data[topic] = pair_list

    else:
        continue

# get mean from all annotations

for topic in json_data:
    for entry in json_data[topic]:
        if statistics.mean(entry['pref']) > 0.5:
            entry['pref'] = 1
        else:
            entry['pref'] = 0

with open('SumEval/data/sumEval_pair_scores.json', 'w') as write_file:
    json.dump(json_data, write_file)
