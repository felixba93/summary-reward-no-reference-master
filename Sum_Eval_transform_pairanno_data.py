import pandas as pd
import json

# pandas
df = pd.read_csv("SumEval/data/PairAnno.csv")
json_data = {}

# Function to check if a given system has an entry in a given topic
def check_system(json, topic, sys_name):
    json_entry = json[topic]
    for summ in json_entry:
        if summ['sys_name'] == sys_name:
            return True
    else:
        return False

# loop through csv
for index, row in df.iterrows():
    summ_list = []
    entry = {}
    score_list = {}
    # get values for one row
    method = row['method_i']

    # catch different folder name
    if method == 'MMR*':
        method = 'MMR_star'

    topic = row['topic']

    # create score_list (could be merged with the steps above, separated for more clarity and flexibility)
    score_list['edit'] = None
    score_list['clarity'] = 0.0
    score_list['grammar'] = 0.0
    score_list['hter'] = 0.0
    score_list['time'] = 0.0
    score_list['overall'] = 0.0
    score_list['focus'] = 0.0
    score_list['redundancy'] = 0.0
    # read summary text
    with open("SumEval/data/summaries/" + str(method) + "/" + str(topic) + ".txt", 'r') as file:
        sys_summ = file.read()
    # create entry
    entry['id'] = str(topic)
    entry['ref'] = "Reference-Summary"
    entry['sys_name'] = str(method)
    entry['sys_summ'] = sys_summ
    entry['scores'] = score_list
    entry['summ_id'] = 0
    entry['rank'] = 0.0
    # fill json structure
    # check if topic already exists
    if topic in json_data:
        print("topic exists")
        # check if system already exists
        system_check = check_system(json_data, topic, entry['sys_name'])

        if system_check:
            continue
        else:
            entry['summ_id'] = len(json_data[topic])
            json_data[topic].append(entry)
            print("Entry extended for", topic)
    else:
        print("New entry created for", topic)
        summ_list.append(entry)
        json_data[topic] = summ_list

with open('SumEval/data/sorted_scores_pair_anno_sumEval.json', 'w') as write_file:
    json.dump(json_data, write_file)
