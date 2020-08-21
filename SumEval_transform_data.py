import pandas as pd
import json

# pandas
df = pd.read_csv("SumEval/data/LikertAnno.csv")
json_data = {}
# loop through csv
for index, row in df.iterrows():
    summ_list = []
    entry = {}
    score_list = {}
    # get values for one row
    method = row['method']
    annotator = row['annotator']
    topic = row['topic']
    non_redundancy = row['Non-Redundancy']
    ref_clarity = row['Referential Clarity']
    grammar = row['Grammaticality']
    focus = row['Focus']
    structure = row['Structure']
    coherence = row['Coherence']
    readability = row['Readability']
    info_content = row['Information Content']
    spelling = row['Spelling']
    length = row['Length']
    quality = row['Overall Quality']
    # create score_list (could be merged with the steps above, seperated for more clarity and flexibility)
    score_list['edit'] = None
    score_list['clarity'] = float(ref_clarity)
    score_list['grammar'] = float(grammar)
    score_list['hter'] = float(structure)
    score_list['time'] = float(coherence)
    score_list['overall'] = float(quality)
    score_list['focus'] = float(focus)
    score_list['structure'] = float(structure)
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
    entry['rank'] = structure
    # fill json structure
    if topic in json_data:
        entry['summ_id'] = len(json_data[topic])
        json_data[topic].append(entry)
        print("Entry extended for", topic)
    else:
        print("New entry created for", topic)
        summ_list.append(entry)
        json_data[topic] = summ_list

with open('SumEval/data/sorted_scores_likert_structure.json', 'w') as write_file:
    json.dump(json_data, write_file)
