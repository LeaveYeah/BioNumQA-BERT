import os
import json
with open("data/bnqa.json") as fp:
    bnqa = json.load(fp)
for i in range(10):
    if os.path.isdir('data/bnqa_fold%d' % i):
        shutil.rmtree('data/bnqa_fold%d' % i)
    os.mkdir('data/bnqa_fold{}'.format(i))
    dev_data = bnqa['data'][0]['paragraphs'][i*60: (i+1)*60]
    train_data = bnqa['data'][0]['paragraphs'][:i*60] +  bnqa['data'][0]['paragraphs'][(i+1)*60:]
    with open("data/bnqa_fold{}/dev.json".format(i), 'w') as fp:
         json.dump({"version": "BioNumQA",
                    "data":[{"paragraphs": dev_data, "title":""}]}, fp)
            
    with open("data/bnqa_fold{}/train.json".format(i), 'w') as fp:
         json.dump({"version": "BioNumQA",
                    "data":[{"paragraphs": train_data, "title":""}]}, fp)
