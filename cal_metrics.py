import sys, json
import collections

def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.split(" ")
    pred_toks = a_pred.split(" ")
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == "__main__":
    file_name = sys.argv[1]
    truth = {}
    q = {}
    for i in range(10):
        with open("./data/bnqa_fold{}/dev.json".format(i)) as fp:
            data = json.load(fp)
            for para in data['data'][0]['paragraphs']:
                for qa in para['qas']:
  
                    truth[qa['id']] = [answer['text'] for answer in qa['answers']]
                    q[qa['id']] = qa['question']
    
    results = {}
    for i in range(10):
        with open("{}{}/nbest_predictions_.json".format(file_name, i)) as fp:
            result = json.load(fp)
            results.update(result)
            
    metrics = []
    for top in [1, 3, 5]:
        yes = 0
        yes_ids = set()
        for id, answers in results.items():
            for answer in answers[:top]:
                text = answer['text']
                for t_answer in truth[id]:
                    if t_answer == text:
                        yes+=1
                        yes_ids.add(id)
                        break
        metrics.append( len(yes_ids) / 600)
    print("\nCross Validation Results: ")
    print("SAcc: {:.2%}\t LAcc-top3: {:.2%}\t LAcc-top5: {:.2%}".format(metrics[0], metrics[1], metrics[2]))
    
    
