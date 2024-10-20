def calculate_metrics(predict_lst, ground_truth_lst):
    correct = 0
    precision_scores = []
    recall_scores = []
    f1_score = []
    
    for i, prediction in enumerate(predict_lst):
        if prediction in ground_truth_lst:
            correct += 1
            precision = correct / (i + 1)
            recall = correct / len(ground_truth_lst)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_score.append(2 * (precision * recall) / (precision + recall))
    
    if precision_scores and recall_scores:
        ap = sum(precision_scores) / len(ground_truth_lst)
    else:
        ap = 0

    return precision_scores, recall_scores, f1_score, ap
