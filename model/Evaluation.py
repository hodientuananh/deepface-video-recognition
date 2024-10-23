class Evaluation:
    predict_lst = list()
    ground_truth_lst = list()
    
    def __init__(self, predict_lst, ground_truth_lst) -> None:
        self.predict_lst = predict_lst
        self.ground_truth_lst = ground_truth_lst
    
    def calculate_metrics(self):
        correct = 0
        precision_scores = []
        recall_scores = []
        f1_score = []
        
        for i, prediction in enumerate(self.predict_lst):
            if prediction in self.ground_truth_lst:
                correct += 1
                precision = correct / (i + 1)
                recall = correct / len(self.ground_truth_lst)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_score.append(2 * (precision * recall) / (precision + recall))
        
        if precision_scores and recall_scores:
            ap = sum(precision_scores) / len(self.ground_truth_lst)
        else:
            ap = 0

        return precision_scores, recall_scores, f1_score, ap
