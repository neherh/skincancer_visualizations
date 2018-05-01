import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

preds  = np.random.randint(low = 2,high =100,size = (1000,10))
# target = np.random.randint(low = 0, high = 2,size = (1000,10))
target = np.zeros(shape = (1000,10))
# target[:,1] = 1


# preds = np.random.randint(50, size=(20, 10))
# target = np.random.randint(30,size=(20,10))

print("predictions")
print(preds)

val = np.array([0,1,2,3,4,5,6,7,8,9])

idx_preds = np.argsort(preds, axis=0)



# print(preds[idx_preds[0][0]])
print("boll")
print(preds[idx_preds[0:10][:],val])

new_preds = preds[idx_preds[0:10][:],val]
new_target = target[idx_preds[0:10][:],val]

print("preds")
print(new_preds)

print("target")
print(new_target)

precision = dict()
recall = dict()
average_precision = dict()
# print('===> re-calculate mAP, calc PR')
# for i in range(10):
#     precision[i], recall[i], _ = precision_recall_curve(new_target[:, i],
#                                                     new_preds[:, i])
#     average_precision[i] = average_precision_score(new_target[:, i], new_preds[:, i])

# print('===> re-calculate mAP, calc mAP')
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(new_target.ravel(),
#     new_preds.ravel())
average_precision["micro"] = average_precision_score(new_target, new_preds,
                                                    average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    .format(average_precision["micro"]))
