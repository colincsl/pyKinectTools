import numpy as np
from collections import Counter
from pyKinectTools.dataset_readers.CADPlayer import read_labels

base_dir = '/Users/colin/Data/CAD_120/'

labels_all = []
labels = read_labels(base_dir, [1,3,4,5], np.arange(10), np.arange(3))
try:
	while 1:
		labels_all += [labels.next()]
except:
	pass

x = labels_all[0]

actions = [x[0]['activity'] for x in labels_all]
subactions = [[y[1][x]['subaction'] for x in y[1]] for y in labels_all]
subactions = np.hstack(subactions)
objects = np.hstack([x[0]['objects'].values() for x in labels_all])
affordances = [np.hstack([y[1][x]['objects'].values() for x in y[1]]) for y in labels_all]
affordances = np.hstack(affordances)


print "Actions:", unique(actions)
print "Total actions:", len(actions)
print
print "Subactions:", np.unique(subactions)
print "Total subactions:", len(subactions)
print
print "Objects:", unique(objects)
print "Total objects:", len(objects)
print
print "Affordances:", np.unique(affordances)
print "Total affordances:", len(affordances)

action_hist = Counter(actions)
subaction_hist = Counter(subactions)
object_hist = Counter(objects.tolist())
affordance_hist = Counter(affordances.tolist())

from pylab import bar, show, xticks

figure(1)
bar(np.arange(10), action_hist.values())
title("Actions")
xticks(np.arange(len(unique(actions)))+.5, action_hist.keys())

figure(2)
bar(np.arange(len(unique(subactions))), subaction_hist.values())
title("Subactions")
xticks(np.arange(len(unique(subactions)))+.5, subaction_hist.keys())

figure(3)
bar(np.arange(len(unique(objects))), object_hist.values())
title("Objects")
xticks(np.arange(len(unique(objects)))+.5, object_hist.keys())

figure(4)
bar(np.arange(len(unique(affordances))), affordance_hist.values())
title("Affordances")
xticks(np.arange(len(unique(affordances)))+.5, affordance_hist.keys())



