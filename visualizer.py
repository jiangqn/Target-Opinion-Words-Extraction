import matplotlib.pyplot as plt

# class Visualizer(object):
#
#     def __init__(self, var_list, save_path):
#         super(Visualizer, self).__init__()
#         self.var_list = var_list
#         self.save_path = save_path
#         self.data = {}
#         for var in var_list:
#             self.data[var] = []
#
#     def add(self, values):
#         assert len(values) == len(self.var_list)
#         for var, value in zip(self.var_list, values):
#             self.data[var].append(value)
#
#     def plot(self):
#         plt.figure(figsize=(10, 5))
#         for i in range(1, len(self.var_list)):
#             plt.plot(self.data[self.var_list[0]], self.data[self.var_list[i]], label=self.var_list[i])
#         plt.xlabel(self.var_list[0])
#         plt.ylabel("%")
#         plt.legend(self.var_list[1:])
#         plt.savefig(self.save_path)

class Visualizer(object):

    def __init__(self, save_path):
        super(Visualizer, self).__init__()
        self.save_path = save_path
        self.epoch = []
        self.train_loss = []
        self.train_precision = []
        self.train_recall = []
        self.train_f1_score = []
        self.val_loss = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1_score = []

    def add(self, epoch, train_loss, train_precision, train_recall, train_f1_score, val_loss, val_precision, val_recall, val_f1_score):
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.train_precision.append(train_precision)
        self.train_recall.append(train_recall)
        self.train_f1_score.append(train_f1_score)
        self.val_loss.append(val_loss)
        self.val_precision.append(val_precision)
        self.val_recall.append(val_recall)
        self.val_f1_score.append(val_f1_score)


    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        # raise ValueError(self.epoch, self.train_loss)
        plt.plot(self.epoch, self.train_loss, label='train_loss')
        plt.plot(self.epoch, self.val_loss, label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train_loss', 'val_loss'])
        plt.subplot(2, 2, 2)
        plt.plot(self.epoch, self.train_precision, label='train_precision')
        plt.plot(self.epoch, self.val_precision, label='val_precision')
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.legend(['train_precision', 'val_precision'])
        plt.subplot(2, 2, 3)
        plt.plot(self.epoch, self.train_recall, label='train_recall')
        plt.plot(self.epoch, self.val_recall, label='val_recall')
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.legend(['train_recall', 'val_recall'])
        plt.subplot(2, 2, 4)
        plt.plot(self.epoch, self.train_f1_score, label='train_f1_score')
        plt.plot(self.epoch, self.val_f1_score, label='val_f1_score')
        plt.xlabel('epoch')
        plt.ylabel('f1_score')
        plt.legend(['train_f1_score', 'val_f1_score'])
        plt.savefig(self.save_path)