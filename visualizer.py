import matplotlib.pyplot as plt

class Visualizer(object):

    def __init__(self, var_list, save_path):
        super(Visualizer, self).__init__()
        self.var_list = var_list
        self.save_path = save_path
        self.data = {}
        for var in var_list:
            self.data[var] = []

    def add(self, values):
        assert len(values) == len(self.var_list)
        for var, value in zip(self.var_list, values):
            self.data[var].append(value)

    def plot(self):
        plt.figure(figsize=(10, 5))
        for i in range(1, len(self.var_list)):
            plt.plot(self.data[self.var_list[0]], self.data[self.var_list[i]], label=self.var_list[i])
        plt.xlabel(self.var_list[0])
        plt.ylabel("%")
        plt.legend(self.var_list[1:])
        plt.savefig(self.save_path)