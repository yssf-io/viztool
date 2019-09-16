import matplotlib.pyplot as plt
from debug import debug

class Visualizer():
    def __init__(self, features):
        self.features = features

    def plotFeature(self, ticker, feature):
        debug("Plotting feature %s" % feature)
        feature_to_plot = self.features[str(ticker)][feature]
        x = [i for i in range(len(feature_to_plot))]
        y = feature_to_plot

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Feature %s for ticker %s" % (feature, ticker))
        ax.set_xlabel("Sub series number")
        ax.set_ylabel("Feature value")
        #ax.set_xticks(x[::200])
        #ax.set_xticklabels(x[::200], rotation=45)
        plt.show()
    
    def compute_features(self):
        #To do: outputs a DataFrame features\stock
        pass