import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.stats import beta
from pre_processing import PreProcess
import sys
import os
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(color_codes=True)

class LearnPreferences(object):
    def __init__(self, df1, df2, metrics, ref_listing, num_matches=30):
        """
        Input:
        Two pre-processed dataframe obejcts, list of metrics (e.g. 'walk_distance', 'space_distance'),
        index of the seed house, the number of matches to return per metric,
        """
        col = df1.columns
        self.df1 = pd.DataFrame(df1.iloc[0,:].to_numpy().reshape(1,8),columns = col)
        self.df2 = df2
        self.ref_listing = ref_listing # the seed house
        self.num_matches = num_matches
        self.recommendations = None
        self.recommendation_history = {}
        self.pairs_served = 0
        self.metrics = metrics
        self.reco_graph = dict((k, []) for k in metrics)
        self.current_pairs = None
        self.scores = {}
        self.params = {}
        self.init_scores_and_params()
        self.sim_mat = self.compute_similarity_matrix(df1, df2, 'cosine')
        self.recommendations = self.update_recommendations()
    
    def init_scores_and_params(self): 
        for metric in self.metrics:
            if metric not in self.scores:
                self.scores[metric] = 0
            if metric not in self.params:
                self.params[metric] = (0, 0)

    def compute_similarity_matrix(self, df1, df2, distance_metric):
        self.sim_mat = pairwise_distances(df1, df2, metric=distance_metric) 
    
    def update_recommendations(self):
        def propose_recommendations(self, metric):

            return res
        if self.pairs_served < 1:
            self.recommendations = {}
            for metric in self.metrics:
                # self.recommendations[metrics] = propose_recommendations(metric)
                self.compute_similarity_matrix(PreProcess(self.df1.copy()).filter_df(metric),PreProcess(self.df2.copy()).filter_df(metric) ,'euclidean')
                self.recommendations[metric] = np.argsort(self.sim_mat[0])[-(self.num_matches):].tolist()
        return self.recommendations

    # def get_most_similar(self, sim_mat, num_matches):
    #     '''
    #     Input: similarity matrix with first arg of parwise distances as rows and
    #            second arg of pairwise distances as columns, the integer index of the 
    #            listing you want to compare other listings to, int for the num of listings
    #            to return.
    #     Output: an numpy array with the indices of the listings that
    #             are most similar to the ref_listing.
    #     '''
    #     return np.argsort(self.sim_mat[self.ref_listing])[-(num_matches):]

    def show_recommendations(self):
        sample_metrics = self.choose_models()
        recommendations = []
        for metric in sample_metrics: # get a recommednation from each of the metrics in this iteration
            recommendations.append(self.get_recommendation(metric))
        self.current_pairs = recommendations

    def get_recommendation(self, metric):
        """
        Input: similarity matrix with first arg of parwise distances as rows and
               second arg of pairwise distances as columns, the integer index of the 
               listing you want to compare other listings to, int for the num of listings
               to return.
        Output: an numpy array with the indices of the listings that
                are most similar to the ref_listing.
        """
        # draw an element at random from the recommendations list
        recommendation = np.random.choice(self.recommendations[metric])
        # remove the element from recommendations
        self.recommendations[metric].pop(self.recommendations[metric].index(recommendation))
        return recommendation
    
    def get_user_choice(self):
        """
        Input: a dataframe for each of the cities
        Output: the recommendation corresponding to the user choice 
        """
        user_choice = 'l'
        sample_metrics = self.choose_models()
        
        if user_choice == "l":
            self.scores[sample_metrics[0]] += 1
            winner = self.current_pairs[0]
        else:
            self.scores[sample_metrics[1]] += 1
            winner = self.current_pairs[1]
        self.pairs_served += 1
        self.update_recommendation_history(self.current_pairs, winner)

    def update_recommendation_history(self, recommendations, winner):
        self.recommendation_history[self.pairs_served] = {'pairs_served': recommendations,'winner': winner}

    def choose_models(self):
        """
        Choose two of the available models, where one is the best estimate of the users preference
        and the other is randomly chosen of the remaining metrics. Assign the best guess to a list
        """
        if self.pairs_served > 0:
            best_guess = self.recommendation_history[self.pairs_served]['estimated_user_preference']
            metrics = [best_guess]
            remaining_metrics = list(self.metrics) # make a copy of the list, so the original is not modified
            remaining_metrics.pop(remaining_metrics.index(best_guess)) # remove the best guess, since it's already in metrics
            metrics.append(np.random.choice(remaining_metrics)) # randomly choose the other metric
            np.random.shuffle(metrics) # shuffle the metrics, so the best guess recommendation is not always the first one presented
        else:
            # this is the starting point and these is no best guess of the best metric
            metrics = np.random.choice(self.metrics, 2, replace=False)
        return metrics 

    def guess_preferences(self):
        """
        Input: no inputs
        Output: no outputs
        Notes: this function will take the updated score for each metric, compute a 
        beta distribution defined by the win/loss scores, sample from each distribution
        and return the metric that corresponds to the greatest probability. The winning
        metric is added to recommendation_history as the best guess of user preference.
        """
        user_preference = None
        max_prob = 0
        for metric in self.metrics:
            self.params[metric] = (self.scores[metric] , self.pairs_served - self.scores[metric])
            prob = beta.rvs(self.params[metric][0] + 1, self.params[metric][1] + 1)
            # sample form the dist for each metric
            if prob > max_prob:
                max_prob = prob
                user_preference = metric
        self.recommendation_history[self.pairs_served]['estimated_user_preference'] = user_preference

    def generate_results(self):
        fig = plt.figure()
        for metric in self.metrics:
            a = self.params[metric][0] + 1
            b = self.params[metric][1] + 1
            x = beta.rvs(a, b, loc=0, scale=1, size=1000)
            sns.kdeplot(x, shade=True, label=metric)
            self.reco_graph[metric].append(self.scores[metric]/np.sum(list(self.scores.values())))
        plt.savefig("prob_dist.png", dpi=600)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,8))
        res = list(self.reco_graph.values())
        ax.stackplot(range(len(res[0])),res[0],res[1],res[2], labels = self.metrics)
        ax.set_xlabel('Recommendation iterations')
        ax.set_ylabel('% of Recommendations')
        lgd = plt.legend()
        ax.set_facecolor('w')
        plt.tight_layout()
        plt.savefig("iterations_prob_dist.png", dpi=600)
        plt.close()
