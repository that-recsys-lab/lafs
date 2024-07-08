# # Data Generation for SCRUF Experiments through LAFS
#
# This code generates simulated recommender system output through a process of LAtent Factor Simulation (LAFS).
#
# For each user, there is a list of items and an associated score. Users can be produced with different propensities towards the features of items, which may be sensitive or not.
# User propensities can be segmented temporally into multiple regimes: such that users with certain characteristics occur first and a set of users with different propensities show up next.

# ## Input - DataGenParameters
#
# Encapsulates the parameters needed to do the generation
#
# * `num_items`: number of items (int)
# * `num_factors`: number of factors (int)
# * `item_feature_propensities`: the distributions used to generate item models ([int x num_factors])
# * `std_dev_factors`: standard deviation for the factor generation (float <0.0,1.0>)
# * `num_agents`: number of agents/protected factors (int)
# * `agent_discount`: subtraction for agents associated items ([(mean,variance) x num_agents])
# * `items_dependency`: an indication whether the first two item protected factors are co-dependent (boolean)
# * `num_users_per_propensity`: number of users per user propensity [int x number of user propensity groups]
# * `user_feature_propensities`: the distributions used to generate user models ( [(propensity) x number of factors] x number of user propensity groups )
# * `initial_list_size`: the size of the list generated for each user (int)
# * `recommendation_size`: the size of the recommendation list delivered as output (int)

import numpy as np
import random
import csv

class DataGenParameters:
    
    def __init__(self):
        self.num_items = 0
        self.num_factors = 0
        self.item_feature_propensities = None
        self.std_dev_factors = 0
        self.num_agents = 0
        self.agent_discount = None
        self.items_dependency = False
        self.num_users_per_propensity = [0]
        self.user_feature_propensities = None       
        self.initial_list_size = 0
        self.recommendation_size = 0

class DataGenIMF:
    
    DEFAULT_USER_PROPENSITY = (0.0, 1.0)
    DEFAULT_ITEM_PROPENSITY = 0.5
    
    def __init__(self, params: DataGenParameters):
        self.params = params
        self.users = None
        self.items = None
        self.user_factors = None
        self.item_factors = None
        self.ratings = None
        
    def generate_data(self):
        self.generate_users()
        self.generate_items()
        self.generate_factors()
        self.generate_ratings()
        
    def normalize_users(self):
        users_min, users_max = np.amin(self.users), np.amax(self.users)
        self.users_normalized = []
        for i, score in enumerate(self.users):
            self.users_normalized.append((score-users_min) / (users_max-users_min))
    
    def save_compatibilities(self):
        with open('compatibilities_users_factors.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(['user_id','agent_id','compatibility'])
            for user_id,row_user in enumerate(self.users_normalized):  
                for agent_id,compatibility in enumerate(row_user):
                    write.writerow([user_id,agent_id,compatibility])
                    
    def save_items_factors(self):
        with open('items_factors.csv', 'w') as f:
            write = csv.writer(f)
            for item_id,row_item in enumerate(self.items):  
                for factor_id,compatibility in enumerate(row_item):
                    write.writerow([item_id,factor_id,compatibility])
        
    def generate_users(self):

        self.users = []
        for i,num_users in enumerate(self.params.num_users_per_propensity):
        
            if (self.params.user_feature_propensities is None) or (i >= len(self.params.user_feature_propensities)):  
                    user_feature_propensities = [self.DEFAULT_USER_PROPENSITY] * self.params.num_factors
            else:
                    user_feature_propensities = self.params.user_propensities[i]

            for j in range(num_users):
                user_j = []
                for factor in range(self.params.num_factors):
                    mu_factor = user_feature_propensities[factor][0]
                    sigma_factor = user_feature_propensities[factor][1]
                    user_ij = np.random.normal(loc=mu_factor, scale=sigma_factor)
                    user_j.append(user_ij)
                self.users.append(user_j)
                
        self.normalize_users()
        self.save_compatibilities()
    
    def generate_items(self):
        if self.params.item_feature_propensities is None:
            self.params.item_feature_propensities = \
                [self.DEFAULT_ITEM_PROPENSITY] * self.params.num_factors
        self.items = []
        for i in range(self.params.num_items):
            item_i = []
            for factor in range(self.params.num_factors):
                if self.params.items_dependency and factor==1 and item_i[0]==1:
                    feature_p = 0.9
                else:
                    feature_p = self.params.item_feature_propensities[factor]
                choice_weights = (1 - feature_p, feature_p)
                item_ij = random.choices([0, 1], weights=choice_weights)
                item_i += item_ij
            self.items.append(item_i)
            
        self.save_items_factors()

    def generate_factors(self):
        self.user_factors = self._create_latent_factors(self.users)
        self.item_factors = self._create_latent_factors(self.items)
        np.savetxt("user_latent_factors.csv", self.user_factors, delimiter=",")
        np.savetxt("item_latent_factors.csv", self.item_factors, delimiter=",")
        
    def _create_latent_factors(self, profile):
        factors = []
    
        for profile_i in profile:
            factor_i = []
            for j,profile_ij in enumerate(profile_i):
                if j+1 > self.params.num_agents:    # not an agent factor
                    factor_ij = np.random.normal(loc=0.0, 
                                                      scale=self.params.std_dev_factors, 
                                                      size=None)
                else:      # an agent factor
                    factor_ij = np.random.normal(loc=profile_ij, 
                                           scale=self.params.std_dev_factors, 
                                           size=None)
                factor_i.append(factor_ij)
            factors.append(factor_i)
        
        return np.array(factors)
    
    def generate_ratings(self):     
        self.ratings = []
        
        for user, user_factor in enumerate(self.user_factors):         
            list_items = np.random.choice(self.params.num_items, 
                                          size=self.params.initial_list_size, 
                                          replace=False)
            
            user_listitems_ratings = []
            for item in list_items:
                item_factor = self.item_factors[item]
                score = np.dot(user_factor, item_factor)
                
                discount = 0
                if self.params.agent_discount:
                    discounts = []
                    for factor_id,discount_params in enumerate(self.params.agent_discount):
                        if self.items[item][factor_id] == 1:
                            discount = np.random.normal(loc=discount_params[0], scale=discount_params[1])
                            discounts.append(discount)
                    if discounts:
                        discount = sum(discounts)/len(discounts) 
                
                user_listitems_ratings.append((user, item, score-discount))
                
            user_listitems_ratings.sort(key=lambda user_listitems_rating: user_listitems_rating[2], reverse=True)
            user_listitems_ratings = user_listitems_ratings[0:self.params.recommendation_size]
            
            self.ratings += user_listitems_ratings
            
        self.save_ratings('ratings.csv')
    
    def save_ratings(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            fields = ["user_id", "item_id", "score"]
            writer.writerow(fields)
    
            for user, item, score in self.ratings:
                writer.writerow([user, item, score])


# # Example

# ## Setup of the Parameters
# params = DataGenParameters()
# params.num_items = 1000
# params.num_factors = 10
# params.item_feature_propensities=[0.1, 0.3, 0.9, 0.5, 0.6, 0.2, 0.5, 0.7, 0.6, 0.1]
# params.std_dev_factors = 1.0
# params.num_agents = 3
# params.agent_discount = [(0.5, 0.1),(0.0, 0.0),(0.0, 0.0)]
# params.items_dependency = False
# params.num_users_per_propensity= [100,100]
# params.user_propensity = [[(0.9, 0.1),(0.1, 0.1),(0.1, 0.1), (0.3, 1.0),(0.6, 1.0),(0.1, 0.6), (0.4, 1.0),(0.9, 1.0),(0.1, 0.6), (0.0, 1.0)],
#                            [(0.5, 0.5),(0.5, 0.5),(0.5, 0.5), (0.3, 1.0),(0.6, 1.0),(0.1, 0.6), (0.4, 1.0),(0.9, 1.0),(0.1, 0.6), (0.0, 1.0)]]
# params.initial_list_size = 200
# params.recommendation_size = 50

# ## Generating of the Output Data
# generator = DataGenIMF(params)
# generator.generate_data()
# generator.ratings