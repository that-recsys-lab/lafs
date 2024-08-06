import numpy as np
import random
import csv
import tomli
from pathlib import Path
import argparse

class DataGenParameters:
    '''Holds parameters for data generation. Parameters can be loaded from a TOML file.

    `num_items`: number of items (int)
    `num_factors`: number of factors (int)
    `item_feature_propensities`: the distributions used to generate item models ([int x num_factors])
    `std_dev_factors`: standard deviation for the factor generation (float <0.0,1.0>)
    `num_sensitive_features`: number of agents/protected factors (int)
    `feature_bias`: subtraction for agents associated items ([(mean,variance) x num_agents])
    `items_dependency`: an indication whether the first two item protected factors are co-dependent (boolean)
    `num_users_per_propensity`: number of users per user propensity [int x number of user propensity groups]
    `user_feature_propensities`: the distributions used to generate user models ( [(propensity) x number of factors] x number of user propensity groups )
    `initial_list_size`: the size of the list generated for each user (int)
    `recommendation_size`: the size of the recommendation list delivered as output (int)
    `ratings_range`: the range to rescale final ratings
    `compatibilities_file`: the file to store user compatibilites
    `item_features_file`: the file to store item feature associations
    `user_factors_file`: the file to store user factor matrix
    `item_factors_file`: the file to store item factor matrix
    `ratings_file`: the file to store final (simulated) recommendation output
    '''

    DEFAULT_COMPATIBILITIES_FILE: str = 'user_compatibilities.csv'
    DEFAULT_ITEM_FEATURES_FILE: str = 'item_features.csv'
    DEFAULT_USER_FACTORS_FILE: str = 'user_factors.csv'
    DEFAULT_ITEM_FACTORS_FILE: str = 'item_factors.csv'
    DEFAULT_RATINGS_FILE: str = 'ratings.csv'

    def __init__(self):
        self.config = None
        self.name = 'Unnamed'
        self.num_items = 0
        self.num_factors = 0
        self.item_feature_propensities = None
        self.std_dev_factors = 0
        self.num_sensitive_features = 0
        self.feature_bias = None
        self.items_dependency = False
        self.num_users_per_propensity = [0]
        self.user_feature_propensities = None       
        self.initial_list_size = 0
        self.recommendation_size = 0
        self.ratings_range = None
        self.compatibilities_file = Path(DataGenParameters.DEFAULT_COMPATIBILITIES_FILE)
        self.item_features_file = Path(DataGenParameters.DEFAULT_ITEM_FEATURES_FILE)
        self.user_factors_file = Path(DataGenParameters.DEFAULT_USER_FACTORS_FILE)
        self.item_factors_file = Path(DataGenParameters.DEFAULT_ITEM_FACTORS_FILE)
        self.ratings_file = Path(DataGenParameters.DEFAULT_RATINGS_FILE)


    def load(self, toml_file):
        # Load the TOML file
        with open(toml_file, "rb") as f:
            self.config = tomli.load(f)

        self.setup()

    def from_string(self, toml_str):
        self.config = tomli.loads(toml_str)

        self.setup()

    def setup(self):
        # Initialize class attributes from the TOML configuration
        self.name = self.config.get('name', 'Unnamed')
        self.num_items = self.config.get('num_items')
        self.num_factors = self.config.get('num_factors')
        self.item_feature_propensities = self.config.get('item_feature_propensities')
        self.std_dev_factors = self.config.get('std_dev_factors')
        self.num_sensitive_features = self.config.get('num_sensitive_features')
        self.feature_bias = self.config.get('feature_bias', None)
        self.num_users_per_propensity = self.config.get('num_users_per_propensity')
        self.user_feature_propensities = self.config.get('user_feature_propensities')
        self.initial_list_size = self.config.get('initial_list_size')
        self.recommendation_size = self.config.get('recommendation_size')
        self.ratings_range = self.config.get('ratings_range', None)
        self.compatibilities_file = Path(self.config.get('compatibilities_file',
                                                    DataGenParameters.DEFAULT_COMPATIBILITIES_FILE))
        self.item_features_file = Path(self.config.get('item_features_file',
                                                    DataGenParameters.DEFAULT_ITEM_FEATURES_FILE))
        self.user_factors_file = Path(self.config.get('user_factors_file',
                                                    DataGenParameters.DEFAULT_USER_FACTORS_FILE))
        self.item_factors_file = Path(self.config.get('item_factors_file',
                                                    DataGenParameters.DEFAULT_ITEM_FACTORS_FILE))
        self.ratings_file = Path(self.config.get('ratings_file',
                                                 DataGenParameters.DEFAULT_RATINGS_FILE))


    def __repr__(self):
        return (f"DataGenParameters({self.name})")


class DataGen:
    '''
    Generates simulated recommender system output via LAtent Factor Simulation (LAFS)

    For command line execution, we assume that save=True. (Otherwise, what's the point?)
    Save can be set to false for interactive / notebook use and the components of the data generation
    can be manually saved with the appropriate function calls.

    For each user, there is a list of items and an associated score. Users can be produced with different
    propensities towards the features of items, which may be sensitive or not.
    User propensities can be segmented temporally into multiple regimes: such that users with certain
    characteristics occur first and a set of users with different propensities show up next.
    '''
    
    DEFAULT_USER_PROPENSITY = (0.0, 1.0)
    DEFAULT_ITEM_PROPENSITY = 0.5
    
    def __init__(self, params: DataGenParameters, save=True):
        self.check_params(params)
        self.params = params
        self.save = save

        self.users = None
        self.items = None
        self.user_factors = None
        self.item_factors = None
        self.ratings = None

    def check_params(self, params: DataGenParameters):
        '''Does minimal consistency checking of the parameters.'''
        if params.initial_list_size < params.recommendation_size:
            msg = f'Bad parameters: Initial list size {params.initial_list_size} < Final list size {params.recommendation_size}'
            raise(ValueError(msg))
        if  len(params.num_users_per_propensity) != len(params.user_feature_propensities):
            msg = f'Bad parameters: Users per propensity {params.num_users_per_propensity} \
does not match feature propensity {params.user_feature_propensities}'
            raise(ValueError(msg))
        if params.num_sensitive_features > params.num_factors:
            msg = f'Bad parameters: Number of sensitive features {params.num_sensitive_features} > \
Number of factors {params.num_factors}'
            raise(ValueError(msg))

    '''
    USERS
    '''

    def generate_users(self):
        '''
        Generates a list of users. A user in this context is a list of feature, propensity pairs.
        The propensity values are drawn from the corresponding feature-specific `user_feature_propensities`
        distribution. Note that to support multiple regimes of users, there is a list of propensity lists
        and a list indicating how many users of each type should be generated.
        '''
        self.users = []
        for i, num_users in enumerate(self.params.num_users_per_propensity):
            user_feature_propensities = self.params.user_feature_propensities[i]

            for j in range(num_users):
                user_j = []
                for factor in range(self.params.num_factors):
                    mu_factor = user_feature_propensities[factor][0]
                    sigma_factor = user_feature_propensities[factor][1]
                    user_ij = np.random.normal(loc=mu_factor, scale=sigma_factor)
                    user_j.append(user_ij)
                self.users.append(user_j)

        self.normalize_users()
        if self.save:
            self.save_compatibilities()

    def normalize_users(self):
        '''Scale user propensities between 0..1. Note that normalized scores are only used for outputting
        the compatibilities file.'''
        users_min, users_max = np.amin(self.users), np.amax(self.users)
        self.users_normalized = []
        for i, score in enumerate(self.users):
            self.users_normalized.append((score - users_min) / (users_max - users_min))

    def save_compatibilities(self):
        '''Compatibilities are normalized feature propensities for users.'''
        with open(self.params.compatibilities_file, 'w') as f:
            write = csv.writer(f)
            write.writerow(['user_id', 'feature_id', 'compatibility'])
            for user_id, row_user in enumerate(self.users_normalized):
                for feature_id, compatibility in enumerate(row_user):
                    write.writerow([user_id, feature_id, compatibility])

    '''
    ITEMS
    '''

    def generate_items(self):
        '''
        Generates a list of items. An item in this context is a list of feature, association pairs.
        The associations are binary and generated from the corresponding feature-specific
        `item_feature_propensities` probabilities.
        '''
        self.items = []
        for i in range(self.params.num_items):
            item_i = []
            for factor in range(self.params.num_factors):
                feature_p = self.params.item_feature_propensities[factor]
                choice_weights = (1 - feature_p, feature_p)
                item_ij = random.choices([0, 1], weights=choice_weights)
                item_i += item_ij
            self.items.append(item_i)

        if self.save:
            self.save_item_features()

    def save_item_features(self):
        with open(self.params.item_features_file, 'w') as f:
            write = csv.writer(f)
            for item_id, row_item in enumerate(self.items):
                for feature_id, association in enumerate(row_item):
                    write.writerow([item_id, feature_id, association])

    '''
    LATENT FACTOR MATRIX
    '''

    def generate_factors(self):
        '''Generate the latent factor matrices. User and item factors are handled the same way'''
        self.user_factors = self._create_latent_factors(self.users)
        self.item_factors = self._create_latent_factors(self.items)
        if self.save:
            np.savetxt(self.params.user_factors_file, self.user_factors, delimiter=",")
            np.savetxt(self.params.item_factors_file, self.item_factors, delimiter=",")
        
    def _create_latent_factors(self, profile):
        factors = []
    
        for profile_i in profile:
            factor_i = []
            for j,profile_ij in enumerate(profile_i):
                if j+1 > self.params.num_sensitive_features:    # not one of the sensitive features
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

    '''
    RATINGS
    '''
    
    def generate_ratings(self):
        '''Generate ratings by multiplying the latent factors'''
        raw_ratings = []
        min_rating = float('inf')
        max_rating = float('-inf')
        
        for user, user_factor in enumerate(self.user_factors):         
            list_items = np.random.choice(self.params.num_items, 
                                          size=self.params.initial_list_size, 
                                          replace=False)
            
            user_listitems_ratings = []
            for item in list_items:
                item_factor = self.item_factors[item]
                score = np.dot(user_factor, item_factor)
                
                bias = 0
                if self.params.feature_bias:
                    biases = []
                    for factor_id, bias_params in enumerate(self.params.feature_bias):
                        if self.items[item][factor_id] == 1:
                            feature_bias = np.random.normal(loc=bias_params[0], scale=bias_params[1])
                            biases.append(feature_bias)
                    if biases:
                        bias = sum(biases)/len(biases)

                rating = score - bias
                if rating < min_rating:
                    min_rating = rating
                if rating > max_rating:
                    max_rating = rating
                
                user_listitems_ratings.append((user, item, rating))
                
            user_listitems_ratings.sort(
                key=lambda user_listitems_rating: user_listitems_rating[2],
                reverse=True)
            user_listitems_ratings = user_listitems_ratings[0:self.params.recommendation_size]
            
            raw_ratings += user_listitems_ratings

        if self.params.ratings_range is not None:
            self.ratings = self.normalize_ratings(raw_ratings, min_rating, max_rating)

        if self.save:
            self.save_ratings(self.params.ratings_file)

    def normalize_ratings(self, raw_ratings, min_rating, max_rating):
        '''Use min-max scaling over the whole generated output to normalize ratings.'''
        normalized = [(rating_row[0],
                       rating_row[1],
                       self.normalize_rating(rating_row[2], min_rating, max_rating))
                       for rating_row in raw_ratings]
        return normalized

    def normalize_rating(self, rating, min_rating, max_rating):
        '''First normalize from 0..1 and then adjust for the experimenter-specified range.'''
        normalized01 = (rating - min_rating) / (max_rating - min_rating)
        adj_factor = self.params.ratings_range[1] - self.params.ratings_range[0]
        normalized_range = normalized01 * adj_factor + self.params.ratings_range[0]
        return normalized_range
    
    def save_ratings(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            fields = ["user_id", "item_id", "rating"]
            writer.writerow(fields)
    
            for user, item, score in self.ratings:
                writer.writerow([user, item, score])

    '''
    MAIN FUNCTION
    '''

    def generate_data(self):
        self.generate_users()
        self.generate_items()
        self.generate_factors()
        self.generate_ratings()


# Usage:
# python data_gen.py config.toml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of the config file for data generation")
    args = parser.parse_args()
    config_file = args.config_file

    params = DataGenParameters()
    params.load(config_file)

    print(f'Generating data using {params} loaded from "{config_file}"')

    data_gen = DataGen(params, save=True)

    data_gen.generate_data()
