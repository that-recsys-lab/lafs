# lafs
LAtent Factor Synthesis for Recommendation Data Generation

Authors: Elena Stefancova, Robin Burke, Amanda Aird and others


 ## Underlying representation
 
 The underlying representation is a set of latent factors for users $U$ and items $I$ ala matrix factorizations. The latent factors are divided between "sensitive" features (first $g$), which are tied to specific fairness agents, and "extra" features ($g$ through $k$), which are not. 


Latent factors are generated on a per-user or per-item basis based on user and feature profiles.

User and feature profiles are generated from the feature-specific propensities.
 
## Process
 
 Input: 
 
* num_items: number of items (int)
* num_factors: number of factors (int)
* item_feature_propensities: the distributions used to generate item models ([int x num_factors])
* std_dev_factors: standard deviation for the factor generation (float <0.0,1.0>)
* num_agents: number of agents/protected factors (int)
* agent_discount: subtraction for agents associated items ([(mean,variance) x num_agents])
* items_dependency: an indication whether the first two item protected factors are co-dependent (boolean)
* num_users_per_propensity: number of users per user propensity [int x number of user propensity groups]
* user_feature_propensities: the distributions used to generate user models ( [(propensity) x number of factors] x number of user propensity groups )
* initial_list_size: the size of the list generated for each user (int)
* recommendation_size: the size of the recommendation list delivered as output (int)

 
 1. Define users (sample from the user propensities for each feature)
 2. Define items (sample [0,1] with probabilities from the item propensities)
 3. Compute user latent factors. Treat the profile value for each user-feature as the mean around which the sampling happens.
 4. Similar for item latent factors.
 5. To generate user interaction data, pick $l$ random items and compute $u_i \dot i_j$ for a predicted ratings.
 6. Sort the items and retain the top $\alpha$ fraction. (This simulates missing not at random dynamics.)
