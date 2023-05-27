# Importing libraries
import aprel
import numpy as np
import gym

# Feature function for the mountain car environment
def feature_func(traj):
    states = np.array([pair[0] for pair in traj])
    min_pos, max_pos = states[:,0].min(), states[:,0].max()
    mean_speed = np.abs(states[:,1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec

# Main function
def main(args):
    # Create the OpenAI Gym environment
    gym_env = gym.make('MountainCarContinuous-v0') # mountain car
    
    # Seed for reproducibility
    np.random.seed(40)
    gym_env.seed(40)

    # Wrap the environment with a feature function
    env = aprel.Environment(gym_env, feature_func)

    # Create a trajectory set
    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=args['num_trajectories'],
                                                          max_episode_length=args['max_episode_length'],
                                                          file_name='MountainCarContinuous-v0', restore=True,
                                                          headless=True, seed=40)
    features_dim = len(trajectory_set[0].features)

    # Initialize the query optimizer
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # true user
    true_user = aprel.HumanUser(delay=0.5)
    
    # Create the human response model and initialize the belief distribution
    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params, logprior=aprel.uniform_logprior,
                                       num_samples=args['num_samples'],
                                       proposal_distribution=aprel.gaussian_proposal,
                                       burnin=200, thin=20)
    # Report the metrics
    print('\n\nEstimated user parameters: ' + str(belief.mean))

    # Initialize a dummy query so that the query optimizer will generate queries of the same kind
    if args['query_type'] == 'preference':
        query = aprel.PreferenceQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'weak_comparison':
        query = aprel.WeakComparisonQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'full_ranking':
        query = aprel.FullRankingQuery(trajectory_set[:args['query_size']])
    else:
        raise NotImplementedError('Unknown query type.')

    # Active learning loop
    for query_no in range(args['num_iterations']):
        # Optimize the query
        queries, objective_values = query_optimizer.optimize(args['acquisition'], belief,
                                                             query, batch_size=args['batch_size'], 
                                                             optimization_method=args['optim_method'],
                                                             reduced_size=100,
                                                             gamma=1,
                                                             distance=aprel.default_query_distance)
        print('Objective Values: ' + str(objective_values))

        # Ask the query to the human
        responses = true_user.respond(queries)
        
        #Update the belief distribution
        belief.update([aprel.Preference(query, response) for query, response in zip(queries, responses)])
        
        # Report the metrics
        print('Estimated user parameters: ' + str(belief.mean))

# Run the main function
if __name__ == '__main__':
    # Parse the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=40,
                        help='Number of trajectories in the discrete trajectory set for query optimization.')
    parser.add_argument('--max_episode_length', type=int, default=None,
                        help='Maximum number of time steps per episode ONLY FOR the new trajectories. Defaults to no limit.')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for the sampling based belief.')
    parser.add_argument('--query_type', type=str, default='preference',
                        help='Type of the queries that will be actively asked to the user. Options: preference, weak_comparison, full_ranking.')
    parser.add_argument('--query_size', type=int, default=2,
                        help='Number of trajectories in each query.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of iterations in the active learning loop.')
    parser.add_argument('--optim_method', type=str, default='exhaustive_search',
                        help='Options: exhaustive_search, greedy, medoids, boundary_medoids, successive_elimination, dpp.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size can be set >1 for batch active learning algorithms.')
    parser.add_argument('--acquisition', type=str, default='random',
                        help='Acquisition function for active querying. Options: mutual_information, volume_removal, disagreement, regret, random, thompson')

    args = vars(parser.parse_args())

    main(args)