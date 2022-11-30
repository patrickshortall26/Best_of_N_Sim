import numpy as np

def average_experiment(results):
    """
    Take the results from an experiment and return 
    the average average opinion at each time step along with the
    standard deviation

    Parameters
    ----------
    results : dataframe
        results dataframe from an experiment run

    Returns
    -------
    Returns a tuple containing the average average opinion
    and the average standard deviation at each time step
    as numpy arrays
    """
    # Extract dataframe with agents opinions
    data = results.variables.BoidsModel

    # Create an empty array to store the results in (num_samples, max_timestep)
    num_samples = data.index.get_level_values(0).max()
    max_timestep = data.index.get_level_values(1).max()
    mean_opinion_data = np.zeros((num_samples,max_timestep))
    std_opinion_data = np.zeros((num_samples,max_timestep))

    # Collect the average opinions at each time step for each simulation run
    for simulation in range(data.index.get_level_values(0).max()):
        # Get dataframe of current simulation
        sim_data = data.loc[simulation]
        # Extract the opinions
        opinions = np.array([*sim_data['opinions']]).T
        # Find the mean and standard deviation at each time step
        mean = np.mean(opinions, 0)
        std = np.std(opinions, 0)
        # Add to 2D array for means and stds on all experiments
        mean_opinion_data[:len(mean)] = mean
        std_opinion_data[:len(std)] = std

    # Calculate the average of each column not counting zero values
    average_opinion = np.mean(np.ma.masked_equal(mean_opinion_data, 0), 0)
    average_std = np.mean(np.ma.masked_equal(std_opinion_data, 0), 0)

    return average_opinion, average_std


