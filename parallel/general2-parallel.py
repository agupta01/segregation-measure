import copy
import os
import sys
import time
from multiprocessing import Event, Manager, Process, Queue, cpu_count
from operator import itemgetter

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gerrychain import (GeographicPartition, Graph, MarkovChain, Partition,
                        accept, constraints, proposals, updaters)
from gerrychain.accept import always_accept
from gerrychain.constraints import (LowerBound, UpperBound, Validator,
                                    WithinPercentRangeOfBounds)
from gerrychain.proposals import propose_chunk_flip, propose_random_flip, recom
from gerrychain.random import random  # FOR REPRODUCIBILITY
from gerrychain.updaters import Tally
from tqdm import tqdm, trange
from twilio.rest import Client

# twilio setup, requires proper env variables to be set up (so it will text you when the chain is done)
# account = os.environ['TWILIO_ACCT']
# auth = os.environ['TWILIO_AUTH']
# client = Client(account, auth)


def load_data(city, state, st_FIPS):
    """
    Parameters
    ----------
    city : string
        example: "Atlanta"
    state : string
        example: "GA"
    st_FIPS : string
        example: "130"

    Returns
    -------
    race_matrix : geopandas.GeoDataFrame
        rows are relevant, non-empty tracts, columns are racial groups.
    """
    os.chdir("../")
    # Load race data
    FILEPATH = './data/block_2010_data/nhgis0005_csv'
    relevant_cols = ['GISJOIN', 'STATEA', 'COUNTYA', 'H7X001', 'H7X002', 'H7X003', 'H7X004',
                                'H7X005', 'H7X006', 'H7X007', 'H7X008']
    race_raw = pd.read_csv(f'{FILEPATH}/nhgis0005_ds172_2010_block.csv',
                           usecols=relevant_cols,
                           dtype={'GISJOIN': str, 'STATEA': str, 'COUNTYA': str})
    column_mapper = dict(zip(relevant_cols, ['GISJOIN', 'state_fips', 'county_fips', 'total', 'white',
                                             'black', 'american_indian_al_native', 'asian',
                                             'hawaiian_pac_islander', 'other', 'two_plus']))
    race_raw.rename(columns=column_mapper, inplace=True)
    print("Race data loaded.")
    race_raw.set_index('GISJOIN', inplace=True)

    # Load relevant shapefile and crosswalks
    city_blocks = gpd.read_file(
        f'./data/block_2010_data/nhgis0005_shape/nhgis0005_shapefile_tl2010_{st_FIPS}_block_2010/{state}_block_2010.shp').set_index('GEOID10')
    city_rl_cw = pd.read_csv(f'./data/outputs/{city}_blocks_2010_crosswalk.csv', dtype={
                             'block_id_2010': str}).set_index('block_id_2010')
    city_blocks = city_blocks.join(
        city_rl_cw, how='outer').dropna().set_index('GISJOIN')
    print('City tract data loaded.')

    # join shapefile with race data
    city = city_blocks.join(race_raw, how='outer').dropna()

    # filter to create R
    R = city.groupby('holc_id_uq').sum().filter(['total', 'white', 'black', 'american_indian_al_native',
                                                 'asian', 'hawaiian_pac_islander', 'hawaiian_pac_islander', 'other', 'two_plus'])

    # find empty districts, if any exist
    empty_districts = np.array(R.loc[(R.total == 0)].index)

    # build race matrix
    race_matrix = city.filter(['total', 'white', 'black', 'american_indian_al_native',
                               'asian', 'hawaiian_pac_islander', 'other', 'two_plus', 'holc_id_uq', 'geometry'])
    race_matrix.rename(columns={'holc_id_uq': 'partition'}, inplace=True)
    # remove districts with population 0
    race_matrix = race_matrix[~race_matrix['partition'].isin(empty_districts)]
    return race_matrix


# entropy implementation
def city_entropy(R, P):
    """
    Computes entropy of a city-region (see White, 1986).

    Parameters
    ----------
    R : numpy.ndarray
        i-by-j matrix, where i=districts and j=ethnicities.
    P : numpy.array
        i-length vector of the total population in a city-region.

    Returns
    -------
    int
        citywide segregation entropy score.
    """
    # define key terms in algorithm
    N = sum(P)
    R_prop = np.apply_along_axis(lambda column: column / P, 0, R)
    r_hat = R.sum(axis=0) / N

    def entropy(x):
        """compute an entropy score with region ethnicity proportion vector x."""
        with np.errstate(divide='ignore'):
            vec = np.log(x)
        vec[np.isneginf(vec)] = 0  # fix special case where 0 * ln(0) = 0
        return (-1) * sum(x*vec)

    # compute district-level entropy scores
    h_i = list(map(entropy, R_prop))

    # compute city-wide entropy
    H_hat = entropy(r_hat)
    H_bar = sum((P/N) * h_i)

    return (H_hat - H_bar) / H_hat


def chain_to_entropy(chainobj, blocks=None):
    """
    Takes element in Markov Chain and computes entropy score using the partition map.
    Parameters
    ----------
    chainobj : dict
        Dict object that dictates which elements belong to which partitions.
    blocks : geopandas.GeoDataFrame
        Dataframe containing the map's census blocks and associated race information.

    Returns
    -------
    int
        entropy score using the partition data from Markov Chain element.
    """
    global race_matrix
    blocks = race_matrix
    # use partition parts and grouping to create R and P
    R = blocks.drop(columns=['geometry'])
    R['partition'] = blocks.index.map(chainobj)
    R = R.groupby('partition').sum()
    R, P = R.to_numpy()[:, 1:], R.to_numpy()[:, 0]

    return city_entropy(R, P)


def save_results(city_name, step_count, chunk_entropies=None, random_entropies=None):
    os.chdir("parallel")
    # sort and convert lists of entropies
    chunk_entropies.sort(), random_entropies.sort()
    chunk_entropies = list(map(itemgetter(1), chunk_entropies))
    random_entropies = list(map(itemgetter(1), random_entropies))

    # save entropy lists
    np.save(
        f"./results/arrays/{city_name.lower()}_cf_{step_count}.npy", chunk_entropies)
    np.save(
        f"./results/arrays/{city_name.lower()}_rf_{step_count}.npy", random_entropies)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.xlabel("City-wide Entropy Score")
    plt.ylabel("Density")
    sns.kdeplot(chunk_entropies)
    plt.scatter(chain_to_entropy(
        dict(init_partition.assignment), race_matrix), 0, c='r')

    plt.subplot(1, 2, 2)
    plt.xlabel("Step in Markov Chain")
    plt.ylabel("City-wide Entropy Score")
    plt.plot(chunk_entropies)
    plt.plot(np.repeat(chain_to_entropy(
        dict(init_partition.assignment), race_matrix), step_count), c='r')

    plt.suptitle(
        f"Chunk Flip Entropies for {city_name}, burn-in for {(0.1)*step_count} steps", y=1)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{city_name.lower()}_cf_{step_count}.png")

    # plot and save result plots
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.xlabel("City-wide Entropy Score")
    plt.ylabel("Density")
    sns.kdeplot(random_entropies)
    plt.scatter(chain_to_entropy(
        dict(init_partition.assignment), race_matrix), 0, c='r')

    plt.subplot(1, 2, 2)
    plt.xlabel("Step in Markov Chain")
    plt.ylabel("City-wide Entropy Score")
    plt.plot(random_entropies)
    plt.plot(np.repeat(chain_to_entropy(
        dict(init_partition.assignment), race_matrix), step_count), c='r')

    plt.suptitle(
        f"Random Flip Entropies for {city_name}, burn-in for {(0.1)*step_count} steps", y=1)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{city_name.lower()}_rf_{step_count}.png")
    # plt.show()


def generator(stop_event, partition_queue, _type, step_count=1000, burn_in=1000):
    """
    Parameters
    ----------
    _type : string
        either "chunk" or "random"
    step_count : int, Default 1000
        steps of chain to run (after burn-in)
    burn_in : int, Default 1000
        steps to burn-in for (not to collect data)
    stop_event: multiprocessing.Event
        tells chain's workers if generation has stopped
    partition_queue: multiprocessing.Queue
        structure that takes each partition as it is generated
    """
    from gerrychain.random import random
    random.seed(2020)
    init_partition = Partition(graph,
                               assignment=race_matrix.to_dict()['partition'],
                               updaters={'population': Tally('population')})
    if _type == "chunk":
        chain = MarkovChain(proposal=propose_chunk_flip,
                            constraints=is_valid,
                            accept=always_accept,
                            initial_state=init_partition,
                            total_steps=step_count + burn_in)
    elif _type == "random":
        chain = MarkovChain(proposal=propose_random_flip,
                            constraints=is_valid,
                            accept=always_accept,
                            initial_state=init_partition,
                            total_steps=step_count + burn_in)
    else:
        raise ValueError("Wrong chain type given. Give 'chunk' or 'random'.")
    iter(chain)
    if _type == "chunk":
        burn_bar = trange(burn_in, desc=f"{_type} flip Burn-in", leave=True, position=0)
        pbar = trange(
            step_count, desc=f"Generating {_type} flip", leave=True, position=0)
    else:
        burn_bar = trange(burn_in, desc=f"{_type} flip Burn-in", leave=True, position=1)
        pbar = trange(
            step_count, desc=f"Generating {_type} flip", leave=True, position=1)
    # burn-in
    for _ in burn_bar:
        next(chain)
    
    for i in pbar:
        partition_queue.put((i, dict(next(chain).assignment)))
    stop_event.set()
    # send a text when done
    client.messages.create(to='+15103785524', from_='+12059272645',
                           body=f"{_type.capitalize()} flip for {CITY_NAME} completed.")
    print(f"{_type.capitalize()} Generator: {stop_event.is_set()}")


def worker(stop_event, partition_queue, entropy_list, timeout=2):
    """
    Calculates entropy from available partitions in queue.

    Parameters
    ----------
    stop_event : multiprocessing.Event
        lets worker know when chain's generation has stopped
    partition_queue : multiprocessing.Manager.list
        queue that stores partitions
    entropy_list : multiprocessing.Manager.list
        list to dump results into
    timeout : float, Default 1
        how long, in seconds, to wait before checking if generation is done
    """
    while True:
        if stop_event.is_set() and partition_queue.empty():
            return
        else:
            try:
                partition = partition_queue.get(block=True, timeout=timeout)
                entropy_list.append(
                    (partition[0], chain_to_entropy(partition[1], race_matrix)))
            except:
                if stop_event.is_set():
                    return
                else:
                    continue
    return


def generate_entropies(_type, step_count, burn_in, results, processes=4):
    """
    Parameters
    ----------
    _type : string
        either "chunk" or "random"
    step_count : int
        how long to run the chain for
    burn_in : int
        how long to run the chain without collecting data for
    results : dict
        key = type, value = list of entropies
    processes : int, Default 4
        number of processors to use
    """
    manager = Manager()
    entropy_list = manager.list()
    # each item in queue is a tuple (step in chain, dict of assignment)
    partition_queue = Queue(1)
    stop_event = Event()

    # define workers and chain generator
    g = Process(target=generator, args=(
        stop_event, partition_queue, _type, step_count, burn_in))
    workers = [Process(target=worker, args=(
        stop_event, partition_queue, entropy_list)) for _ in range(processes)]

    begin = time.time()
    for p in workers:
        p.start()

    try:
        g.start()
        g.join()
        _ = [p.join() for p in workers]
    except KeyboardInterrupt:
        _ = [p.kill() for p in workers]
        g.kill()

    print(f"Elapsed: {round(time.time() - begin, 3)} s")
    print(partition_queue.empty())
    print(entropy_list[-5:], len(entropy_list))
    results[_type] = list(entropy_list)


if __name__ == '__main__':
    # twilio setup, requires proper env variables to be set up (so it will text you when the chain is done)
    account = os.environ['TWILIO_ACCT']
    auth = os.environ['TWILIO_AUTH']
    client = Client(account, auth)
    
    STEP_COUNT = 1000000
    BURN_IN = int(0.1 * STEP_COUNT)
    CITY_NAME = 'Brooklyn'
    STATE = 'NY'
    STATE_FIPS = "360"

    manager = Manager()
    results = manager.dict()

    race_matrix = load_data(CITY_NAME, STATE, STATE_FIPS)

    # build chain
    graph = Graph.from_geodataframe(race_matrix, adjacency='queen')
    nx.set_node_attributes(
        graph, race_matrix['total'].to_dict(), name='population')
    init_partition = Partition(graph,
                               assignment=race_matrix.to_dict()['partition'],
                               updaters={'population': Tally('population')})

    # validators
    def mean_pop(part): return np.mean(list(part['population'].values()))
    def min_pop(part): return min(list(part['population'].values()))
    def sd_pop(part): return np.std(list(part['population'].values()))
    mean_one_sd_up = mean_pop(init_partition) + (2/3)*sd_pop(init_partition)
    mean_one_sd_down = mean_pop(init_partition) - (2/3)*sd_pop(init_partition)

    # initalize and run both chains
    is_valid = Validator([LowerBound(min_pop, min_pop(init_partition) % 50),
                          UpperBound(mean_pop, mean_one_sd_up),
                          LowerBound(mean_pop, mean_one_sd_down),
                          WithinPercentRangeOfBounds(sd_pop, 25)])

    chunk = Process(target=generate_entropies, args=(
        "chunk", STEP_COUNT, BURN_IN, results))
    random = Process(target=generate_entropies, args=(
        "random", STEP_COUNT, BURN_IN, results))
    chunk.start(), random.start()
    chunk.join(), random.join()
    save_results(CITY_NAME, STEP_COUNT, results['chunk'], results['random'])
