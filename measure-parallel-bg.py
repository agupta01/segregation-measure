import argparse
import os
import time
from multiprocessing import Event, Manager, Process, Queue
from operator import itemgetter
import hashlib

import geopandas as gpd
from gerrychain.constraints import contiguous, no_vanishing_districts
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gerrychain import Graph, MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.constraints import (
    LowerBound,
    UpperBound,
    Validator,
    WithinPercentRangeOfBounds,
)
from gerrychain.proposals import propose_chunk_flip
from gerrychain.random import random  # FOR REPRODUCIBILITY
from gerrychain.updaters import Tally
from tqdm import trange
from twilio.rest import Client


def load_data(city, state, st_FIPS):
    """
    Loads census data and HOLC representations from disk.

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
    # Load race data
    FILEPATH = "./data/nhgis0010_csv/nhgis0011_csv"
    relevant_cols = [
        "GISJOIN",
        "STATEA",
        "COUNTYA",
        "U7C001",
        "U7C002",
        "U7C005",
        "U7C006",
        "U7C007",
        "U7C008",
        "U7C009",
        "U7C010",
        "U7C011",
    ]
    race_raw = pd.read_csv(
        f"{FILEPATH}/nhgis0011_ds248_2020_blck_grp.csv",
        usecols=relevant_cols,
        dtype={"GISJOIN": str, "STATEA": str, "COUNTYA": str},
    )
    column_mapper = dict(
        zip(
            relevant_cols,
            [
                "GISJOIN",
                "state_fips",
                "county_fips",
                "total",
                "hispanic",
                "white",
                "black",
                "american_indian_al_native",
                "asian",
                "hawaiian_pac_islander",
                "other",
                "two_plus",
            ],
        )
    )

    race_raw.rename(columns=column_mapper, inplace=True)
    print("Race data loaded.")
    race_raw.set_index("GISJOIN", inplace=True)

    # Load relevant shapefile and crosswalks
    city_blocks = gpd.read_file(
        f"./data/nhgis0009_shape/{state}_blck_grp_2020.shp"
    ).set_index("GEOID")
    city_rl_cw = pd.read_csv(
        f"./data/2020_outputs/{city}_bg_2020_crosswalk.csv", dtype={"bg_id_2020": str}
    ).set_index("bg_id_2020")
    city_blocks = (
        city_blocks.join(city_rl_cw, how="outer").dropna().set_index("GISJOIN")
    )

    print("City block group data loaded.")

    # join shapefile with race data
    city = city_blocks.join(race_raw, how="outer").dropna()

    # filter to create R
    R = (
        city.groupby("holc_id_uq")
        .sum()
        .filter(
            [
                "total",
                "hispanic",
                "white",
                "black",
                "american_indian_al_native",
                "asian",
                "hawaiian_pac_islander",
                "hawaiian_pac_islander",
                "other",
                "two_plus",
            ]
        )
    )

    # find empty districts, if any exist
    empty_districts = np.array(R.loc[(R.total == 0)].index)

    # build race matrix
    race_matrix = city.filter(
        [
            "total",
            "hispanic",
            "white",
            "black",
            "american_indian_al_native",
            "asian",
            "hawaiian_pac_islander",
            "other",
            "two_plus",
            "holc_id_uq",
            "geometry",
        ]
    )
    race_matrix.rename(columns={"holc_id_uq": "partition"}, inplace=True)
    # remove districts with population 0
    race_matrix = race_matrix[~race_matrix["partition"].isin(empty_districts)]
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
        with np.errstate(divide="ignore"):
            vec = np.log(x)
        vec[np.isneginf(vec)] = 0  # fix special case where 0 * ln(0) = 0
        return (-1) * sum(x * vec)

    # compute district-level entropy scores
    h_i = list(map(entropy, R_prop))

    # compute city-wide entropy
    H_hat = entropy(r_hat)
    H_bar = sum((P / N) * h_i)

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
    R = blocks.drop(columns=["geometry"])
    R["partition"] = blocks.index.map(chainobj)
    R = R.groupby("partition").sum()
    R, P = R.to_numpy()[:, 1:], R.to_numpy()[:, 0]

    return city_entropy(R, P)


def save_results(city_name, final_step_count, chain_id, entropies=None):
    """Plots and saves results in graphical and array formats."""

    # sort and convert lists of entropies
    entropies.sort()
    entropies = list(map(itemgetter(1), entropies))

    # save entropy lists
    np.save(
        f"./results_2020/arrays/{city_name.lower()}_cf_{final_step_count}_{chain_id}.npy",
        entropies,
    )

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.xlabel("City-wide Entropy Score")
    plt.ylabel("Density")
    sns.kdeplot(entropies)
    plt.scatter(
        chain_to_entropy(dict(init_partition.assignment), race_matrix), 0, c="r"
    )

    plt.subplot(1, 2, 2)
    plt.xlabel("Step in Markov Chain")
    plt.ylabel("City-wide Entropy Score")
    plt.plot(entropies)
    plt.plot(
        np.repeat(
            chain_to_entropy(dict(init_partition.assignment), race_matrix),
            final_step_count,
        ),
        c="r",
    )

    plt.suptitle(
        f"Entropies for {city_name}, burn-in for {(0.1)*final_step_count} steps",
        y=1,
    )
    plt.tight_layout()
    plt.savefig(
        f"./results_2020/plots/{city_name.lower()}_cf_{final_step_count}_{chain_id}.png"
    )

    # plt.show()


def generator(
    stop_event,
    partition_queue,
    step_count=1000,
    burn_in_ratio=0.1,
    thinning=5,
    seed=2020,
):
    """
    Creates and runs generator of map proposals

    Parameters
    ----------
    step_count : int, Default 1000
        steps of chain to run (after burn-in)
    burn_in_ratio : float, Default 0.1
        steps to burn-in for, as a ratio of the total step count (not to collect data)
    stop_event: multiprocessing.Event
        tells chain's workers if generation has stopped
    partition_queue: multiprocessing.Queue
        structure that takes each partition as it is generated
    thinning: int, Default 5
        Take every <thinning>th result from the chain to minimize dependence
    seed: int, Default 2020
        Random seed for reproducibility
    """
    # FOR REPRODUCIBILITY
    from gerrychain.random import random

    random.seed(seed)

    init_partition = Partition(
        graph,
        assignment=race_matrix.to_dict()["partition"],
        updaters={"population": Tally("population")},
    )

    chain = MarkovChain(
        proposal=propose_chunk_flip,
        constraints=is_valid,
        accept=always_accept,
        initial_state=init_partition,
        total_steps=step_count + burn_in_ratio * step_count,
    )

    iter(chain)

    burn_bar = trange(
        int(burn_in_ratio * step_count), desc=f"Burn-in", leave=True, position=0
    )
    pbar = trange(
        int(burn_in_ratio * step_count) + step_count,
        desc=f"Generating maps",
        leave=True,
        position=0,
    )

    # burn-in
    # for _ in burn_bar:
    #     next(chain)

    for i in pbar:
        map_proposal = (i, dict(next(chain).assignment))
        # only push proposal to queue if it is <thinning>th proposal
        if i % thinning == 0:
            partition_queue.put(map_proposal)
    stop_event.set()
    # # send a text when done (SEE FIELDS)
    # client.messages.create(
    #     to=<YOUR PHONE NUMBER>,
    #     from_=<TWILIO SOUCE NUMBER>,
    #     body=f"{_type.capitalize()} flip for {CITY_NAME} completed.",
    # )
    print(f"{CITY_NAME} Generator: {stop_event.is_set()}")


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
    timeout : float, Default 2
        how long, in seconds, to wait before checking if generation is done
    """
    while True:
        if stop_event.is_set() and partition_queue.empty():
            return
        else:
            try:
                partition = partition_queue.get(block=True, timeout=timeout)
                entropy_list.append(
                    (partition[0], chain_to_entropy(partition[1], race_matrix))
                )
            except:
                if stop_event.is_set():
                    return
                else:
                    continue


# TODO: write this function!
def partition_polsby_popper(partition):
    """Checks if partition is within polsby-popper metric

    Args:
        partition (gerrychain partition): partition map from a single step in the Markov Chain

    Returns:
        function that takes partition and checks if it's within the bounds
    """
    # get all shapes from each district
    # compute polsby-popper on all districts, get min
    return 0.5


def generate_entropies(chain_id, step_count, burn_in_ratio, results, processes=5):
    """
    Generates entropies for given proposal type in a parallel manner.

    Parameters
    ----------
    chain_id : string
        chain id
    step_count : int
        how long to run the chain for
    burn_in_ratio : float
        how long to run the chain without collecting data for, as a ratio of total step count
    results : dict
        key = chain id, value = list of entropies
    processes : int, Default 5
        number of processors to use
    """
    manager = Manager()
    entropy_list = manager.list()
    # each item in queue is a tuple (step in chain, dict of assignment)
    partition_queue = Queue(1)
    stop_event = Event()

    # define workers and chain generator
    g = Process(
        target=generator,
        args=(stop_event, partition_queue, step_count, burn_in_ratio, 2022),
    )
    workers = [
        Process(target=worker, args=(stop_event, partition_queue, entropy_list))
        for _ in range(processes)
    ]

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
    results[chain_id] = list(entropy_list)


if __name__ == "__main__":
    # # twilio setup, requires proper env variables to be set up (so it will text you when the chain is done)
    # account = os.environ["TWILIO_ACCT"]
    # auth = os.environ["TWILIO_AUTH"]
    # client = Client(account, auth)

    # get hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        help="number of steps for each markov chain",
        default=100000,
    )
    parser.add_argument("city", type=str, help="city name, i.e. Atlanta")
    parser.add_argument("state", type=str, help="state code, i.e. GA")
    parser.add_argument(
        "fips", help="state FIPS code (zero-padded on the end), i.e. 130"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="total # of worker processes across both proposals",
        default=10,
    )
    args = parser.parse_args()

    STEP_COUNT = args.steps
    BURN_IN_RATIO = 0.1
    CITY_NAME = args.city
    STATE = args.state
    STATE_FIPS = str(args.fips)
    TOT_WORKERS = args.workers

    manager = Manager()
    results = manager.dict()

    race_matrix = load_data(CITY_NAME, STATE, STATE_FIPS)

    # build chain
    graph = Graph.from_geodataframe(race_matrix, adjacency="queen")
    nx.set_node_attributes(graph, race_matrix["total"].to_dict(), name="population")
    init_partition = Partition(
        graph,
        assignment=race_matrix.to_dict()["partition"],
        updaters={"population": Tally("population")},
    )

    # validators
    def mean_pop(part):
        return np.mean(list(part["population"].values()))

    def min_pop(part):
        return min(list(part["population"].values()))

    def sd_pop(part):
        return np.std(list(part["population"].values()))

    mean_one_sd_up = mean_pop(init_partition) + (2 / 3) * sd_pop(init_partition)
    mean_one_sd_down = mean_pop(init_partition) - (2 / 3) * sd_pop(init_partition)

    # initalize and run chains
    # TODO: record descent
    is_valid = Validator(
        [
            LowerBound(min_pop, min_pop(init_partition) % 50),
            UpperBound(mean_pop, mean_one_sd_up),
            LowerBound(mean_pop, mean_one_sd_down),
            WithinPercentRangeOfBounds(sd_pop, 25),
            # contiguous,
            LowerBound(
                partition_polsby_popper, bound=partition_polsby_popper(init_partition)
            ),
            no_vanishing_districts,
        ]
    )

    # make sure init_partition passes validators
    assert is_valid(init_partition)

    chain_one = Process(
        target=generate_entropies,
        args=("chain_two", STEP_COUNT, BURN_IN_RATIO, results, 2),
    )
    # chain_two = Process(
    #     target=generate_entropies,
    #     args=("chain_two", STEP_COUNT, BURN_IN_RATIO, results, 2),
    # )
    # chain_three = Process(
    #     target=generate_entropies,
    #     args=("chain_three", STEP_COUNT, BURN_IN_RATIO, results, 2),
    # )
    # chain_four = Process(
    #     target=generate_entropies,
    #     args=("chain_four", STEP_COUNT, BURN_IN_RATIO, results, 2),
    # )
    # chain_five = Process(
    #     target=generate_entropies,
    #     args=("chain_five", STEP_COUNT, BURN_IN_RATIO, results, 2),
    # )

    # start all chains
    chain_one.start()
    chain_one.join()
    # chain_two.start()
    # chain_two.join()
    # chain_three.start()
    # chain_three.join()
    # chain_four.start()
    # chain_four.join()
    # chain_five.start()
    # chain_five.join()

    # save results
    save_results(CITY_NAME, STEP_COUNT, "chain2", results["chain_two"])
    # save_results(CITY_NAME, STEP_COUNT, "chain2", results["chain_two"])
    # save_results(CITY_NAME, STEP_COUNT, "chain3", results["chain_three"])
    # save_results(CITY_NAME, STEP_COUNT, "chain4", results["chain_four"])
    # save_results(CITY_NAME, STEP_COUNT, "chain5", results["chain_five"])
