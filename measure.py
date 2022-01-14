# FOR REPRODUCIBILITY
import time
from gerrychain.random import random

random.seed(2020)

import os
import argparse
from twilio.rest import Client
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gerrychain import (
    GeographicPartition,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    proposals,
    updaters,
)
from gerrychain.accept import always_accept
from gerrychain.constraints import (
    LowerBound,
    UpperBound,
    Validator,
    WithinPercentRangeOfBounds,
    no_vanishing_districts,
)
from gerrychain.proposals import propose_chunk_flip, propose_random_flip, recom

from gerrychain.updaters import Tally
from tqdm import tqdm
from joblib import Parallel, delayed


def load_data(city, state, st_FIPS, fake=False):
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
    fake : boolean, Default False
        if True, replaces real data with random matrix. Only for testing purposes.

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

    if fake:
        # 0th column is total, last 2 columns are holc_id and geometry
        R_base = race_matrix.iloc[:, 1:-2].to_numpy()
        R = np.random.choice(R_base.flatten(), size=R_base.shape)
        race_matrix.iloc[:, 1:-2] = R
        race_matrix["total"] = R.sum(axis=1)
        del R_base

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


def chain_to_entropy(chainobj, blocks):
    """
    Takes element in Markov Chain and computes entropy score using the partition map.
    Parameters
    ----------
    chainobj : gerrychain.partition.partition.Partition
        Partition object that dictates which elements belong to which partitions.
    blocks : geopandas.GeoDataFrame
        Dataframe containing the map's census blocks and associated race information.

    Returns
    -------
    int
        entropy score using the partition data from Markov Chain element.
    """
    # use partition parts and grouping to create R and P
    R = blocks.drop(columns=["geometry"]).copy()
    R["partition"] = blocks.index.map(dict(chainobj.assignment))
    R = R.groupby("partition").sum()
    R, P = R.to_numpy()[:, 1:], R.to_numpy()[:, 0]

    return city_entropy(R, P)


def save_results(city_name, final_step_count, chain_id, baseline, entropies=None):
    """Plots and saves results in graphical and array formats."""

    # sort and convert lists of entropies - not necessary for sequential processing
    # entropies.sort()
    # entropies = list(map(itemgetter(1), entropies))

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
    plt.scatter(baseline, 0, c="r")

    plt.subplot(1, 2, 2)
    plt.xlabel("Step in Markov Chain")
    plt.ylabel("City-wide Entropy Score")
    plt.plot(entropies)
    plt.plot(
        np.repeat(
            baseline,
            len(entropies),
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


# np.percentile(list(init_partition['population'].values()), [2.5, 97.5])
# {key:value for (key, value) in init_partition['population'].items() if value == 0}
# this should output an empty dictionary. if not, do not run chain, remove these districts first


def run_full_chain(chain_name):
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
    args = parser.parse_args()

    STEP_COUNT = args.steps
    BURN_IN_RATIO = 0.1
    CITY_NAME = args.city
    STATE = args.state
    STATE_FIPS = str(args.fips)
    THINNING_FACTOR = 5  # measure entropy only once every these many iterations of MC

    race_matrix = load_data(CITY_NAME, STATE, STATE_FIPS, fake=False)
    R_scratch = race_matrix[
        ["partition", "geometry"]
    ]  # scratch version of R for the polsby-popper computation

    print(race_matrix.head())

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
        return np.min(list(part["population"].values()))

    def sd_pop(part):
        return np.std(list(part["population"].values()))

    # TODO: only check if GISJOIN in minimum P-P partition have changed
    # TODO: cache set of GISJOINs for minimum partition for lowest P-P partition
    # TODO: compare this set to the new one when given a partition
    # TODO: if set is different, recompute P-P for whole partition, else do nothing
    def partition_polsby_popper(part, R=R_scratch):
        """Checks if partition is within polsby-popper metric

        Args:
            partition (gerrychain partition): partition map from a single step in the Markov Chain
            R (geopandas.GeoDataFrame): columns 'partition' and 'geometry' for getting the polygons

        Returns:
            function that takes partition and checks if it's within the bounds
        """
        # get all shapes from each district
        # compute polsby-popper on all districts, get min
        pd.options.mode.chained_assignment = None
        R.loc[:, "partition"] = race_matrix.index.map(dict(part.assignment))
        R_temp = R.copy(deep=True).dissolve(by="partition")
        polsby_popper = lambda d: (4 * np.pi * d.area) / (
            d.length ** 2
        )  # d is a polygon
        # srs = R["geometry"].map(polsby_popper).values
        # print(np.min(srs), np.mean(srs), np.max(srs))
        # return srs.min()
        return R_temp["geometry"].map(polsby_popper).min()
        # return min(polsby_popper_from_R(R).values())

    def polsby_popper_from_R(R):
        """A more stable version of geopandas dissolve."""
        from shapely.ops import unary_union

        # loop through all partitons and unary join them, the return a dict indexed by partition id
        result = {}
        polsby_popper = lambda d: (4 * np.pi * d.area) / (
            d.length ** 2
        )  # d is a polygon
        for pid in R["partition"].unique():
            # get all geometries
            geom = R.loc[R["partition"] == pid]["geometry"].values
            result[pid] = polsby_popper(unary_union(geom))
        return result

    def partition_polsby_popper_min(
        part,
        R=R_scratch,
    ):
        nonlocal min_partition_id
        nonlocal min_partition_gisjoins
        nonlocal min_partition_p_p
        pd.options.mode.chained_assignment = None
        R.loc[:, "partition"] = race_matrix.index.map(dict(part.assignment))
        same_gisjoins = (
            set(R.loc[R["partition"] == min_partition_id].index.values)
            == min_partition_gisjoins
        )
        if min_partition_id is not None and same_gisjoins:
            # no change, return the old one
            return min_partition_p_p
        else:
            # something changed, so recompute all partitions
            # R_temp = R.copy(deep=True).dissolve(by="partition")
            # p_p_scores = R_temp["geometry"].map(polsby_popper)
            # min_partition_p_p = p_p_scores.min()
            # min_partition_id = R_temp.iloc[np.argmin(p_p_scores.values)].name
            p_p_scores = polsby_popper_from_R(R)
            min_partition_p_p = min(p_p_scores.values())
            min_partition_id = min(p_p_scores.items(), key=lambda x: x[1])[0]
            min_partition_gisjoins = set(
                R.loc[R["partition"] == min_partition_id].index.values
            )
            if (
                min_partition_p_p < 0.147
            ):  # initial oakland partition has min score of 0.147
                print("Rejected with score", min_partition_p_p)
            return min_partition_p_p

    mean_one_sd_up = mean_pop(init_partition) + (2 / 3) * sd_pop(init_partition)
    mean_one_sd_down = mean_pop(init_partition) - (2 / 3) * sd_pop(init_partition)

    min_partition_id, min_partition_gisjoins, min_partition_p_p = None, set(), None

    # initalize and run chains
    # TODO: record descent
    is_valid = Validator(
        [
            LowerBound(min_pop, min_pop(init_partition) % 50),
            UpperBound(mean_pop, mean_one_sd_up),
            LowerBound(mean_pop, mean_one_sd_down),
            WithinPercentRangeOfBounds(sd_pop, 25),
            # contiguous,
            # LowerBound(
            #     partition_polsby_popper, bound=partition_polsby_popper(init_partition)
            # ),
            # LowerBound(
            #     partition_polsby_popper_min,
            #     bound=partition_polsby_popper_min(init_partition),
            # ),
            no_vanishing_districts,
        ]
    )

    # make sure init_partition passes validators
    assert is_valid(init_partition)

    chain = MarkovChain(
        proposal=propose_chunk_flip,
        constraints=is_valid,
        accept=always_accept,
        initial_state=init_partition,
        total_steps=(STEP_COUNT * THINNING_FACTOR) + int(STEP_COUNT * BURN_IN_RATIO),
    )
    print(f"Prereqs created, {chain_name} running...")
    # burn-in of 1000
    iter(chain)
    # print(f"Burn-in: ({int(STEP_COUNT * BURN_IN_RATIO)} steps)")
    for i in range(int(STEP_COUNT * BURN_IN_RATIO)):
        if i % 100 == 0:
            print(f"{chain_name} BURN IN => {i}/{int(STEP_COUNT * BURN_IN_RATIO)}")
        next(chain)
    # print(f"Measurement: ({STEP_COUNT} steps)")
    entropies = []
    scores = []
    start_time = time.time()

    for i in range(STEP_COUNT * THINNING_FACTOR):
        if i % 25 == 0:
            print(
                f"{chain_name} ELAPSED {round(time.time() - start_time, 1)}s => {len(entropies)}/{STEP_COUNT}"
            )
        if i % THINNING_FACTOR == 0:
            part = next(chain)
            entropies.append(chain_to_entropy(part, race_matrix))
            scores.append(partition_polsby_popper_min(part))
        else:
            next(chain)

    np.save("./results_2020/polsby_popper_oakland.npy", np.array(scores))

    save_results(
        CITY_NAME,
        STEP_COUNT,
        chain_name,
        baseline=chain_to_entropy(init_partition, race_matrix),
        entropies=entropies,
    )

    # notify when done
    # client.messages.create(
    #     to="+15103785524",
    #     from_="+12059272645",
    #     body=f"Random flip for {CITY_NAME} completed.",
    # )


if __name__ == "__main__":
    NUM_CHAINS = 1
    Parallel(n_jobs=NUM_CHAINS, verbose=11)(
        delayed(run_full_chain)(f"chain{i+1}") for i in range(NUM_CHAINS)
    )
