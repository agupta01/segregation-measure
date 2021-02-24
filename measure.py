# FOR REPRODUCIBILITY
from gerrychain.random import random

random.seed(2020)

import os
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
)
from gerrychain.proposals import propose_chunk_flip, propose_random_flip, recom

from gerrychain.updaters import Tally
from tqdm import tqdm

# twilio setup, requires proper env variables to be set up (so it will text you when the chain is done)
account = os.environ["TWILIO_ACCT"]
auth = os.environ["TWILIO_AUTH"]
client = Client(account, auth)


# Load race data
FILEPATH = "../data/block_2010_data/nhgis0005_csv"
relevant_cols = [
    "GISJOIN",
    "STATEA",
    "COUNTYA",
    "H7X001",
    "H7X002",
    "H7X003",
    "H7X004",
    "H7X005",
    "H7X006",
    "H7X007",
    "H7X008",
]
race_raw = pd.read_csv(
    f"{FILEPATH}/nhgis0005_ds172_2010_block.csv",
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
    "../data/block_2010_data/nhgis0005_shape/nhgis0005_shapefile_tl2010_130_block_2010/GA_block_2010.shp"
).set_index("GEOID10")
city_rl_cw = pd.read_csv(
    "../data/outputs/Atlanta_blocks_2010_crosswalk.csv", dtype={"block_id_2010": str}
).set_index("block_id_2010")
city_blocks = city_blocks.join(city_rl_cw, how="outer").dropna().set_index("GISJOIN")
city_blocks.plot(figsize=(10, 10))
print("City tract data loaded.")

# join shapefile with race data
city = city_blocks.join(race_raw, how="outer").dropna()


# FOR NOTEBOOK USE: plot race data for verification
# import contextily as ctx
# def plot_map(df, column, figsize=(10,10), cmap='plasma'):
#     ax = df.to_crs(epsg=3857).plot(figsize=figsize, cmap=cmap, column=column)
#     ax.set_axis_off()
#     ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
#
# plot_map(city, 'black')


# filter to create R
R = (
    city.groupby("holc_id_uq")
    .sum()
    .filter(
        [
            "total",
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


# build chain
graph = Graph.from_geodataframe(race_matrix, adjacency="queen")
nx.set_node_attributes(graph, race_matrix["total"].to_dict(), name="population")
init_partition = Partition(
    graph,
    assignment=race_matrix.to_dict()["partition"],
    updaters={"population": Tally("population")},
)
# np.percentile(list(init_partition['population'].values()), [2.5, 97.5])
# {key:value for (key, value) in init_partition['population'].items() if value == 0}
# this should output an empty dictionary. if not, do not run chain, remove these districts first


# validators
mean_pop = lambda part: np.mean(list(part["population"].values()))
min_pop = lambda part: min(list(part["population"].values()))
sd_pop = lambda part: np.std(list(part["population"].values()))
mean_one_sd_up = mean_pop(init_partition) + (2 / 3) * sd_pop(init_partition)
mean_one_sd_down = mean_pop(init_partition) - (2 / 3) * sd_pop(init_partition)
mean_pop(init_partition), mean_one_sd_up, mean_one_sd_down, sd_pop(init_partition)


# initalize and run both chains
is_valid = Validator(
    [
        LowerBound(min_pop, min_pop(init_partition) % 50),
        UpperBound(mean_pop, mean_one_sd_up),
        LowerBound(mean_pop, mean_one_sd_down),
        WithinPercentRangeOfBounds(sd_pop, 25),
    ]
)
BURN_IN = 1000
STEP_COUNT = 1000000
CITY_NAME = "Atlanta"
chunk_chain = MarkovChain(
    proposal=propose_chunk_flip,
    constraints=is_valid,
    accept=always_accept,
    initial_state=init_partition,
    total_steps=STEP_COUNT + BURN_IN,
)
random_chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=is_valid,
    accept=always_accept,
    initial_state=init_partition,
    total_steps=STEP_COUNT + BURN_IN,
)
print("Prereqs created, Chunk flip chain running...")
# burn-in of 1000
iter(chunk_chain)
print("Burn-in:")
_ = [next(chunk_chain) for i in tqdm(range(BURN_IN))]
chunk_entropies = [
    chain_to_entropy(next(chunk_chain), race_matrix) for i in tqdm(range(STEP_COUNT))
]
del chunk_chain
np.save(f"results/arrays/{CITY_NAME.lower()}_cf_{STEP_COUNT}.npy", chunk_entropies)
# notify chain is done
client.messages.create(
    to="+15103785524",
    from_="+12059272645",
    body=f"Chunk flip for {CITY_NAME} completed.",
)

print("Random flip chain running...")
# burn-in of 1000
iter(random_chain)
print("Burn-in:")
_ = [next(random_chain) for i in tqdm(range(BURN_IN))]
random_entropies = [
    chain_to_entropy(next(random_chain), race_matrix) for i in tqdm(range(STEP_COUNT))
]
del random_chain
np.save(f"results/arrays/{CITY_NAME.lower()}_rf_{STEP_COUNT}.npy", random_entropies)
# notify chain is done
client.messages.create(
    to="+15103785524",
    from_="+12059272645",
    body=f"Random flip for {CITY_NAME} completed.",
)


# %%
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.xlabel("City-wide Entropy Score")
plt.ylabel("Density")
sns.kdeplot(chunk_entropies)
plt.scatter(chain_to_entropy(init_partition, race_matrix), 0, c="r")

plt.subplot(1, 2, 2)
plt.xlabel("Step in Markov Chain")
plt.ylabel("City-wide Entropy Score")
plt.plot(chunk_entropies)
plt.plot(np.repeat(chain_to_entropy(init_partition, race_matrix), STEP_COUNT), c="r")

plt.suptitle(
    f"Chunk Flip Entropies for {CITY_NAME}, run for {STEP_COUNT} steps", y=1.02
)
plt.tight_layout()
plt.savefig(f"results/plots/{CITY_NAME.lower()}_cf_{STEP_COUNT}.png")


# plot and save result plots
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.xlabel("City-wide Entropy Score")
plt.ylabel("Density")
sns.kdeplot(random_entropies)
plt.scatter(chain_to_entropy(init_partition, race_matrix), 0, c="r")

plt.subplot(1, 2, 2)
plt.xlabel("Step in Markov Chain")
plt.ylabel("City-wide Entropy Score")
plt.plot(random_entropies)
plt.plot(np.repeat(chain_to_entropy(init_partition, race_matrix), STEP_COUNT), c="r")

plt.suptitle(
    f"Random Flip Entropies for {CITY_NAME}, run for {STEP_COUNT} steps", y=1.02
)
plt.tight_layout()
plt.savefig(f"results/plots/{CITY_NAME.lower()}_rf_{STEP_COUNT}.png")
