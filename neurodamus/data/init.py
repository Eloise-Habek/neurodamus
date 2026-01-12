"""Neurodamus is a software for handling neuronal simulation using neuron.

Copyright (c) 2018 Blue Brain Project, EPFL.
All rights reserved
"""

import logging
import sys

from neuron import h

from neurodamus import commands
from neurodamus.utils.cli import extract_arguments
from neurodamus.utils.visualization import *

def main():
    args = []
    try:
        args = extract_arguments(sys.argv)
    except ValueError:
        logging.exception()
        return 1
    
    exit_code = commands.neurodamus(args)

    config_path = args[0]
    try:
        #We can also add these as command arguments later if we want to
        bin = 5
        simulationData, allData, cellData, spikeData = load_data(config_path)
        cell_to_rank = geom_rank_cell_nodes(cellData, simulationData)
        pop_size = cellData.shape[0]
        psth_plot(spikeData, f"", cell_to_rank, simulationData,population_size=pop_size, bin_width=bin, save_histogram=f"reporting/{bin}bin_smoothed.png", smoothed=True)

    except Exception as e:
        print("The data visualization failed.")
        # print(e)

    return exit_code


if __name__ == "__main__":
    # Returns exit code and calls MPI.Finalize
    h.quit(main())
