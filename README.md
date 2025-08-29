# PM-and-EB-QKD-Simulations

Simulation code for Quantum Key Distribution experiments used in this dissertation. The two protocols that have been simulated are:

- BB84 (Prepare and Measure)
- E91 (Entanglement Based)


This workspace contains:

- Prepare and Measure simulations, topology sweeps and visualisation in [Prepare and Measure Simulations](Prepare and Measure Simulations)

- Entanglement Based E91 experiments and plots in [Entanglement Based Simulations](Entanglement Based Simulations)

- Figures and Results are saved in [PM QKD Results](Prepare and Measure Simulations/PM QKD Results) and [EB QKD Results](EB QKD Results)

## Dependencies

- Python 3.10+
- pip 22+
- NetSquid

Python packages used for simulations and analysis:

- numpy
- scipy
- pandas
- matplotlib
- networkx

## How to run

- Create a virtual environment and install the dependencies:
    - To install NetSquid, follow the installation instructions found on the [NetSquid website](https://netsquid.org/)
    - Output: figures in Prepare and Measure Simulations/PM QKD Results

- Prepare and Measure (BB84):
    - python "Prepare and Measure Simulations/main.py"
    - python "Entanglement Based Simulations/E91 Simulations.py"
    - Output: figures in EB QKD Results

