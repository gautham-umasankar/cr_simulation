# Cross Resonance Gate Simulation

This package has tools to numerically simulate Cross Resonance (CR) Gates in a superconducting qubit processor with fixed-frequency transmon qubits. It has tools to:

1. Characterize an arbitrarily-coupled transmon processor - dressed frequencies, ZZ shifts, proximity to poles in the CR frequency landscape
2. Estimate the effective Hamiltonian of a two-qubit system
3. Numerically calibrate single qubit gates and CR gates between arbitrary pairs of qubits
4. Estimate the Fidelity and Error Budget of the single qubit and CR gates - Errors due to imperfect rotation, leakage, spectator rotation etc (Note that decoherence is not taken into account)

All these capabilities are illustrated in the Example_CR_Simulation.ipynb notebook.

Instructions to use:

1. Clone this whole repository.
2. Create a new conda environment (Not necessary, but recommended, some package versions might be changed)
3. In the directory containing this repository, run ``` pip install -e cr_simulation ```
4. Use the example notebook to understand use all features
