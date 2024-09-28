# Optimizing Cellular Frequency Assignment with Simulated Annealing, Memetic Algorithm, and Civilization Simulation

## Overview
This repository presents our solutions to the Fixed Channel Assignment Problem (FCAP), which involves allocating frequency channels to transmitters in cellular networks while minimizing interference. We applied three approaches to solve this problem: **Simulated Annealing (SA)**, **Memetic Algorithm (MA)**, and our unique **Civilization Simulation (CS)** technique. 

### Contributors:
- MohammadReza Javaheri 
- Maryam Hosseinali 

### Course:
Bio-Computing, University of [Tehran]

## Problem Definition
The **Frequency Assignment Problem** is a well-known optimization challenge in cellular networks, where the goal is to allocate frequencies in a way that minimizes interference and maximizes system performance. Our project explores this problem in depth using three different optimization techniques: Simulated Annealing, Memetic Algorithm, and Civilization Simulation (a novel approach combining elements of SA and Genetic Algorithms).

## Approaches and Methodology

### 1. Simulated Annealing (SA)
Simulated Annealing is a probabilistic optimization technique inspired by the annealing process in metallurgy. In our implementation:
- We initialize a random channel assignment and iteratively explore better solutions by decreasing the temperature.
- As the temperature decreases, the algorithm moves towards convergence by refining the solution and minimizing interference.

### 2. Memetic Algorithm (MA)
Our Memetic Algorithm is a hybrid of Genetic Algorithms (GA) and Local Search (LS):
- The GA component explores the solution space by generating new populations through crossover and mutation.
- The LS component (Tabu Search) refines each solution, reducing interference and improving fitness.
- We enhanced the MA by simplifying the crossover process, avoiding issues noted in existing literature.

### 3. Civilization Simulation (CS)
Civilization Simulation is a new approach we developed that combines Simulated Annealing and Genetic Algorithms:
- It simulates generations of a society where genes age and evolve.
- Younger genes explore more diverse solutions, while older genes exploit their experience to refine the solution.
- Over time, genes that do not improve are removed, mimicking a natural selection process.

## Problem Specifications
We tested our algorithms on eight problem instances with varying numbers of radio cells, channels, and compatibility matrices. The details of each problem are as follows:

| Problem # | Number of Cells (n) | Number of Channels (m) | Compatibility Matrix (C) | Demand Vector (D) |
|-----------|---------------------|------------------------|--------------------------|-------------------|
| 1         | 4                   | 11                     | C1                       | D1                |
| 2         | 25                  | 73                     | C2                       | D2                |
| 3         | 21                  | 381                    | C3                       | D3                |
| 4         | 21                  | 533                    | C4                       | D3                |
| 5         | 21                  | 533                    | C5                       | D3                |
| 6         | 21                  | 221                    | C3                       | D4                |
| 7         | 21                  | 309                    | C4                       | D4                |
| 8         | 21                  | 309                    | C5                       | D4                |

## Code Structure

### Problem Definition and Data Handling
The **Problem** class represents an instance of the FCAP, with attributes for the number of cells, number of required channels, compatibility matrix, and demand vector. We load the problem specifications from the `config.json` file, which contains all necessary data.

### Channel Assignment Algorithms
- **Simulated Annealing (SA)**: Iteratively optimizes a solution by probabilistically accepting better or worse solutions based on the temperature.
- **Memetic Algorithm (MA)**: Combines global exploration (GA) with local refinement (Tabu Search).
- **Civilization Simulation (CS)**: Simulates societal evolution, combining the exploration power of SA with the generational learning of GA.

## Results

We evaluated our algorithms on eight problem instances. Below are the results:

| Problem # | SA Fitness | MA Fitness | CS Fitness |
|-----------|------------|------------|------------|
| 1         | 1          | 0          | 0          |
| 2         | 10         | 4          | 7          |
| 3         | 32         | 25         | 27         |
| 4         | 20         | 11         | 12         |
| 5         | 84         | 52         | 57         |
| 6         | 48         | 37         | 37         |
| 7         | 48         | 4          | 9          |
| 8         | 93         | 75         | 70         |

### Performance Summary:
- **Simulated Annealing (SA)** was the fastest, completing in under one minute but had higher interference values.
- **Memetic Algorithm (MA)** produced the best results but took significantly more time (~27 minutes).
- **Civilization Simulation (CS)** offered a balance between accuracy and speed, taking around 20 minutes and providing strong results.


