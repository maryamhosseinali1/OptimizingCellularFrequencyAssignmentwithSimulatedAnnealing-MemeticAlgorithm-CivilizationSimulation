#!/usr/bin/env python
# coding: utf-8

# #  <span style='font-family:"Times New Roman"'> <span styel=''> Fixed Channel Assignment Problem

# #  <span style='font-family:"Times New Roman"'> <span styel=''> MohammadReza Javaheri 610300038 - Maryam Hosseinali 610398209

# ##  <span style='font-family:"Times New Roman"'> <span styel=''> Analyzing Cellular Mobile Frequency Assignment Problem:

# ### <span style='font-family:"Times New Roman"'> <span styel=''> The chosen paper discusses a new approach to solve a problem called the frequency assignment problem in cellular radio networks.
#     
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The frequency assignment problem is a well-known optimization problem which involves allocating frequencies to different transmitters or communication devices in a way that minimizes interference and maximizes overall system performance.
#     
# ### <span style='font-family:"Times New Roman"'> <span styel=''> the main goal of  frequency assignment is to ensure that each transmitter is assigned a frequency channel that allows it to operate without causing significant interference to neighboring transmitters. This interference can negatively impact the quality of communication, limit the coverage range, and result in overall inefficiency in the wireless network.
#     
# ### <span style='font-family:"Times New Roman"'> <span styel=''> the authors employed the memetic algorithm (combination of genetic algorithms and local search) to solve this problem. The GA helps in exploring a wide range of solutions through evolutionary operators, while the tabu search (specific implementation of local search) focuses on refining the solutions through local search strategies.
#     
# ### <span style='font-family:"Times New Roman"'> <span styel=''> By applying the memetic algorithm, the authors aim to find good solutions to the frequency assignment problem efficiently. The algorithm iteratively evaluates and modifies potential assignments based on their fitness (degree of interference), gradually improving the overall quality of the assignments.
# 
# 
# 
# 
# 

# ## <span style='font-family:"Times New Roman"'> <span styel=''> general description of our project:

# ### <span style='font-family:"Times New Roman"'> <span styel=''> In our project, we explored different approaches to solve the problem of fixed channel assignment. We implemented three techniques: Simulated Annealing (SA), Memetic algorithm, and our own innovative approach called civilization simulation.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''>  We first applied Simulated Annealing, which helped us refine the frequency assignments by gradually reducing the temperature of the system. This approach allowed us to explore different solutions and converge towards optimal ones.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> Next, we employed the Memetic algorithm(combination of genetic algorithms and local search). This technique enabled us to explore a wide range of solutions and refine them through iterative improvements.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> In addition to SA and Memetic algorithm, we introduced our own innovative approach called civilization simulation, This approach combined ideas from Simulated Annealing and genetic algorithms to find optimal frequency assignments. It showed promising results in our experiments.
# ### <span style='font-family:"Times New Roman"'> <span styel=''> In our project report, we will discusse these three approaches in detail, focusing on their strengths and how well they worked for solving the fixed channel assignment problem. We will also share information about how we implemented them and explained the results of our experiments.
# 
# 

# ## <span style='font-family:"Times New Roman"'> <span styel=''>  import required libraries

# In[2]:


import json
from bitarray import bitarray
import random
import math
import copy


# - json: for reading JSON data from a file
# - bitarray: for working with bit arrays
# - copy: for creating deep copies of objects


# ## <span style='font-family:"Times New Roman"'> <span styel=''> Data Extraction and Configuration from Article Table
# 
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> We performed data extraction and configuration from a table in our article using a separate code. we extracted important specifications from article's table . We then populated these values in the config.json file, which serves as a centralized configuration file for our algorithm. This approach allows us to separate the data extraction process from the main code, making it easier to update and configure the algorithm based on different problem scenarios.

# ## <span style='font-family:"Times New Roman"'> <span styel=''> Loading JSON Data from File

# In[3]:


data = json.load(open('config.json'))

#loads the JSON data from the file 'config.json' and stores it in the data variable


# ## <span style='font-family:"Times New Roman"'> <span styel=''> class Problem
# ### <span style='font-family:"Times New Roman"'> <span styel=''> the Problem class represents a specific instance of the problem.It has instance variables n, m, C, and D to store the number of radio Cells(# of Cells), Lower Bound (number of required channel), compatibility matrix, and Demand vector , respectively.
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The purpose of the Problem class is to store the problem data and provide a structured way to access and work with it.
# ###  <span style='font-family:"Times New Roman"'> <span styel=''>The parameters for each problem instance are provided based on the information in the article:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> n: Represents the number of cells in the problem instance.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> m: Represents minimum number of frequencies or channels required for the problem instance.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> C: Represents the compatibility matrix, which describes the interference between different cells.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> D: Represents the demand vector, which specifies the number of required channels for each cell (need of each cell for frequencies).

# In[4]:


class Problem:
    def __init__(self,n,m,C,D) -> None:
        self.n = n
        self.m = m
        self.C = C
        self.D = D
        pass


# Problem Specifications
problems = {}
problems[1] = Problem( 4 , 11, data['C1'], data['D1']) 
problems[2] = Problem(25,  73, data['C2'], data['D2']) 
problems[3] = Problem(21, 381, data['C3'], data['D3']) 
problems[4] = Problem(21, 533, data['C4'], data['D3']) 
problems[5] = Problem(21, 533, data['C5'], data['D3']) 
problems[6] = Problem(21, 221, data['C3'], data['D4']) 
problems[7] = Problem(21, 309, data['C4'], data['D4']) 
problems[8] = Problem(21, 309, data['C5'], data['D4'])

#filling the dictionary with information related to the Table I: Problem Specifications.


# # <center>Problems</center>
# 
# | Problem # | number of radio cells (n) | number of channels (m) | Compatibility matrix (C) | Demand vector (D) |
# | :-----: | :-----: | :-----: | :-----: | :-----: | 
# | 1 | 4 | 11 | C1 | D1 |
# | 2 | 25 | 73 | C2 | D2 |
# | 3 | 21 | 381 | C3 | D3 |
# | 4 | 21 | 533 | C4 | D3 |
# | 5 | 21 | 533 | C5 | D3 |
# | 6 | 21 | 221 | C3 | D4 |
# | 7 | 21 | 309 | C4 | D4 |
# | 8 | 21 | 309 | C5 | D4 |

# ## <span style='font-family:"Times New Roman"'> <span styel=''> class Result
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The Result class represents a solution generated for a specific problem instance.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. if the flag is True, the initialization is skipped, allowing the instance to be created for other purposes. When flag is False, the instance is created for a new assignment, and the following steps are executed: 
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. An empty result array is created to store the assignment for each cell and frequency.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 3. The assignment is generated randomly based on the compatibility matrix, demand vector, and number of channels provided by the problem instance.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 4. The fitness of the assignment is calculated using the Cost_Function method, which evaluates the quality of the assignment based on interference and channel conflicts.
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The purpose of the Result class is to represent a potential solution to the problem instance and calculate its fitness.

# In[5]:


class Result:
    def __init__(self,problem:Problem,flag = False):
    # Problem instance associated with this Result and
    #flag indicate if the Result is initializing from an existing solution
        if(flag):
            return

        
        # Generate an initial solution for the problem using random assignment
        self.res = [bitarray([0 for _ in range(problem.m)]) for _ in range(problem.n)]
        for i in range(problem.n):
            index = random.randint(0,problem.m - (problem.D[i]-1) * problem.C[i][i] +1)
            for _ in range(problem.D[i]):
                index %= problem.m
                try:

                   # Find an available frequency channel for the current cell
                    while (self.res[i][index] == 1):
                        index = (index + random.randint(0, problem.C[i][i])) % (problem.m)
                except:
                    print(index,problem.m)
                    
                # Assign the frequency channel to the cell     
                self.res[i][index] = 1
                
                # Update the index, considering the compatibility and demand constraints
                index += problem.C[i][i] + random.randint(0,int((problem.m-((problem.D[i]-1)*problem.C[i][i]+1))/problem.D[i]))
                index = index% (problem.m)
        # Calculate the fitness of the initial solution using the Cost_Function
        self.fitness = Cost_Function(problem,self)
    


# ## Channel Evaluation and Update Functions

# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> This section of code includes four functions: Cell_Cost_Function, Conflict_Counter, Cost_Function, and Update_Result. Each function serves a specific purpose within the channel assignment process and is utilized in SA, memetic algorithm, and civilization simulation.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> Cell_Cost_Function: this function calculates the cost or interference between channels within a single cell. It iterates over the channels within a single cell and checks their assignment status. If two channels are assigned, it compares their indices and increments the cost if the difference is less than the compatibility value. The final cost value represents the cost between channels within the cell.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> Conflict_Counter: this function counts the interference between channels across different cells.It iterates over all cells and channels, excluding the current cell and channel combination. If a channel is assigned in a different cell and channel combination, it increments the conflict count based on the compatibility value. The final count represents the conflicts or interference between the given channel and other channels across different cells.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> Cost_Function: this function calculates the overall cost or interference in the channel assignments. It provides a measure of the total interference or conflicts in the entire channel assignment solution. It iterates over the cells and channels, excluding unassigned channels. It compares each channel with channels in subsequent cells, skipping the same cell and lower channel index. The cost is incremented if the difference between the channels is less than the compatibility value between the cells. The final result represents the overall cost or interference in the channel assignments.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> Update_Result: this function performs local search by updating channel assignments through swapping.By randomly selecting a cell and swapping the channel assignments, the Update_Result function explores different local changes to the channel assignments and evaluates their impact on interference levels. This process allows for potential improvements in the channel allocation by iteratively refining the assignments based on local search heuristics.
# 
# 

# In[30]:



def Cell_Cost_Function(problem: Problem, result: Result, cell):
    # Calculate the cost between channels within a single cell.

    res = 0
    for i in range(problem.m):
        if result.res[cell][i] != 1: # If the channel is not assigned in the cell, skip to the next channel.
            continue
        for j in range(i + 1, problem.m):
            if result.res[cell][j] != 1: # If the next channel is not assigned in the cell, skip to the next next channel.
                continue
            res += int(abs(i - j) < problem.C[cell][cell]) 
            # Increment the cost if the differencebetween the channel
            #indices is less than the compatibility value between the cells.
    return res




def Conflict_Counter(problem: Problem, result: Result, cell, channel) :
    # Count the conflicts between channels across different cells.

    res = 0
    for i in range(problem.n):
        for j in range(problem.m):
            if i == cell and j == channel: # Skip the current cell and channel combination.
            #(avoid counting conflicts between the same cell and channel)
                continue
            if result.res[i][j]: 
            # If the channel is assigned in the current cell and channel combination, increment the conflict count.
                res += int(abs(channel - j) < problem.C[cell][i])
                # Increment the count if the difference between the channels is less than the compatibility value between the cells.
    return res


def Cost_Function(problem: Problem, result: Result):
    # Calculate the overall cost in the channel assignments.

    res = 0
    for i in range(problem.n): # Iterate over the cells.
        for k in range(problem.m): # Iterate over the channels in the current cell.
            if not result.res[i][k]: # If the channel is not assigned in the current cell, skip.
                continue
            for j in range(i, problem.n): # Iterate over the cells starting from the current cell.
                for l in range(problem.m): # Iterate over the channels in the second cell.
                    if (not result.res[j][l]) or (i == j and l <= k): # If the channel is not assigned in the second cell or it is the same cell with a lower channel index, skip.
                        continue
                    res += int(abs(k - l) < problem.C[i][j]) # Increment the cost if the difference between the channels is less than the compatibility value between the cells.
    return res

def Update_Result(problem: Problem, result: Result, temperature):
    # Update the channel assignments based on the acceptance probability.

    # Randomly select a cell from the problem instance.
    cell = random.randint(0, problem.n - 1)

    # Randomly choose an unassigned channel from the selected cell.
    channel1 = random.choice(result.res[cell].search(1))

    # Randomly choose an assigned channel from the selected cell.
    channel2 = random.choice(result.res[cell].search(0))

    # Update the channel assignments by swapping channel1 with channel2 within the selected cell.
    # Calculate the fitness (interference) of the new channel assignments for channel1 and channel2.
    result.res[cell][channel1] = 0
    fitness1 = Conflict_Counter(problem, result, cell, channel1)
    result.res[cell][channel2] = 1
    fitness2 = Conflict_Counter(problem, result, cell, channel2)


    # Compare the fitness values and determine whether to accept the new assignment or revert to the previous one.
    if fitness1 > fitness2 or random.random() > math.exp((fitness2 - fitness1) / temperature):
        # Accept the new assignment.
        result.fitness += fitness2 - fitness1
    else:
        # Revert the channel assignments to their previous state.
        result.res[cell][channel1] = 1
        result.res[cell][channel2] = 0


# ## <span style='font-family:"Times New Roman"'> <span styel=''>  First approach; Simulated Annealing
# ###  <span style='font-family:"Times New Roman"'> <span styel=''> Introduction: Simulated Annealing is a stochastic optimization algorithm that aims to find the global optimum of a given problem. It utilizes a cooling schedule to iteratively explore the solution space and escape local optima.
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Advantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. SA is effective in escaping local optima and searching for global optima.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. It can handle complex optimization problems with a large solution space.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 3. SA allows occasional moves towards solutions with higher cost, enabling exploration of suboptimal solutions.
# 
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Disadvantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. The performance of SA heavily relies on the choice of temperature schedule and cooling rate.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. SA may require a large number of iterations to converge to a near-optimal solution.
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The code below initializes the result, sets the initial temperature, and performs iterations to update the result by calling the Update_Result function. The temperature is gradually decreased according to the cooling rate until the target fitness is achieved or the temperature reaches zero.

# In[22]:


def SA(problem: Problem, cooling_rate, initial_temperature, target):
    
   # Initialize the result object and temperature
    result = Result(problem)
    temperature = initial_temperature
    
    
    while temperature > 1:
        # Perform iterations at current temperature
        for _ in range(10):
            Update_Result(problem, result, temperature)
        
        # Decrease the temperature
        temperature = temperature * cooling_rate
        
        # Check if the target fitness value has been reached
        if result.fitness <= target:
            break

    
    return result


# In[ ]:


Cost_Function()


# ## <span style='font-family:"Times New Roman"'> <span styel=''> Second appoach; Memetic Algorithm
# ### <span style='font-family:"Times New Roman"'> <span styel=''> Introduction: Memetic Algorithm is a hybrid optimization algorithm that combines evolutionary computation with local search techniques. It aims to improve the quality of solutions by incorporating global exploration and local exploitation.
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Advantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. Combines global and local search to efficiently explore and refine solutions.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. Helps improve solution quality by leveraging local search techniques.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 3. Adaptable to problem characteristics by customizing the local search process.
# 
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Disadvantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. requires careful parameter tuning for optimal performance.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. can be computationally expensive for large problem instances.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''>  3. Sensitive to the quality of initial solutions.
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The provided code shows the implementation of the MA algorithm. It initializes a population of results and performs iterations to improve the fitness of the population through crossover operations and local search using the local_search function.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> local_search function: This function is a key component of the memetic algorithm. It performs a local search by randomly updating channel assignments within a cell. The process involves selecting a cell randomly, choosing two channels (one with a value of 1 and the other with a value of 0) within that cell, and swapping their assignments. The fitness of the result is then updated based on the change in conflict count.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> Experience function: This function simulates years of experience by updating the channel assignments. It iteratively applies the Update_Result function to the result object, simulating the accumulation of experience over time. The number of iterations is determined based on the age parameter, allowing for increased experience as the age increases.
# 
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> crossOver function: This function performs crossover between two parent solutions to generate a new child solution. It randomly selects elements from the parent solutions to create a new solution with a combination of their traits. In this case, the function selects elements from each parent's channel assignment and creates a new child solution. The fitness of the child solution is calculated using the Cost_Function.
#     
# ### **PS** : <span style='font-family:"Times New Roman"'> <span styel=''> Due to the potential for mistakes and confusion in understanding the crossover process described in the article, we decided to implement our own approach in the Memetic Algorithm (MA) implementation. The article suggests calculating Pi for each parent to guide the crossover process, but since the values of Cii and di are the same for both parents, it results in identical Pi values for them.
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> To ensure a clearer and less prone to errors implementation, we chose a different strategy for crossover in our MA. Our crossover function randomly selects genetic information from the parent solutions and combines them to create a new child solution. This approach simplifies the process and eliminates the need for complex Pi calculations.
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> By adopting our own crossover approach, we aim to maintain simplicity and reduce the risk of potential mistakes or misunderstandings. Although our method differs from the one described in the article, we believe it provides an effective way to combine genetic information and produce diverse offspring within the MA framework. 
#     
# ### <span style='font-family:"Times New Roman"'> <span styel=''> It is important to note that our crossover function is designed and implemented to efficiently handle changes without recalculating from the beginning. Instead, it continues its process based on the existing information and calculations up to that point.

# In[8]:


def local_search(result: Result, problem: Problem):
    # Perform local search by randomly updating channel assignments within a cell
    for _ in range(100):
        
        # Select a random cell and its associated channels
        cell = random.randint(0, problem.n - 1)
        channel1 = random.choice(result.res[cell].search(1))
        channel2 = random.choice(result.res[cell].search(0))
        
        # Update channel assignments for the selected cell
        result.res[cell][channel1] = 0
        fitness1 = Conflict_Counter(problem, result, cell, channel1)
        result.res[cell][channel2] = 1
        fitness2 = Conflict_Counter(problem, result, cell, channel2)
        
        # Check if the new assignment improves fitness, and update accordingly
        if fitness1 > fitness2:
            result.fitness += fitness2 - fitness1
        else:
            result.res[cell][channel1] = 1
            result.res[cell][channel2] = 0
            
            
def Experience(problem: Problem, result: Result, initial_temperature, age):
    # Simulate years of experience by updating the channel assignments
    for i in range(int(math.sqrt(age + 1) * 30)):
        # Update channel assignments using the Update_Result function
        Update_Result(problem, result, initial_temperature)
        
        
        
            
def crossOver(parents, problem: Problem):
    # Perform crossover between two parent solutions to generate a new child solution
    
    # empty list to store the channel assignments for the child solution
    children = [[] for _ in range(problem.n)]
    
    
    for i in range(len(parents[1].res)):
        # Randomly choose which parent's channel assignment to inherit
        if random.random() < 0.5:
            children[i] = parents[0].res[i].copy() # Inherit from parent 1
        else:
            children[i] = parents[1].res[i].copy() # Inherit from parent 2
    
    # a new result object for the child solution
    res = Result(problem, True)
    
    # Assign the channel assignments to the child solution
    res.res = children
    
    # Calculate the fitness value for the child solution
    res.fitness = Cost_Function(problem, res)
    
    # Return the child solution
    return res
            


# ### <span style='font-family:"Times New Roman"'> <span styel=''> The Memetic Algorithm (MA) is a optimization technique used for channel assignment.It starts by creating an initial population of channel assignment solutions.
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The algorithm iteratively selects parent solutions, performs crossover to create new solutions, and applies local search to improve their quality.The population is updated based on fitness, and the process continues until the desired fitness level is reached. The MA combines genetic operators (crossover) with local search to explore diverse solutions and converge towards better channel assignments.

# In[57]:


def MA(problem: Problem, iterationNumber, population_count):
    # Initialize the population
    population = [Result(problem) for _ in range(population_count)]
    population.sort(key=lambda x:x.fitness)
    
    for i in range(iterationNumber):
        for _ in range(int(population_count / 2)):
            # Perform crossover and local search on selected parents
            child = crossOver(random.choices(population, weights = [x.fitness+1 for x in population], k=2), problem)
            local_search(child, problem)
            
            # Add child to the population
            population.append(child)
        
        population.sort(key=lambda x: x.fitness)
        population = population[:population_count]
        
    return population[0]


# ## <span style='font-family:"Times New Roman"'> <span styel=''> Third approach; Civilization Simulation
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> Introduction: Civilization Simulation is our unique and creative approach that combines Simulated Annealing (SA) and Genetic Algorithm (GA) to solve optimization problems. It offers a novel way to address the fixed channel assignment problem.
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> In Civilization Simulation, we introduce the concept of simulating a society where individuals represent different channel assignments. Each generation is executed by performing one round of SA for a certain number of stages. However, instead of using the traditional temperature parameter, we replace it with the age of the individual gene in the simulation.
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> The idea behind this approach is to mimic the behavior of individuals in a society. Younger genes tend to explore more random answers, similar to the exploration phase in SA. As genes age, they gradually shift towards more accurate answers, resembling the exploitation phase in SA. This dynamic balance between exploration and exploitation allows for effective search and convergence towards optimal channel assignments.
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> To introduce the genetic algorithm component, individuals are allowed to mate from a certain age onwards. This enables the exchange of genetic information between individuals and the creation of offspring with potentially improved channel assignments. This combination of SA and GA techniques enhances the diversity and adaptability of the population, leading to better exploration of the solution space.
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> To manage population size and prevent stagnation, we have implemented mechanisms such as individuals committing suicide after a certain age without achieving the desired goal. This encourages the removal of ineffective solutions from the population. Additionally, in cases where the population becomes too large, lower-performing adults may be eliminated to maintain a balance and prevent overcrowding.
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Advantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. The Civilization Simulation approach combines the strengths of simulated annealing and genetic algorithms to enhance exploration and exploitation capabilities.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. It can efficiently search for high-quality solutions in complex optimization problems.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 3. The integration of multiple algorithms allows for a more robust and adaptive optimization process.
# 
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Disadvantages:
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 1. The performance of Civilization Simulation may depend on parameter tuning and the specific problem domain.
# > ### <span style='font-family:"Times New Roman"'> <span styel=''> 2. The complexity of the algorithm may require more computational resources compared to individual algorithms.
# 
# ### <span style='font-family:"Times New Roman"'> <span styel=''> The provided code demonstrates the implementation of the Civilization Simulation algorithm. It creates a population of individuals (Person class) and iteratively performs simulation years, incorporating experience, crossover operations, and local search. The population evolves based on fitness and age constraints until the target fitness is achieved or population size and age criteria are met.
# 
# 
# 

# In[10]:


class Person:
    def __init__(self, problem: Problem, flag: bool = False) -> None:
        # Initialize the Person object and age
        self.age = 0
        if flag:
            return
        # If flag is True, it means the Person object is being initialized from an existing individual,
        # so no initialization is needed
        
        # Create a Result object for the person
        self.result = Result(problem)
    
    def nextYear(self, problem: Problem):
        # Simulate one year of experience for the person
        Experience(problem, self.result, 50 - self.age, self.age)
        self.age += 1


# In[69]:


def CivSim(problem: Problem, iterationNumber, populationSize, SuicideAge, PubertyAge):
    # Initialize the population with two individuals
    Population = [Person(problem) for _ in range(2)]
    Population.sort(key=lambda x: x.result.fitness)
    
    # Civilization Simulation loop
    for _ in range(iterationNumber):
        # Simulate one year of experience for each person in the population
        for people in Population:
            people.nextYear(problem)
        
        # Select adults (individuals older than PubertyAge)
        adults = [person for person in Population if person.age > PubertyAge]
        
        # Perform crossover and local search on selected parents
        if len(adults) >= 2:
            for _ in range(2):
                child = crossOver(random.choices(list(map(lambda x: x.result, adults)), weights=[x.result.fitness for x in adults], k=2),problem)
                local_search(child, problem)
                newChild = Person(problem, True)
                newChild.result = child
                Population.append(newChild)
        
        # Sort the population based on fitness values
        Population.sort(key=lambda x: x.result.fitness)
        
        if(Population[0].result.fitness==0):
            return Population[0]
        
        # Remove individuals older than SuicideAge from the population
        Population = [people for people in Population if people.age < SuicideAge]
        
        # Control population size based on fitness and age
        if len(adults) > 0 and len(Population) > 2 * populationSize:
            mean = sum(map(lambda x: x.result.fitness, adults)) / len(adults)
            Population = [person for person in Population if (person.result.fitness < mean or person.age < PubertyAge)]
            
    return Population[0]
    


# ## <span style='font-family:"Times New Roman"'> <span styel=''> Results and Discussian

# ### <span style='font-family:"Times New Roman"'> <span styel=''> After running the implementations, we obtained the following results:
# 
# ## <span style='font-family:"Times New Roman"'> <span styel=''> SA algorithm:
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(1): 1
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(2): 10
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(3): 32
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(4): 20
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(5): 84
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(6): 48
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(7): 48
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(8): 93
# 
# ## <span style='font-family:"Times New Roman"'> <span styel=''> MA algorithm:
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(1): 0
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(2): 4
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(3): 25
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(4): 11
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(5): 52
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(6): 37
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(7): 4
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(8): 75
#         
# ## <span style='font-family:"Times New Roman"'> <span styel=''> Civilization Simulation:
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(1): 0
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(2): 7
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(3): 27
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(4): 12
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(5): 57
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(6): 37
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(7): 9
# * ### <span style='font-family:"Times New Roman"'> <span styel=''> problem(8): 70
# 
# 
# 
#     
#   
# ### <span style='font-family:"Times New Roman"'> After evaluating the SA, memetic algorithm, and civilization simulation approaches for the channel assignment problem, we compared their computational efficiency. Our analysis focused on two key factors: accuracy and speed.
# 
#  ### <span style='font-family:"Times New Roman"'> SA algorithm provided the solutions in less than one minute. MA algorithm took about 27 minutes, and Civilization Simulation took about 20 minutes
#     
# ### <span style='font-family:"Times New Roman"'> The memetic algorithm and civilization simulation approaches provided good accuracy due to their ability to generate and evaluate multiple offspring in each generation. This allowed for a thorough exploration of the solution space, resulting in potentially higher quality solutions. However, this came at the cost of increased computational time.
# 
# ### <span style='font-family:"Times New Roman"'> On the other hand, the SA algorithm prioritized speed over exhaustive exploration. It focused on refining a single solution through iterative optimization, leading to faster computation. While this approach may not always guarantee the best possible solution, it offers a reasonable trade-off between accuracy and speed.
# 
# ### <span style='font-family:"Times New Roman"'> If accuracy is of utmost importance, and computational time is not a major concern, the memetic algorithm or civilization simulation can be viable choices. These approaches can provide more comprehensive exploration of the solution space and potentially yield higher quality solutions.
# 
# ### <span style='font-family:"Times New Roman"'> However, if speed is a critical factor, and obtaining a reasonably good solution within a shorter time frame is the priority, the SA algorithm is the recommended choice. Its streamlined process allows for faster computation by focusing on optimizing a single solution.
# 
# ## <span style='font-family:"Times New Roman"'> In conclusion, the choice between the memetic algorithm, civilization simulation, and SA depends on the specific requirements of the problem. If accuracy is paramount, the memetic algorithm or civilization simulation can be suitable. If speed is the priority, the SA algorithm offers a more efficient solution. For a balance between accuracy and speed, the civilization simulation approach can be considered.
# 
# 
