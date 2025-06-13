# Interaction-Aware Motion Planning for Autonomous Driving

## Problem Description

Motion planning under social interactions with other road users is one of the most challenging tasks for autonomous driving.  
Human drivers exhibit diverse behaviors depending on surrounding traffic and their individual social preferences.  
Game theory is often used to model the interaction between the ego vehicle and other traffic participants.  
However, traditional game-theoretic approaches typically assume that all agents act rationally and follow a Nash equilibrium [1].

In reality, due to cognitive limitations, human drivers are unable to calculate optimal actions in complex scenarios.  
Therefore, considering human reasoning levels and bounded rationality is essential for planning more efficient and safer trajectories for autonomous vehicles.

To address this, researchers have proposed approaches using:  
- **Quantal Level-k Reasoning** [2] [3] [4]  
- **Quantal Cognitive Hierarchy Models** [5]

The primary goal of this project is to develop an interaction-aware motion planning algorithm for autonomous driving based on game theory, explicitly considering different human reasoning levels.

The interaction between an autonomous vehicle and human drivers is modeled as a **Partially Observable Markov Decision Process (POMDP)**.  
To solve the motion planning efficiently, **Monte Carlo Tree Search (MCTS)** methods will be applied, following the works in [2] [3].

The developed approach will be demonstrated through simulations in various driving scenarios.

## Vehicle Dynamics

The ego vehicle is modeled using a kinematic bicycle model, following the formulation described in [6].

The vehicle state is defined by:

$$
\begin{aligned}
\dot{x} &= v \cos(\theta) \\
\dot{y} &= v \sin(\theta) \\
\dot{\theta} &= \frac{v}{L} \tan(\delta) \\
\dot{v} &= a
\end{aligned}
$$

where $L$ is the distance between the wheels.

The control inputs are:
- $a$: longitudinal acceleration (m/s²)  
- $\delta$: steering angle (radians)

## Driver Modeling

The behavior of human-driven vehicles is modeled using the **Intelligent Driver Model (IDM)** [7].  
The IDM captures longitudinal driving behavior, particularly acceleration and deceleration, based on:
- Desired speed  
- Current speed  
- Distance to the leading vehicle  
- Relative speed (approaching or receding)

The acceleration is computed as:

$$
a = a_{\text{max}} \left( 1 - \left( \frac{v}{v_0} \right)^4 - \left( \frac{s^*(v, \Delta v)}{s} \right)^2 \right)
$$

where:
- $v$ is the current speed  
- $v_0$ is the desired speed  
- $s$ is the actual gap to the leading vehicle  
- $\Delta v$ is the speed difference to the leading vehicle  
- $s^*(v, \Delta v)$ is the desired minimum gap, defined by:

$$
s^*(v, \Delta v) = s_0 + v\,T + \frac{v\,\Delta v}{2\sqrt{a_{\text{max}}\,b}}
$$

and
- $s_0$ is the minimum gap (meters)  
- $T$ is the desired time headway (seconds)  
- $a_{\text{max}}$ is the maximum acceleration (m/s²)  
- $b$ is the comfortable deceleration (m/s²)

## Quantal Level-k Polices

In this project, the behavior of human drivers is modeled using **quantal level-k reasoning** [4].

- The **level \(k\)** represents the depth of a driver's reasoning, where a level-0 agent acts randomly, and higher-level agents best respond to lower-level agents' behavior.
- The **rationality parameter $\lambda$** controls the stochasticity in decision-making: a higher $\lambda$ implies more deterministic, rational behavior, whereas a lower $\lambda$ reflects more random, boundedly rational actions.

For efficient simulation, the Q-functions are precomputed for each reasoning level. The Q-function for agent $i$ at reasoning level $k$ is defined as:

$$ Q_i^k(s_t, a_{i,t}) = \mathbb{E}_i^k \left[ r(s_t, a_{i,t}) + \gamma V_i^k(s_{t+1}) \right] $$

where $V_i^k$ is the value function representing the expected future reward.

The value functions are computed offline using **value iteration** over a discretized state space to ensure computational efficiency during online planning.


## POMCP Formulation

The interaction-aware planning problem is formulated as a **Partially Observable Markov Decision Process (POMDP)** [8], solved with **Partially Observable Monte Carlo Planning (POMCP)** [9].

- **State Space (S):**
  The state includes the position $(x, y)$ of the ego vehicle, the positions of all human-driven vehicles, and the hidden internal states of human drivers. These internal states consist of their reasoning level $k$ from the quantal level-k model and a rationality parameter $\lambda$.

- **Action Space (A):**
  - **Ego vehicle actions:**
    - Accelerate
    - Brake
    - Keep Velocity
    - Indicate Lane Change
    - Lane Change
  - **Human vehicle actions:**
    - Accelerate
    - Brake
    - Keep Velocity

- **Transition Model (T):**
  The transition model defines the probabilistic evolution of the system, incorporating the ego vehicle dynamics (kinematic bicycle model) and the behavior of human agents (IDM). The internal human states $k$ and $\lambda$ remain constant within each episode.

- **Reward Model (R):**
  The reward function is composed of:
  - A large penalty for collisions.
  - A positive reward for reaching the desired lane.
  - A speed reward encouraging the ego vehicle to drive close to the maximum desired speed.

- **Observation Space (O):**
  The observations include the positions of all vehicles. The ego vehicle can perfectly observe the positions but has no direct observation of the human drivers' internal reasoning states $k$ and $\lambda$.


# Experiments
- Explain environment: Straight Street with two lanes. Human agents can be modeled with IDM. To simulate defensive or aggresive drivers, I have introduced a yield area of the side of the car (also in front) which let the driver yield [look at paper 10]. The algorithm is tested in various scenarios. Perform Lane Change with different driving styles and perform LC with many human agents. Second experiment is forcing a lane change or overtaking an obstacle/slow car in front of the ego Car

## Experiment 1: Lane Change
### Calm Driver
- Setup: 
- Result: Belief of the hidden state
- Observation:
  ![Simulation](experiment1/animation_calm1.gif)
### Aggresive Driver
- Setup:
- Result: Belief of the hidden state
- Observation:
  ![Simulation](experiment1/animation_agg1.gif)

## Experiment 2: Lane Change with multiple Drivers
- Setup:
- Result: Belief of the hidden state
- Observation:



## Experiment 3: Forced Lane Change / Overtaking
- Setup: 
- Result: Belief of the hidden state
- Observation:
  ![Simulation](experiment2/animation.gif)


# Bibliography

1. M. Wang, Z. Wang, J. Talbot, J. C. Gerdes, and M. Schwager, “Game-theoretic planning for self-driving cars in multivehicle competitive scenarios,” *IEEE Transactions on Robotics*, vol. 37, no. 4, pp. 1313–1325, 2021.  
2. R. Tian, L. Sun, M. Tomizuka, and D. Isele, “Anytime game-theoretic planning with active reasoning about humans’ latent states for human-centered robots,” in *IEEE ICRA*, 2021, pp. 4509–4515.  
3. S. Dai, S. Bae, and D. Isele, “Game theoretic decision making by actively learning human intentions applied on autonomous driving,” *arXiv preprint arXiv:2301.09178*, 2023.  
4. Y. Breitmoser, J. H. Tan, and D. J. Zizzo, “On the beliefs off the path: Equilibrium refinement due to quantal response and level-k,” *Games and Economic Behavior*, vol. 86, pp. 102–125, 2014.  
5. M. Dang, D. Zhao, Y. Wang, and C. Wei, “Dynamic game-theoretical decision-making framework for vehicle-pedestrian interaction with human bounded rationality,” *arXiv preprint arXiv:2409.15629*, 2024.  
6. J. Kong, M. Pfeiffer, G. Schildbach, and F. Borrelli, “Kinematic and dynamic vehicle models for autonomous driving control design,” in *2015 IEEE Intelligent Vehicles Symposium (IV)*, pp. 1094–1099, 2015.  
7. M. Treiber, A. Hennecke, and D. Helbing, “Congested Traffic States in Empirical Observations and Microscopic Simulations,” *arXiv preprint cond-mat/0002177*, 2000.  
8. L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” *Artificial Intelligence*, vol. 101, no. 1–2, pp. 99–134, 1998.  
9. D. Silver and J. Veness, “Monte-Carlo planning in large POMDPs,” *Curran Associates Inc.*, 2010.
10. S. Bae, D. Saxena, A. Nakhaei, C. Choi, K. Fujimura and S. Moura, "Cooperation-Aware Lane Change Maneuver in Dense Traffic based on Model Predictive Control with Recurrent Neural Network," 2020 American Control Conference (ACC), Denver, CO, USA, 2020, pp. 1209-1216, doi: 10.23919/ACC45564.2020.9147837.
