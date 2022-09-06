# Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation

This Repository provides unofficial source code for the paper Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation. Unfortuntely, it's just an unofficial implementation and due to the lack of GPUs (at least now), I'm not able to verify it thoroughly. However, please be assured that this repository is fully able to represent the basic idea of our paper. 

> Tang Q, Kong Y, Pan L, Lee C. Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation. arXiv preprint arXiv:2207.09860. 2022 Jul 20. </cite>

Paper [[PDF](https://arxiv.org/pdf/2207.09860)] 

## Usage
The code includes the implementation of following approaches:

* Generate dataset: run ``data_generator/vrpDatagen.py`` for CVRP and run ``data_generator/vrptwDatagen.py`` for CVRPTW.
* CVRP: run ``src/run_vrp.py``.
* CVRPTW: run ``src/run_vrptw.py``.

## Trajectory Shaping
We improve the model performance by intervening the trajectory generation process to boost the quality of the agent’s training information. The motivation is similar to modifying the expression of return. Due to the large search space and the sparsity of optima, guiding the agent to explore and learn the ’good’ actions can be very slow or easily trapped into local optima, especially if the initial state solution is far from the true global optimum. With the underlying model being deterministic and we can easily obtain the next state's reward and cost, we suggest a post-action rejection rule deciding whether to reject the candidate solution respectively when non-improved and improved solutions are found to modify the generated trajectories.
 
<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;P(\textnormal{Reject})&space;=&space;\left\{&space;&space;&space;&space;&space;&space;&space;&space;\begin{array}{ll}&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;\phi&space;&&space;\quad&space;\textnormal{if&space;improved}&space;\\&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;1&space;-&space;\phi&space;&&space;\quad&space;\textnormal{if&space;not&space;improved},&space;&space;&space;&space;&space;&space;&space;&space;\end{array}&space;&space;&space;&space;\right." title="https://latex.codecogs.com/svg.image? P(\textnormal{Reject}) = \left\{ \begin{array}{ll} \phi & \quad \textnormal{if improved} \\ 1 - \phi & \quad \textnormal{if not improved}, \end{array} \right." />

<div align=center><img src="figs\tc_not_tc.png" width="800"></div>

## Modified Return 
The expression of $G_{t}$ is specially designed to encourage better performance in soft-constrained VRPs with 2-exchange moves. First, the immediate reward is penalized by the immediate cost such that the agent is encouraged to find better moves while balancing the reward and cost with iteratively updated $\lambda$s. In addition, We calculate the cumulative value using the maximum value of all pairs of subsequent moves from $s_{t}$ to $s_{t'}$ instead of a summation over all consecutive moves from $s_{t}$ to $s_{t+1}$ as in the $Return$ definition. "Bad" operations that do not improve the objective function will be suppressed, while only the 'good' actions are rewarded with the $\max$ function. It also tends to decorrelate the effect of a series of historical operations so that the agent is less affected by locally optimal trajectories. To sum up, we apply such modification to better mimic the heuristic search process by encouraging more immediate and effective actions that improve the cost-penalized objective function. The following figure provides a visual representation of the definition of $G_t$.
<div align=center><img src="figs\return.png" width="800"></div>
<div align=center><img src="figs\return_vs.png" width="800"></div>

## Performance 
We observed slightly better performance than Google OR-Tools and close performance to LKH-3. 
<div align=center><img src="figs\perf.PNG" width="800"></div>

## Concerns on the dataset
Although generation of VRP/CVRP datasets is pretty intuitive, VRPTW datasets are tricky to deal with. In our implementation we generate first a CVRP scenario and then a CVRP solution by heuristics. Time windows are then generated according to arrival time in the CVRP solution to make sure that there is at least one valid sulution. However, we believe that there are better ways to generate VRPTW/CVRPTW datsaets. 
