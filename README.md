# Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation

This Repository provides unofficial source code for the paper Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation. Please note that it's just an unofficial implementation and due to the lack of GPUs, I'm not able to verify it thoroughly. However, please be assured that this repository contains the basic idea of our paper. 

> Tang Q, Kong Y, Pan L, Lee C. Learning to Solve Soft-Constrained Vehicle Routing Problems with Lagrangian Relaxation. arXiv preprint arXiv:2207.09860. 2022 Jul 20. </cite>

Paper [[PDF](https://arxiv.org/pdf/2207.09860)] 

## Trajectory Shaping

## Modified Return 

## Performance 

## Concerns on the dataset
Although generation of VRP/CVRP datasets is pretty intuitive, VRPTW datasets are tricky to deal with. In our implementation we generate first a CVRP scenario and then a CVRP solution by heuristics. Time windows are then generated according to arrival time in the CVRP solution to make sure that there is at least one valid sulution. However, we believe that there are better ways to generate VRPTW/CVRPTW datsaets. 