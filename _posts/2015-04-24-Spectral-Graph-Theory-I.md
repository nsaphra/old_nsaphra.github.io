---
title: Spectral Graph Theory, Pt I - The Laplacian
layout: post
category: Machine_Learning
tags: linear_algebra graphs
year: 2015
month: 04
day: 24
published: true
summary: I explain the Graph Laplacian to myself - and possibly to you!
image: post_one.jpg
---

My goal here is to explain some basic elements of spectral graph theory to myself.
This post may also be useful to an audience with a similar background to mine, including rudimentary linear algebra and lay-computer-scientist exposure to graph theory.

Spectral graph theory is concerned with the analysis of a graph in terms
of the eigenvectors, algebraic and geometric multiplicities, and other
properties of matrix representations of the graph.
These properties provide a way of efficiently extracting
properties of the graph itself, such as communities and stability.
I may address specific applications in machine learning and representation learning in later posts.

In this post, the goal is to define and understand the basic properties of the Laplacian matrix, which is just one way to represent a graph in matrix form. The Laplacian is useful for identifying communities and clusters within the graph, and in analyzing the stability of a system.

Simple Matrix Representations
-------------

Let's start with some intuitive representations of a graph $G = (V,E)$.
We will use ${\bf X}(i,j)$ interchangeably with ${\bf X}_{i,j}$ for entries in
matrix ${\bf X}$, and ${\bf x}(i)$ interchangeably with ${\bf x}_i$ for entries in
vector ${\bf{x}}$.

The *adjacency matrix* ${\bf A} \in \{0,1\}^{|V| \times |V|}$ tells us which nodes are connected.
We set ${\bf A}(u,v) = 1$ if there is an edge from $u$ to $v$.
If the edge $(u,v) \not\in E$, then the corresponding entry in $A$ is set to 0.

In an unweighted graph, the adjacency matrix is sufficient for describing $G$.
However, in our analysis we will often rely on information from another matrix characterizing the graph.
The *degree matrix* ${\bf D} \in \mathbb{N}^{|V| \times |V|}$ gives us information about how connected a node is.
It is a diagonal matrix in which ${\bf D}(u,u) = {\textrm{deg}(u)}$, which corresponds to the number of edges coming from node $u$.

The Walk Matrix
-------------

If a matrix represents a graph, what do vectors represent?
A weight vector ${\bf x} \in \mathbb{R}^{|V|}$ assigns some mass to node $v$, which we can retrieve by multiplying the weight vector by a one-hot vector ${\bf v}$, with one bit active at the element corresponding to $v$.
So we can view a weight vector as a function mapping nodes onto reals.

The *walk matrix* of $G$, defined as ${\bf W} = {\bf A} {\bf D}^{-1}$, provides a way of simulating a random walk through the graph.
If we input a one-hot vector ${\bf v}$, representing a starting point at node $v$, we compute the probability that we end up at node $u$ after one step in a random walk:

$$
\begin{aligned}
\left[{\bf W} {\bf v}\right](u) &=  \sum_{w \in V} \left[ {\bf A} {\bf D}^{-1} \right](u,w) {\bf v}(w)  \\
&= \sum_{w \in V} {\bf A}(u,w) \frac{1}{\textrm{deg}(w)} {\bf v}(w)\\
&= \begin{cases}
    \frac{1}{\textrm{deg}(u)} &\mbox{if } (v,u) \in E \\
    0 & \mbox{otherwise}.
\end{cases}
\end{aligned}
$$

Another way to think of this is in terms of the flow of mass.
We start with a mass of 1 on node $v$, and at each timestep diffuses the mass moves down nearby edges.
If we want to model the mass after $k$ timesteps, starting at node $v$, we can compute it with ${\bf W}^k {\bf v}$.

<!-- TODO: Eigenvalues  -->

The Laplacian
-------------

To understand the Laplacian matrix, let's first define the Laplacian ${\bf L}$
for a simple graph[^1] $G = (V,E)$.
For two node indices $u,v \in V$, the corresponding cell in the Laplacian is given in terms of the degree of each node.
$${\bf L}(u,v) = \begin{cases}
    {\textrm{deg}(u)} &\mbox{if } u = v \\
    -1 & \mbox{if } \textrm{if $(u,v) \in E$}\\
    0 & \mbox{otherwise}.
\end{cases}$$

For the normalized Laplacian $\mathcal{L}$, we instead say
$$\mathcal{L}(u,v) = \begin{cases}
    1 &\mbox{if } \textrm{$u = v$ and ${\textrm{deg}(v)} \not= 0$} \\
    -\frac{1}{\sqrt{ {\textrm{deg}(u)} {\textrm{deg}(v)} }} & \mbox{if } \textrm{if $(u,v) \in E$}\\
    0 & \mbox{otherwise}.
\end{cases}$$

The normalized Laplacian is the form we will use for most analysis.
It is equivalent to normalizing every row and column in the Laplacian.
Let's look at some of its properties:

- Every row and column sums to 0.
- For a graph with no isolated nodes, the diagonals of $\mathcal{L}$ are all 1. In this case, we can decompose the matrix in terms ${\bf D}$ and ${\bf A}$.
- Note that ${\mathcal{L}}(u,v) = {\bf L}(u,v) \frac{1}{\sqrt{ {\textrm{deg}(u)} {\textrm{deg}(v)} }} = {\bf L}(u,v) {\bf D}^{-1/2}(u){\bf D}^{-1/2}(v)$. We therefore know that ${\mathcal{L}}$ decomposes as:

    $$
    \begin{aligned}
    {\mathcal{L}}(u,v) &= {\bf D}^{-1/2} {\bf L} {\bf D}^{-1/2}
    \end{aligned}
    $$.
- The cells
in ${\bf L}$ can be expressed as $${\bf L}(u,v) = \begin{cases}
    -{\bf A}(u,v) &\mbox{if } u \not= v\\
    {\bf D}(u,u) &\mbox{if } u = v
\end{cases}$$, and in a simple graph we know ${\bf A}(u,u) = 0$ since there are no self-cycles. Therefore, we have the decomposition:

    $$
    \begin{aligned}
    {\mathcal{L}}(u,v) &= {\bf D}^{-1/2} ({\bf D} - {\bf A}) {\bf D}^{-1/2}\\
    &= {\bf I} - {\bf D}^{-1/2} {\bf A} {\bf D}^{-1/2}
    \end{aligned}
    $$

### What does it mean?

A intuition for the normalized Laplacian can be gleaned from its product
with a vector $g \in V^n$.

$$\begin{aligned}
\left[ {\mathcal{L}}g \right](u) &= \frac{1}{\sqrt{ {\textrm{deg}(u)} }} \sum_{v: (u,v) \in E} \left(\frac{g(u)}{\sqrt{ {\textrm{deg}(u)} }} - \frac{g(v)}{\sqrt{ {\textrm{deg}(v)} }} \right)\\
&= \frac{1}{\sqrt{ {\textrm{deg}(u)} }} \left({\textrm{deg}(u)} \frac{g(u)}{\sqrt{ {\textrm{deg}(u)} }} -  \sum_{v: (u,v) \in E} \frac{g(v)}{\sqrt{ {\textrm{deg}(v)} }} \right)\\
&= g(u) - \sum_{v: (u,v) \in E} \frac{g(v)}{\sqrt{ {\textrm{deg}(u)} {\textrm{deg}(v)} }}
\end{aligned}$$

For a one-hot vector $g$ representing node $v$, $\[{\mathcal{L}}g\](v) = 1$ and for all other $u \in V$,
$\[{\mathcal{L}}g\](u) = -\frac{1}{\sqrt{ {\textrm{deg}(u)} {\textrm{deg}(v)} }}$
if $u$ and $v$ are adjacent, and otherwise $\[{\mathcal{L}}g\](v) = 0$.
Note that the sum of output cells for the nodes neighboring $v$ will
be -1, and since the only other nonzero entry will be a 1 (in cell $v$), the
sum of values in the resulting vector will be 0! In fact, any output
vector will have a sum of 0.

If we activate more bits in $g$ in addition to $v$, what happens to
$\[{\mathcal{L}}g\](u)$? If we activate every node in the graph and each
cell receives a uniform weight, then the product will be 0. When else
does this happen? If we start with any node $v$ and then activate cells
in our vector corresponding to each of its neighbors, and go on
recursively for each neighbor, then every time we add a score of 1 for a
node, we also add the sum of scores of its neighbors (always -1), with a
resulting sum of 0 for every cell in the output vector. In general, a
product of 0 occurs if the nodes activated in the input vector form a
closed subgraph – that is, no node has an edge connecting it to a node
that is not activated.

<!-- Another way to think of the product ${\mathcal{L}}g$ is as a potential
function where the nodes with positive weights in $g$ “attract” while
their neighbors “repel”. A steady state can be achieved with a vector
representing a closed subgraph, so these attractions and repulsions are
perfectly balanced, resulting in an output vector of 0s.

Notice, however, that a closed subgraph is not the only vector that
induces a steady state! These closed subgraphs form the null space of
the Laplacian. But any eigenvector of the Laplacian will, by definition,
yield a rescaled version of itself. The output vector’s weights on each
node will remain unchanged.

All this gives us a hint about what the Laplacian reveals.
If everyone in your social circle catches a cold, but nobody interacts with people outside of your social circle, the cold will never spread further. You achieved a stable state in which -->

*What are the eigenvalues of a Laplacian? What do the eigenvectors of a Laplacian tell us about the graph itself? What does all this have to do with clustering, or community detection, or representation learning? Answers to these questions and more coming soon in the next thrilling installment of this series of blog posts on Spectral Graph Theory!*

[^1]: A simple graph is a directed graph with no loops and no more than one edge connecting any two nodes.
