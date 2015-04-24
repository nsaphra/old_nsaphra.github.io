---
title: Spectral Graph Theory, Pt I: The Laplacian
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

The Basics of Spectral Graph Theory
===================================

Spectral graph theory is concerned with the analysis of a graph in terms
of the eigenvectors, algebraic and geometric multiplicities, and other
properties of the graph represented as a Laplacian or edge-weight
matrix. These properties provide a way of efficiently extracting
properties of the graph itself, such as communities and stability.

In machine learning, spectral graph theory can be used to identify
clusters within a graph. These clusters can be used to develop
embeddings. In particular, a graph can be used to for a kernel when we
do not have complete similarity information.

The Laplacian
-------------

To understand the Laplacian matrix, let's first define the Laplacian $L$
for a simple[^1] graph $G = (V,E)$. For two node
indices $u,v \in V$, the corresponding cell in the Laplacian is given:
$$L(u,v) = \begin{cases}
    {\textrm{deg}(v)} &\mbox{if } u = v \\
    -1 & \mbox{if } \textrm{if $(u,v) \in E$}\\
    0 & \mbox{otherwise}.
\end{cases}$$ We will use $X(i,j)$ interchangeably with $X_{i,j}$ for
matrix $X$, and ${{\bf{x}}}(i)$ interchangeably with ${{\bf{x}}}_i$ for
vector ${{\bf{x}}}$.

For the normalized Laplacian $\mathcal{L}$, we instead say:
$$\mathcal{L}(u,v) = \begin{cases}
    1 &\mbox{if } \textrm{$u = v$ and ${\textrm{deg}(v)} \not= 0$} \\
    -\frac{1}{\sqrt{{\textrm{deg}(u)}{\textrm{deg}(v)}}} & \mbox{if } \textrm{if $(u,v) \in E$}\\
    0 & \mbox{otherwise}.
\end{cases}$$

The normalized Laplacian is the form we will use for most analysis. For
a graph with no isolated nodes, note the diagonals of $\mathcal{L}$ are
all 1. In this case, we can easily decompose $\mathcal{L}$ in terms of
two matrices: the diagonal *degree matrix* $D$ such that
$D(u,u) = {\textrm{deg}(u)}$, and the *adjacency matrix* $A$ such that
$A(u,v) = 1$ if $u$ and $v$ are adjacent and otherwise is 0.

To illustrate, first note that
${\mathcal{L}}(u,v) = L(u,v) \frac{1}{\sqrt{{\textrm{deg}(u)} {\textrm{deg}(v)}}} = L(u,v) D^{-1/2}(u)D^{-1/2}(v)$,
and thus $${\mathcal{L}}(u,v) = D^{-1/2} L D^{-1/2}$$ Because the cells
in $L$ can be expressed as $$L(u,v) = \begin{cases}
    -A(u,v) &\mbox{if } u \not= v\\
    {\textrm{deg}(u)} &\mbox{if } u = v
\end{cases}$$ we also have $$\begin{aligned}
{\mathcal{L}}(u,v) &= D^{-1/2} (D - A) D^{-1/2}\\
&= I - D^{-1/2} A D^{-1/2} {\addtocounter{equation}{1}\tag{\theequation}}\end{aligned}$$

### What does it mean? {#what-does-it-mean .unnumbered}

A intuition for the normalized Laplacian can be gleaned from its product
with a vector $g \in V^n$. $$\begin{aligned}
[{\mathcal{L}}g](u) &= \frac{1}{\sqrt{{\textrm{deg}(u)}}} \sum_{v: (u,v) \in E} \left(\frac{g(u)}{\sqrt{{\textrm{deg}(u)}}} - \frac{g(v)}{\sqrt{{\textrm{deg}(v)}}} \right)\\
&= \frac{1}{\sqrt{{\textrm{deg}(u)}}} \left({\textrm{deg}(u)} \frac{g(u)}{\sqrt{{\textrm{deg}(u)}}} -  \sum_{v: (u,v) \in E} \frac{g(v)}{\sqrt{{\textrm{deg}(v)}}} \right)\\
&= g(u) - \sum_{v: (u,v) \in E} \frac{g(v)}{\sqrt{{\textrm{deg}(u)} {\textrm{deg}(v)}}} {\addtocounter{equation}{1}\tag{\theequation}}\end{aligned}$$
For a one-hot vector $g$ representing node $v$,
$[{\mathcal{L}}g](v) = 1$ and for all other $u \in V$,
$[{\mathcal{L}}g](u) = -\frac{1}{\sqrt{{\textrm{deg}(u)} {\textrm{deg}(v)}}}$
if $u$ and $v$ are adjacent, and otherwise $[{\mathcal{L}}g](v) = 0$.
Note that the sum of output cells for the nodes neighboring $v$ will
be -1, and since the only other nonzero entry will be a 1 in $v$, the
sum of values in the resulting vector will be 0! In fact, any output
vector will have a sum of 0.

If we activate more bits in $g$ in addition to $v$, what happens to
$[{\mathcal{L}}g](u)$? If we activate every node in the graph and each
cell receives a uniform weight, then the product will be 0. When else
does this happen? If we start with any node $v$ and then activate cells
in our vector corresponding to each of its neighbors, and go on
recursively for each neighbor, then every time we add a score of 1 for a
node, we also add the sum of scores of its neighbors (always -1), with a
resulting sum of 0 for every cell in the output vector. In general, a
product of 0 occurs if the nodes activated in the input vector form a
closed subgraph – that is, no node has an edge connecting it to a node
that is not activated.

Another way to think of the product ${\mathcal{L}}g$ is as a potential
function where the nodes with positive weights in $g$ “attract” while
their neighbors “repel”. A steady state can be achieved with a vector
representing a closed subgraph, so these attractions and repulsions are
perfectly balanced, resulting in an output vector of 0s.

Notice, however, that a closed subgraph is not the only vector that
creates a steady state! These closed subgraphs form the null space of
the Laplacian. But any eigenvector of the Laplacian will, by definition,
yield a rescaled version of itself. The output vector’s weights on each
node will remain unchanged.

What are the eigenvalues of a Laplacian? What do the eigenvectors of a Laplacian tell us about the graph itself? What does all this have to do with clustering, or community detection, or representation learning? Answers to these questions and more coming soon in the next thrilling installment of this series of blog posts on Spectral Graph Theory!
