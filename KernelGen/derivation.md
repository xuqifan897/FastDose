## The derivation of the optimization algorithm for the exponential kernel
Conventions:
* $x_{i+1}$: the x coordinate of the $i$'th point.
* $y_i$: the value of the tabulated kernel at $x_{i+1}$
* $A$, $a$, $B$, $b$: the parameters
* The exponential kernel (in the cumulative form):
    $$
    f(x) = \frac{A}{a}\left(1-e^{-ax}\right) + \frac{B}{b}\left(a-e^{-bx}\right)
    $$
    We then reformulate $A:=\frac{A}{a}$, $B:=\frac{B}{b}$, then the kernel becomes:
    $$
    f(x) = A(1-e^{-ax}) + B(1-e^{-bx})
    $$
* The loss function:
    $$
    \begin{align*}
    L &= \sum_i (y_i - f(x_{i+1}))^2 \\
    &= \sum_i \left(\left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i\right)^2
    \end{align*}
    $$
* We have
    $$
    \begin{align*}
    \frac{\partial L}{\partial A} &= \sum_i 2\left( \left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i \right) (1-e^{-ax_{i+1}}) \\
    \frac{\partial L}{\partial B} &= \sum_i 2\left( \left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i \right) (1-e^{-bx_{i+1}}) \\
    \frac{\partial L}{\partial a} &= \sum_i 2\left( \left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i \right) Ax_{i+1}e^{-ax_{i+1}} \\
    \frac{\partial L}{\partial a} &= \sum_i 2\left( \left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i \right) Bx_{i+1}e^{-bx_{i+1}}
    \end{align*}
    $$

* Then we calculate the Hessian matrix:
    $$
    \begin{align*}
    \frac{\partial^2 L}{\partial A^2} &= \sum_i 2(1-e^{-ax_{i+1}})^2 \\
    \frac{\partial^2 L}{\partial A \partial B} &= \sum_i 2(1-e^{-ax_{i+1}})^2 \\
    \frac{\partial^2 L}{\partial A\partial a} &= \sum_i 2(1-e^{-ax_{i+1}})Ax_{i+1}e^{-ax_{i+1}} + 2\left( \left(A(1-e^{-ax_{i+1}}) + B(1-e^{-bx_{i+1}})\right)-y_i \right)x_{i+1}e^{-ax_{i+1}} \\
    \frac{\partial^2 L}{\partial A \partial b} &= 2Bx_{i+1}e^{-bx_{i+1}}(1-e^{-ax_{i+1}})
    \end{align*}
    $$
    I cannot finish the derivation. So I decide to use JAX instead.

## BEV and PVCS dose conversion
In this implementation, an idea was borrowed from an existing work on FCBB dose calculation, in which the dose was calculated in a parallel beam setting and then transformed to a divergent beam setting. Here we deduce the dose transformation between the two settings.

<!-- * Coordinate systems
    * Here we define the BEV coordinate system: $(u,v,r)$, where $(u,v)$ is the point at which the ray connecting the source and the point $\vec{x}$ intersects the reference plane, and $r$ is the distance between the intersection point and $\vec{x}$. We define the distance between the source and the center of the reference plane to be $s$, the distance between the source and the intersection point to be $r_0$. Such that
    $$
    r_0 = \sqrt{u^2+v^2+s^2}
    $$
    * The Terma to point $\vec{x}$ can be expressed as:
        $$
        T(u,v,r) = \frac{r_0^2}{(r+r_0)^2}\exp{(-\hat r(u,v,r))},
        $$
        where $\hat{r}(u,v,r)$ is the radiological path length between the source and $\vec{x}$:
        $$
        \hat{r}(u,v,r)=\int_{(u,v,-r_0)}^{(u,v,r)}\eta (u,v,t)dt
        $$
    * The Terma in a parallel beam at $(x, y, z)$
        
        Assume the radiological density $\eta(u,v,r)$ remains constant with different $s$. -->

* Coordinate systems

    Here we define the BEV coordinate systems: $(u,v,d)$, where $(u,v)$ is the point at which the ray connecting the source and the point $\vec{x}$ intersects the reference plane, and $d$ is the distance between the point $\vec{x}$ and the reference plane. Assume the distance between the source and the center of the reference plane to be $d_0$, and the distance between the source and the intersection point to be $r_0$, such that:
    $$
    r_0 = \sqrt{u^2+v^2+d_0^2}.
    $$
    The dose at point $(u,v,d)$ can be expressed as:
    $$
    T(u,v,d) = \frac{d_0^2}{(d+d_0)^2}\exp{(-\hat{r}(u,v,d))},
    $$
    where $d_0^2/(d+d_0)^2$ reflexes the inverse square law w.r.t. the reference plane, $\hat{r}(u,v,d)$ is the radiological path between the source and the interaction point:
    $$
    \hat{r}(u,v,d)=\frac{r_0}{d_0}\int_{(u,v,-d_0)}^{(u,v,d)}\eta(u,v,t)dt
    $$
* Terma for a parallel beam

    We assume that the electron density $\eta(u,v,d)$ is invariant w.r.t. $d_0$. Take the limit $d_0 \to +\infty$, we have
    $$
    T^*(u,v,d) = \exp{(-\hat{r}^*(u,v,d))},
    $$
    where $\hat{r}^*(u,v,d) = \int_{(u,v,-d_0)}^{(u,v,d)} \eta(u,v,t)dt$.
* Dose transformation

    To convert the BEV coordinates to the original Cartesian coordinates:
    $$
    \begin{align*}
    x &= \frac{(d+d_0)u}{d_0}, \\
    y &= \frac{(d+d_0)v}{d_0}, \\
    z &= d.
    \end{align*}
    $$

## CCCS kernel derivation
Consider the energy deposition along a collapsed cone, and assume the particle transmission to be rectilinear. According to [Lu et al](https://doi.org/10.1088/0031-9155/50/4/007), the dose deposition along this collapsed cone direction is formulated as:
$$
\tag{1}
D(x)=\int_0^x T(s)k(r(s;x))\eta(s)ds,
$$
where $r(s;x)$ is the radiological path length between the source $s$ and the destination $x$:
$$
\begin{align*}
r(s;x) &= \int_{t=s}^{t=x}\eta(t)dt, \\
\frac{\partial r(s;x)}{\partial s} &= -\eta(s).
\end{align*}
$$
For simplicity, we define $r_x(s) = r(s;x)$, and we have $\frac{dr_x(s)}{ds} = -\eta(s)$. Then we can rewrite equation 1 as:
$$
\tag{2}
D(x) = \int_x^0 T(s)k(r_x(s))dr_x(s).
$$
When $r_x(s)$ is invertible,
$$
\tag{3}
D(x)=\int_0^{r(0;x)} T(s(r_x))k(r_x)dr_x
$$
We then further define
$$
\begin{align*}
K(r_x) &= \int_0^{r_x}k(t)dt, \\
dK(r_x) &= k(r_x)
\end{align*}
$$
So $D(x)$ can be further rewritten as:
$$
\tag{4}
D(x) = \int_x^0 T(s)dK(r_x(s)).
$$
We then segment the path (from $x$ to $0$) into intervals, each being the intersection of the path with a voxel:
$$
\tag{5}
D(x) = \sum_{j=0}^{i-1} \int_{x_{j+1}}^{x_j} T(s)dK(r_x(s)),
$$
with $x_i = x$. This sum can be split into two parts:
$$
\tag{6}
\begin{align*}
D(x) &= \int_{x}^{x_{i-1}}T(s)dK(r_x(s)) + \sum_{j=0}^{i-2}\int_{x_{j+1}}^{x_j}T(s)dK(r_x(s)) \\
&=T_{i-1}\left( K(r_x(x_{i-1})) - K(r_x(x)) \right) + \sum_{j=0}^{i-2}T_j\left((K(r_x(x_j))-K(r_x(x_{j+1}))) \right) \\
&=T_{i-1}K(r_x(x_{i-1})) + \sum_{j=0}^{i-2}T_j\left( K(r_x(x_j)) - K(r_x(x_{j+1}))\right) \\
&= T_{i-1}K(r(x_{i-1};x)) + \sum_{j=0}^{i-2} T_j(K(r(x_j; x)) - K(r(x_{j+1}; x)))
\end{align*}
$$
The dose to a voxel can be viewed as the average of dose through the line segment, at which the path intersects with the voxel. Here we define the two endpoints of the line segment to be $x_{i-1}$ and $x_i$. Then the average dose can be written as:
$$
\tag{7}
\begin{split}
D_{i-1} =& \frac{1}{r(x_{i-1}; x_i)}\int_{x_{i-1}}^{x_i}D(x)\eta(x)dx \\
=&\frac{1}{r(x_{i-1}; x_i)}\left(T_{i-1}\int_{x_{i-1}}^{x_i} K(r(x_{i-1}; x))\eta(x)dx \right)+ \\

& \frac{1}{r(x_{i-1}; x_i)} \left(\sum_{j=0}^{i-2} T_j\left( \int_{x_{i-1}}^{x_i}K(r(x_j;x)) \eta(x) dx-\int_{x_{i-1}}^{x_i}K(r(x_{j+1};x))\eta(x)dx \right) \right) \\

=& \frac{1}{r(x_{i-1}; x_i)} \left( T_{i-1}\int_{x_{i-1}}^{x_i} K(r(x_{i-1};x))dr(x_{i-1};x) \right) + \\
& \frac{1}{r(x_{i-1}; x_i)} \left( \sum_{j=0}^{i-2} T_j \left( \int_{x_{i-1}}^{x_i}K(r(x_j;x))dr(x_j;x) - \int_{x_{i-1}}^{x_i}K(r(x_{j+1};x))dr(x_{j+1};x) \right) \right)
\end{split}
$$
Let's then define
$$
C(r) = \int_0^r K(t)dt,
$$
then equation 7 can be rewritten as:
$$
\tag{8}
D_{i-1} = \frac{1}{r(x_{i-1};x_i)}\left( T_{i-1}C(r(x_{i-1};x_i)) + \sum_{j=0}^{i-2}T_j \left( C(r(x_j;x_i)) - C(r(x_j; x_{i-1})) - C(r(x_{j+1};x_i)) + C(r(x_{j+1};x_{i-1})) \right) \right).
$$

### Exponential kernel approximation
If we approximate the kernel $k(r)$ with the sum of two exponential terms:
$$
\tag{9}
k(r) = Ae^{-ar} + Be^{-br},
$$
then,
$$
\tag{10}
\begin{split}
K(r) &= \int_0^r (Ae^{-at} + Be^{-bt})dt = \frac{A}{a}(1-e^{-ar}) + \frac{B}{b}(1-e^{-br}),
\end{split}
$$

$$
\tag{11}
C(r) = \int_0^r K(t)dt = \frac{A}{a}\left( r - \frac{1}{a}(1-e^{-ar}) \right) + \frac{B}{b}\left( r - \frac{1}{b}(1-e^{-br}) \right),
$$

$$
\tag{12}
\begin{split}
C(r(x_j&;x_i)) - C(r(x_j;x_{i-1})) \\
=& \frac{A}{a}r(x_{i-1};x_i) + \frac{A}{a^2}\left( e^{-ar(x_j;x_i)} -e^{-ar(x_j;x_{i-1})} \right) \\
& + \frac{B}{b}r(x_{i-1};x_i) + \frac{B}{b^2}\left( e^{-br(x_j;x_i)} -e^{-br(x_j;x_{i-1})} \right) \\
=& \frac{A}{a}r(x_{i-1};x_i) + \frac{A}{a^2}e^{-ar(x_j;x_{i-1})}(e^{-ar(x_{i-1};x_i)}-1) \\
& + \frac{B}{b}r(x_{i-1};x_i) + \frac{B}{b^2}e^{-br(x_j;x_{i-1})}(e^{-br(x_{i-1};x_i)}-1),
\end{split}
$$

$$
\tag{13}
\begin{split}
C(r(x_j&;x_i)) - C(r(x_j;x_{i-1})) - C(r(x_{j+1};x_i)) + C(r(x_{j+1};x_{i-1})) \\
=& \frac{A}{a^2} (e^{-ar(x_j;x_{i-1})} - e^{-ar(x_{j+1};x_{i-1})})(e^{-ar(x_{i-1};x_i)}-1) \\
&+ \frac{B}{b^2} (e^{-br(x_j;x_{i-1})} - e^{-br(x_{j+1};x_{i-1})})(e^{-br(x_{i-1};x_i)}-1) \\
=& \frac{A}{a^2} e^{-ar(x_j;x_{i-1})} (1 - e^{ar(x_j;x_{j+1})})(e^{-ar(x_{i-1};x_i)}-1) \\
&+ \frac{B}{b^2} e^{-br(x_j;x_{i-1})} (1 - e^{br(x_j;x_{j+1})})(e^{-br(x_{i-1};x_i)}-1)
\end{split}
$$

$$
\tag{14}
D_{i-1} = \frac{1}{r(x_{i-1};x_i)} \left( T_{i-1}C(r(x_{i-1};x_i)) +\sum_{j=0}^{i-2}T_j(C_{j,i}-C_{j,i-1}-C_{j+1,i}+C_{j+1,i-1}) \right)
$$

Again, for simplicity, we denote $p_{i-1}=r(x_{i-1};x_i)$, and only include the terms associated with $A$ and $a$. We firstly examine the coefficient of $T_{i-1}$:
$$
\tag{15}
\begin{align*}
& \frac{1}{p_{i-1}}\left( \frac{A}{a} \left( p_{i-1} - \frac{1}{a}(1-e^{-ap_{i-1}}) \right) \right) \\
&= \frac{A}{a}\left( 1-\frac{1-e^{-ap_{i-1}}}{ap_{i-1}} \right).
\end{align*}
$$

Then we consider the remaining part. For $D_{i-1}$, the remaining part is
$$
\tag{16}
\begin{split}
&\frac{1}{p_{i-1}}\sum_{j=0}^{i-2} T_j \cdot \frac{A}{a^2}e^{-ar(x_j;x_{i-1})} (1 - e^{ar(x_j;x_{j+1})}) (e^{-ar(x_{i-1};x_i)} -1) \\
=&\frac{1}{p_{i-1}} \frac{A}{a^2} (1-e^{-ap_{i-1}}) \sum_{j=0}^{i-2} T_j e^{-ar(x_j; x_{i-1})} (e^{ap_j}-1)
\end{split}
$$

Let us define
$$
\begin{align*}
X_i &= \left\{
\begin{aligned}
&0, && \text{if }i=0,\\
& \sum_{j=0}^{i-1} T_j e^{-ar(x_j; x_i)} (e^{ap_j}-1) && \text{if }i\geq 1.
\end{aligned}
\right.
\end{align*}
$$
And
$$
g_i = \frac{1-e^{-ap_i}}{ap_i}
$$

So we have
$$
D_i = \frac{A}{a}[ (1-g_i)T_i + g_iX_i ].
$$

We also have
$$
\begin{split}
X_{i+1} &= \sum_{j=0}^{i} T_je^{-ar(x_j; x_{i+1})} (e^{ap_j}-1) \\
&=T_i e^{-ar(x_i;x_{i+1})} (e^{ap_i} -1) + \sum_{j=0}^{i-1}T_je^{-ar(x_j;x_{i+1})}(e^{ap_j}-1) \\
&=T_i e^{-ap_i}(e^{ap_i} - 1) + e^{-ap_i} \sum_{j=0}^{i-1}T_j e^{-ar(x_j; x_i)}(e^{ap_i}-1) \\
&=T_i (1-e^{-ap_i}) + e^{-ap_i}X_i.
\end{split}
$$