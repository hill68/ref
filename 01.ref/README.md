# 基于区域的机器人群阵型控制.md
>Region-based shape control for a swarm of robots

\begin{equation}
\int\limits_{-\infty}^\infty f(x) \delta(x - x_0) dx = f(x_0)
\end{equation}


\begin{equation}
M_{i}(x_{i})\ddot{x}_{i} + D_{i}(x_{i}) \dot{x}_{i} + g_{i}(x_{i}) = u_{i}
\end{equation}

\begin{equation}
C_{i}(x_{i}  \dot{x}_{i})\dot{x}_{i}
\end{equation}

\begin{equation}
D_{i}(x_{i}) \dot{x}_{i}
\end{equation}

\begin{equation}
g_{i}(x_{i})=u_{i}
\end{equation}

\begin{equation}
M_{i}(x_{i})\ddot{x}_{i}+C_{i}(x_{i}, \dot{x}_{i})\dot{x}_{i}+D_{i}(x_{i}) \dot{x}_{i}+g_{i}(x_{i})=u_{i}
\end{equation}

Click **Generate PDF** on the rightside ( $ N $ ) panel to output pdf file $n$.

我们考虑一组 ( `$N$` ) 个启动的移动机器人，其具有 $n$ 个自由度的第 \[ i \] 个机器人的动力学模型可以描述为（Fossen，1994; Slotine＆Li，1991）：
