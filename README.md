### hello md


# 基于区域的机器人群阵型控制
>Region-based shape control for a swarm of robots



`Chien Chern Cheah`, `Saing Paul Hou`, and `Jean Jacques E. Slotine`

<p>
<details>
<summary>Chien Chern Cheah</summary>

School of Electrical and Electronic Engineering, Nanyang Technological University, Block S1, Nanyang Avenue, S(639798), Republic of Singapore

Chien Chern Cheah was born in Singapore. He received B.Eng. degree in Electrical Engineering from National University of Singapore in 1990, M.Eng. and Ph.D. degrees in Electrical Engineering, both from Nanyang Technological University, Singapore, in 1993 and 1996, respectively.

From 1990 to 1991, he worked as a design engineer in Chartered Electronics Industries, Singapore. He was a research fellow in the Department of Robotics, Ritsumeikan University, Japan from 1996 to 1998. He joined the School of Electrical and Electronic Engineering, Nanyang Technological University as an assistant professor in 1998. Since 2003, he has been an associate professor in Nanyang Technological University. In November 2002, he received the oversea attachment fellowship from the Agency for Science, Technology and Research (A*STAR), Singapore to visit the Nonlinear Systems laboratory, Massachusetts Institute of Technology.

He was the program chair of the International Conference on Control, Automation, Robotics and Vision 2006. He has served as an associate editor of the IEEE Robotics and Automation Society Conference Editorial Board since 2007.

</details>
</p>

<p>
<details>
<summary>Saing Paul Hou</summary>

School of Electrical and Electronic Engineering, Nanyang Technological University, Block S1, Nanyang Avenue, S(639798), Republic of Singapore

Saing Paul Hou was born in Kandal, Cambodia in 1982. He received B.Eng. degree with first class honor in Electrical and Electronic Engineering from Nanyang Technological University, Singapore in 2006. He was the recipient of Control Chapter Book Prize and Motorola Book Prize in 2006. He has been pursuing his Ph.D. degree at Nanyang Technological University, Singapore since 2006. His research interests include formation control of multi-robot systems and adaptive control.

</details>
</p>

<p>
<details>
<summary>Jean Jacques E. Slotine</summary>

Nonlinear Systems Laboratory, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA

Jean-Jacques E. Slotine was born in Paris in 1959, and received his Ph.D. from the Massachusetts Institute of Technology in 1983. After working at Bell Labs in the computer research department, in 1984 he joined the faculty at MIT, where he is now Professor of Mechanical Engineering and Information Sciences, Professor of Brain and Cognitive Sciences, and Director of the Nonlinear Systems Laboratory. He is the co-author of the textbooks “Robot Analysis and Control” (Wiley, 1986) and “Applied Nonlinear Control” (Prentice-Hall, 1991). Prof. Slotine was a member of the French National Science Council from 1997 to 2002, and is a member of Singapore’s A*STAR Sign Advisory Board.

</details>
</p>

https://www.sciencedirect.com/science/article/pii/S0005109809003215

## Abstract

本文介绍了一种用于机器人群的基于区域的形状控制器。在该控制方法中，机器人在期望区域内组团移动，同时保持它们之间的最小距离。通过选择适当的目标函数，可以形成各种形状的期望区域。组团中的机器人只需要与邻近的机器人通讯，而不是与整个团体进行通信。机器人在组内没有特定的身份或角色。因此，所提出的方法不需要限定机器人在该区域内的特定顺序或位置，故一群机器人可以形成不同的构造。本文采用类Lyapunov函数对多机器人系统进行收敛性分析。其仿真结果说明了本文提出的控制器的性能。
>This paper presents a region-based shape controller for a swarm of robots. In this control method, the robots move as a group inside a desired region while maintaining a minimum distance among themselves. Various shapes of the desired region can be formed by choosing the appropriate objective functions. The robots in the group only need to communicate with their neighbors and not the entire community. The robots do not have specific identities or roles within the group. Therefore, the proposed method does not require specific orders or positions of the robots inside the region and yet different formations can be formed for a swarm of robots. A Lyapunov-like function is presented for convergence analysis of the multi-robot systems. Simulation results illustrate the performance of the proposed controller.
