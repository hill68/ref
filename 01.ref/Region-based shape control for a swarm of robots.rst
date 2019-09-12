.. _header-n3597:

2 基于区域的机器人蜂群阵型控制
===============================



   Region-based shape control for a swarm of robots

``Chien Chern Cheah``, ``Saing Paul Hou``, and
``Jean Jacques E. Slotine``

https://www.sciencedirect.com/science/article/pii/S0005109809003215

.. _header-n3604:

Abstract
--------

本文介绍了一种用于机器人蜂群的基于区域的形状控制器。在该控制方法中，机器人在期望区域内组团移动，同时保持它们之间的最小距离。通过选择适当的目标函数，可以形成各种形状的期望区域。组团中的机器人只需要与邻近的机器人通讯，而不是与整个团体进行通信。机器人在组内没有特定的身份或角色。因此，所提出的方法不需要限定机器人在该区域内的特定顺序或位置，故一群机器人可以形成不同的构造。本文采用类Lyapunov函数对多机器人系统进行收敛性分析。其仿真结果说明了本文提出的控制器的性能。

   This paper presents a region-based shape controller for a swarm of
   robots. In this control method, the robots move as a group inside a
   desired region while maintaining a minimum distance among themselves.
   Various shapes of the desired region can be formed by choosing the
   appropriate objective functions. The robots in the group only need to
   communicate with their neighbors and not the entire community. The
   robots do not have specific identities or roles within the group.
   Therefore, the proposed method does not require specific orders or
   positions of the robots inside the region and yet different
   formations can be formed for a swarm of robots. A Lyapunov-like
   function is presented for convergence analysis of the multi-robot
   systems. Simulation results illustrate the performance of the
   proposed controller.

.. _header-n3608:

Keywords
--------

Shape control; Co-operative control; Region following; Trajectory
tracking; Adaptive control; Lyapunov stability

.. _header-n3610:

1. INTRODUCTION
---------------

近几十年来，多机器人系统的协同控制（Murray，2007）一直是广泛研究的论题。在基于行为的多机器人控制（Balch＆Arkin，1998;
Lawton，Beard，＆Young，2003; Reif＆Wang，1999;
Reynolds，1987）一文中，学者成功控制机器人个体作出了一整套期望的动作。文章通过定义所有行为的相对重要性，从而形成了多机器人系统的一些整体行为。然而，这种方法的主要问题是难以在数学上分析整个系统，从而透彻理解机器人蜂群的控制问题。与此同时，此方法也不可能得知系统是否收敛到了理想的状态。

   Cooperative control of multi-robot systems (Murray, 2007) has been
   the subject of extensive research in recent decades. In
   behavior-based control of multiple robots (Balch & Arkin, 1998;
   Lawton, Beard, & Young, 2003; Reif & Wang, 1999; Reynolds, 1987), a
   desired set of behaviors is implemented onto individual robots. By
   defining the relative importance of all the behaviors, the overall
   behavior of the multi-robot system is formed. The main problem of
   this approach is that it is difficult to analyze the overall system
   mathematically to gain insights into the control problems. It is also
   not possible to show that the system converges to a desired
   formation.

在领导者跟踪方法中（Consolini，Morbidi，Prattichizzo，＆Tosques，2008;
Das et al。，2002; Desai，Kumar，＆Ostrowski，2001;
Dimarogonas，Egerstedt，＆Kyriakopoulos，2006; Fredslund＆Mataric，2002;
Ji， Ferrari-Trecate，Egerstedt，＆Buffa，2008;
Ogren，Egerstedt，＆Hu，2002;
Wang，1991），领导者被确定，并且追随者被设定为跟随他们各自的领导者。

   In leader-following approach (Consolini, Morbidi, Prattichizzo, &
   Tosques, 2008; Das et al., 2002; Desai, Kumar, & Ostrowski, 2001;
   Dimarogonas, Egerstedt, & Kyriakopoulos, 2006; Fredslund & Mataric,
   2002; Ji, Ferrari-Trecate, Egerstedt, & Buffa, 2008; Ogren,
   Egerstedt, & Hu, 2002; Wang, 1991), the leaders are identified and
   the followers are defined to follow their respective leaders.

通常，跟随者需要保持自身与领导者之间的期望距离与方向，因此而形成的阵型是死板的。为了解决这个问题，文章提出了几种方法，比如允许追随者相对于领导者的位置可以灵活一些（Consolini等，2008;
Dimarogonas等，2006;
Ji等，2008）。Consolini等人（2008）认为，跟随者可以沿着以领导者位置为中心的圆改变其位置，不过跟随者和领导者之间的距离仍然是固定的。

   Generally, the followers need to maintain a desired distance and
   orientation to their respective leaders and hence the formation is
   rigid. To alleviate this problem, several approaches are proposed to
   allow some flexibility on the positions of the followers with respect
   to the leaders (Consolini et al., 2008; Dimarogonas et al., 2006; Ji
   et al., 2008). In Consolini et al. (2008), the follower can vary its
   position along a circular arc centered at the leader position but the
   distance between the follower and the leader is still fixed.

而Dimarogonas（2006）和Ji等人（2008）认为，几个领导者首先应当建立起静态阵型，然后再命令跟随者留在领导者形成的多面体内。然而，多面体的形状取决于领导者的数量。部署太少的领导者限制了团队最后呈现的阵型，而太多的领导者增加了控制问题的复杂性。这是因为在这个问题情形下，必须先为领导者建立一个编队控制器以确保其形成多面体阵型。相比基于行为的多机器人控制而言，领导者跟踪方法更容易分析，但最显著的问题是领导者机器人的崩溃导致整个系统的崩溃。

   In Dimarogonas et al. (2006) and Ji et al. (2008) several leaders are
   first used to establish a static formation and the followers are then
   commanded to stay within the polytope formed by the leaders. However,
   the shape of the polytope depends on the number of leaders. The
   deployment of too few leaders limits the shape of the group while too
   many leaders increases the complexity of the control problem since it
   is necessary to first establish a formation controller for the
   leaders themselves to form the polytope. The leader–following
   approach is easier to analyze but one obvious problem is that the
   failure of one robot (i.e. leader) leads to the failures of the
   entire system.

在虚拟结构方法中（Egerstedt＆Hu，2001; Lewis＆Tan，1997;
Ren＆Beard，2004），整个阵型被认为是单个实体，并且阵型期望的运动被分配给结构。这种阵型组织方法非常严格，因为系统中机器人之间在运动过程中必须严格保持一定的几何关系，因此，阵型通常不可能随时间变化。除此之外，避障也是一个问题。很显然，虚拟结构方法不适合控制大量机器人，因为随着机器人数量的增加，机器人之间的约束关系变得更加复杂。

   In the virtual structure method (Egerstedt & Hu, 2001; Lewis & Tan,
   1997; Ren & Beard, 2004), the entire formation is considered as a
   single entity and desired motion is assigned to the structure. The
   formation in this approach is very rigid as the geometric
   relationship among the robots in the system must be rigidly
   maintained during the movement. Therefore, it is generally not
   possible for the formation to change with time, and obstacle
   avoidance is also a problem. The virtual structure approaches are not
   suitable for controlling a large group of robots because the
   constraint relationships among robots become more complicated as the
   number of robots in the group increases.

控制一组机器人以形成阵型的另一种方法是使用约束函数（Ihle，Jouffroy，＆Fossen，2006;
Zhang＆Hu，2008;
Zou，Pagilla，＆Misawa，2007）。这种方法与虚拟结构方法具有类似的问题。由于机器人相互间的约束关系的复杂程度随着机器人数量的增加而增加，因此这种方法也不适合控制大量机器人。

   Another method to control a group of robots to establish a formation
   is by using constraint functions (Ihle, Jouffroy, & Fossen, 2006;
   Zhang & Hu, 2008; Zou, Pagilla, & Misawa, 2007). This approach has a
   similar problem as the virtual structure method because the
   complexity of the constraint relationships increases as the number of
   robots increases and hence is also not suitable for controlling a
   large group of robots.

为了控制大量机器人，通常使用潜在场方法（Gazi，2005;
Leonard＆Fiorelli，2001; Olfati-Saber，2006;
Pereira＆Hsu，2008）。然而，这种方法难以形成整体的阵型，因为机器人仅被命令以组为单位呆在一起，以及避免它们之间的碰撞。

   To control a large group of robots, the potential field approach
   (Gazi, 2005; Leonard & Fiorelli, 2001; Olfati-Saber, 2006; Pereira &
   Hsu, 2008) is normally used. However, it is difficult to form a
   desired shape for the swarm system as the robots are only commanded
   to stay close together as a group and avoid collision among
   themselves.

Belta和Kumar（2004）提出了一种控制方法，可以让大量机器人沿着指定路径移动。然而，由于整个组的阵型取决于组中机器人的数量，因此该控制策略也无法控制所需阵型。对于大量机器人，阵型固定为椭圆形，而对于少数机器人，阵型固定为矩形。

   Belta and Kumar (2004) propose a control method for a large group of
   robots to move along a specified path. However, this proposed control
   strategy also has no control over the desired formation since the
   shape of the whole group is dependent on the number of the robots in
   the group. For large numbers of robots, the formation is fixed as an
   elliptical shape, whereas for a small number of robots the formation
   is fixed as a rectangular shape.

在本文中，我们为机器人蜂群设计了一个基于区域的控制器。在我们提出的控制方法中，组中的每个机器人作为一组（全局目标）在移动区域内运动，并且同时保持彼此的最小距离（局部目标）。期望的区域可以被指定为各种形状，因此可以形成不同的形状和阵型。该组中的机器人只需要与邻近的机器人沟通，而不是和整个组群沟通。机器人在组内没有特定的身份或角色。因此，我们所提出的方法不需要限定机器人在区域内有特定次序或位置，从而由给定的一组机器人可以形成不同的阵型。在阵型控制系统的稳定性分析中，本文也考虑了机器人的动力学模型。此外，倘若任何机器人都可以进入或离开阵型、且不影响其他机器人，说明该系统是可扩展的。Lyapunov理论用于证明多机器人系统的稳定性。其仿真结果用于说明所提出的阵型控制器的性能。

   In this paper, we propose a region-based controller for a swarm of
   robots. In our proposed control method, each robot in the group stays
   within a moving region as a group (global objective) and, at the same
   time, maintains a minimum distance from each other (local objective).
   The desired region can be specified as various shapes, hence
   different shapes and formations can be formed. The robots in the
   group only need to communicate with their neighbors and not the
   entire community. The robots do not have specific identities or roles
   within the group. Therefore, the proposed method does not require
   specific orders or positions of the robots inside the region and
   hence different shapes can be formed by a given swarm of robots. The
   dynamics of the robots are also considered in the stability analysis
   of the formation control system. The system is scalable in the sense
   that any robot can move into the formation or leave the formation
   without affecting the other robots. Lyapunov theory is used to show
   the stability of the multi-robot systems. Simulation results are
   presented to illustrate the performance of the proposed shape
   controller.


.. _header-n3638:

2. Region-based shape controls
------------------------------

我们考虑一\ :math:`N`\ 个启动的移动机器人，其具有\ :math:`n`\ 个自由度的第
:math:`i` 个机器人的动力学模型可以描述为（Fossen，1994;
Slotine＆Li，1991）

   We consider a group of N fully actuated mobile robots whose dynamics
   of the ith robot with n degrees of freedom can be described as
   (Fossen, 1994; Slotine & Li, 1991)

.. math:: M_{i}(x_{i})\ddot{x}_{i}+C_{i}(x_{i},\dot{x}_{i})\dot{x}_{i}+D_{i}(x_{i}) \dot{x}_{i}+g_{i}(x_{i})=u_{i}\tag{1}
