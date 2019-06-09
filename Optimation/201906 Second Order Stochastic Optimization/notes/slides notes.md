# slides notes


## 0. 要求

1. 背景介绍（大于1/3）
2. 算法介绍
3. 理论介绍

+ 总时间不得大于12min，全部时间大于等于15min

## 1. 背景介绍

1. 一般都用什么方法？
   + 一阶随机方法——这很大是由于它能够在大规模的模型训练中提供可负担的线性计算成本。
     + 线性时间是指每次迭代，花费线性时间（计算梯度）
     + **什么是快速一阶方法？**
   + 仅使用梯度信息的优化算法称为一阶优化算法，如梯度下降；
   + 使用hessian矩阵的优化算法称为二阶优化算法，如牛顿法。
2. 有什么优点？
   + 每次迭代具有高效的复杂度？
   + 它的计算成本是我们可以负担的
3. 有什么缺点？（or可以改进的地方）
   + 二阶方法比它能提供更快的收敛速度
4. 那为什么不用二阶方法呢？
   + 因为计算二阶信息的成本比较高，虽然迭代次数更短，但是计算时间更长（比如hessian）
   + 主要有两个缺陷，一个是需要计算Hessian矩阵，需要O(np^2) 的复杂度，另外一个便是计算Hessian的逆矩阵需要O(p^3 )的复杂度。而在p维度很高的时候，传统二阶优化法显然不能够适用。
5. 我们要怎么改进能将二阶方法应用进来呢？
   + 用随机采样做estimator来估计hessian 矩阵，从而得到一个有效的Newton step；这个estimator是从Tylor expansion中得到的
6. 研究的baseline
   + 二阶方法：Newton法
     + 理论性质？？？
   + 目标函数：经验⻛险最小化(ERM)问题：最小化平均训练误差（经验分布替代了真实分布）
   + 函数形式GLM（m个损失函数的平均，再加上一个正则化的惩罚项）
7. 流程
   1. 
   2. 我们⾸先提出必要的定义，符号和约定**conventions**。
   3. 在**Section 3**中，我们描述了了我们对LiSSA的估计量量，并对LiSSA的收敛性保证进行了证明。
   4. 在Section 4中，在我们给出了一个一阶方法与牛顿法耦合**couple**的⼀一般化程序**generic procedure**
   5. 我们在**Section 5**给出了了LiSSA-Sample和相关的快速⼆二次求解器器**fast quadratic solver**。
   6. 然后，我们在**Section 6**中给出了了关于⾃和谐函数**self-concordant functions**的结果。
   7. 最后，在**Section 7**中对LiSSA进⾏了实验评估。

## 2. 算法介绍

1. LiSSA(***Algorithm 1***)
   1. 第一阶段，用任何有效的一阶算法FO来将函数值缩小**到我们的算法可以线性收敛的状态**
      + 这里的$T_1$在实际应用中会设置它是一阶算法要达到xx精度所需要的总时间（为什么开始又说是$T_1$ steps）
   2. 第二阶段，用我们的开始来进一步在第一阶段给出的$x_1$的基础上，更新到$x_{T+1}$
      1. 根据给定的输入，一共要执行$T$次更新
      2. 其中每一次都是一个$S_1$个量的平均值（）
      3. 是什么量呢？是我们要迭代的次数，tylor展开
2. LiSSA的核心思想是利用tylor展开式构造一个对hessian的逆的估计量。（那么只要有hessian就可以了）
   + 级数中包含越多的项，对矩阵的逆的估计越无偏
3. LiSSA-Sample是利用牛顿法对二次函数的观点——是不是特殊考虑二次型的意思？——减弱到二次子问题
4. 
5. 特点：但是LiSSA则是通过对Hessian的逆的泰勒展开式，得出Hessian逆与Hessian的等式，再通过**对Hessian的进行采样估计**，来**直接估计出Hessian的逆**。LiSSA的思路完全不同于上述两种算法，同时LiSSA算法采用Hessian-vector product对形式因而对广义线性模型具有更加明显的优势。

## 5. 现有的疑问

1. 什么**是条件数**？√
   + 这个应该是一个比较普遍的概念，有说hessian矩阵的条件数是衡量这些二阶导数的变化范围
   + hessian矩阵的最大特征值和最小特征值的比
2. 什么叫中级⼆次**intermediate quadratic**问题
3. 有效地利用了子问题的二次性质？


===============

 **LiSSA** 

- **Natural** Stochastic Newton Method
- Every iteration in **O(d)** time. Linear in **Input Sparsity** 每次迭代是线性时间O(d)，尤其是当输入是稀疏的时候O(s)，
  - 如果损失函数是GLM，Hessian isoftheform−g x,y 𝑥𝑥/   matrix-vector product → vector-vector product  **input sparsity time** 
- 一共要xx次迭代——所以是线性收敛的（<= 1/2）𝐥𝐨𝐠 𝟏𝝐 iterations (Linear Convergence )
- 更好的依赖于控制数？Better dependence on the **condition number** 
  - $\kappa_l \leq \kappa$
- 比已知的一阶算法有更好的局部收敛性better local convergence than known FO methods empirically 

• **LiSSA ++** 

- 更好的收敛数**better condition number** 
- 和矩阵采样等技术结合，对于m>>d的情况，具有最好的运行时间——主要是加和维数的变化？Couple with Matrix Sampling/ Sketching techniques - **Best known running time** for 𝒎 ≫ 𝒅 

1. 可能存在的问题：
   1. 假设比较强，alpha beta强凸，实际上就是说hessian矩阵的范围，还说是lipschitz连续的；——实际上不一定这么好，甚至nn是non-convex的

============

# outline

1. （slides 2）整个文章的核心思想是他提出了lissa这个算法，而后面的lissa-sample是他将他的lissa同文献中已经给出的一种先进的matrix sampling的结合起来，从而进一步优化。所以我讲的核心会更加围绕lissa，这个是我觉着本文最核心最优越的想法。
2. （slides 3 background）
   1. 我们优化问题一般可以用哪些方法？
      1. 一阶，用梯度信息
      2. 二阶，进一步用hessian信息。
   2. 但是为什么一般用的都是一阶算法呢？
      1. 我们以牛顿法为例
      2. 需要算hessian，o(d^2)，m是因为问题中的函数是m个hessian，这个不是大问题
      3. 求矩阵的逆，一般是3次，根据文献现在最优化的是w=2.37左右
   3. 所以我们进一步就是想优化这个计算量的问题，让它能够用到数据量很大的ml问题中。
   4. **可能存在的问题：**
      1. 为什么比一阶算法好？
         - 曲率+saddle
3. （slides 4 background：baseline）
   1. 优化目标：经验风险最小化问题，对f性质做了假设
   2. 从newton法出发，优化计算量
   3. 我这里没给出函数性质的假设，用到的时候会说一下，我觉着比较有必要先解释的是condition number这个概念
   4. 它表示的是hessian矩阵最大和最小的特征值的比，衡量相对输入值改变而言，输出值在最坏条件下的变化，太大了叫病态；在这篇文章中，也可以定义为上下用二次曲面逼近的，这就是一个感观上的认识
   5. 这里面还有几种类似的定义，它算法的一个优点实际上也是跟它对condition number的优化有关。具体在讲定理的时候说
4. （slides 5 lissa：main idea）
   1. 为了解决矩阵逆计算的问题，做了一个tylor展开
      - 什么时候级数收敛？
        - 范数小于1
        - 论文中这个是通过scale f(x)，整体缩放hessian达到这个要求
   2. aj是第j项tylor，计算一下可以得到这个。
   3. 我们根据这个定义了一个estimator，从单位阵开始，逐次迭代
   4. 注意到，aj在j趋近于无穷的时候是逼近矩阵的逆的；因为我迭代的深度不同，会存在误差
5. 为什么不给出一个终止条件呢？
   1. 它是通过设定参数，这个参数满足xx条件的时候，以多大的概率满足我们要的精度，实际上已经将终止的精度要求包含在内了
6. 如果不强凸会怎么样？
   - 为什么一般的凸不行呢？
7. lissa和lissa-sample的区别？
   - 对a进行了sample，
8. 达到二阶收敛速率了嘛？
   - 没有，线性的
   - 为什么连超线性都没有呢？
     - 因为他的限制，我s2限制，要我不能特别精确地估计，但是我觉着还是有好处的
     - lissa是一阶的，lissa-sample是二阶的，但是它没有明确地写出来，证明中的过程表示了