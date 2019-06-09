# Simple Notes

## 0. Pre要求

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
   + 目标函数：经验⻛险最小化(ERM)问题：最小化平均训练误差（经验分布替代了真实分布）
   + 函数形式GLM（m个损失函数的平均，再加上一个正则化的惩罚项）

## 2. 算法简介

1. LiSSA(***Algorithm 1***)
	+ 核心思想是利用tylor展开式构造一个对hessian的逆的估计量。（那么只要有hessian就可以了）
   + 级数中包含越多的项，对矩阵的逆的估计越无偏
2. LiSSA-Sample是利用牛顿法对二次函数的观点——减弱到二次子问题
	+ 本质上是LiSSA到LiSSA-Quad再到LiSSA-Sample
	+ LiSSA-Sample相当于是LiSSA再加上文献中的Matrix Sample技术
3. 特点：但是LiSSA则是通过对Hessian的逆的泰勒展开式，得出Hessian逆与Hessian的等式，再通过**对Hessian的进行采样估计**，来**直接估计出Hessian的逆**。LiSSA的思路完全不同于上述两种算法，同时LiSSA算法采用Hessian-vector product对形式因而对广义线性模型具有更加明显的优势。

## 3. 核心优缺点
1.  **LiSSA** 
    + **Natural** Stochastic Newton Method
    + Every iteration in **O(d)** time. Linear in **Input Sparsity** 每次迭代是线性时间O(d)，尤其是当输入是稀疏的时候O(s)，
    + 如果损失函数是GLM，Hessian isoftheform−g x,y 𝑥𝑥/   matrix-vector product → vector-vector product  **input sparsity time** 
    + 一共要xx次迭代——所以是线性收敛的（<= 1/2）𝐥𝐨𝐠 𝟏𝝐 iterations (Linear Convergence )
    + 更好的依赖于控制数？Better dependence on the **condition number** 
    + $\kappa_l \leq \kappa$
    + 比已知的一阶算法有更好的局部收敛性better local convergence than known FO methods empirically 

2. **LiSSA ++** 
   + 更好的收敛数**better condition number** 
   + 和矩阵采样等技术结合，对于m>>d的情况，具有最好的运行时间——主要是加和维数的变化？Couple with Matrix Sampling/ Sketching techniques - **Best known running time** for 𝒎 ≫ 𝒅 

3. 可能存在的问题：
   1. 假设比较强，alpha beta强凸，实际上就是说hessian矩阵的范围，还说是lipschitz连续的；——实际上不一定这么好，甚至nn是non-convex的
   2. 貌似网上有人应用说不收敛
   3. 理论上讲，lissa-sample应该是二阶收敛速的，且m足够大的时候是最快的。但是文中并没有给出实验证明；后来发现v5版的论文给出了，但是实际效果并不一定有acc的一阶算法好

## 4. Key Questions & My thoughts

1. condition number代表什么？

   + 定义：hessian矩阵最大和最小的特征值的比
   + 衡量相对输入值改变而言，输出值在最坏条件下的变化（太大了叫病态）；
   + 在这篇文章中，也可以定义为上下用二次曲面逼近的，这就是一个感观上的认识
   + 现场让推了一下local和global的大小区别，还是比较显然的
   + 我觉着sample比较小也是显然的

2. 为什么不给出一个终止条件呢？

   - 它是通过设定参数，这个参数满足xx条件的时候，以多大的概率满足我们要的精度，实际上已经将终止的精度要求包含在内了
   - 要是想改的话也可以改成这样

3. 如果不强凸会怎么样？为什么一般的凸不行呢？

   - 强凸是为了后面那个证明，见7

4. lissa和lissa-sample的区别？

   1. 从收敛速率方向：
      - 个人理解，lissa的证明只用到了hessian矩阵的Lipschitz性质，lissa-sample进一步在quad基础上应用了$\alpha$strongly convex和$\beta$smooth的条件，进一步缩小了收敛阶，使lissa-sample具有二阶收敛性质。

   2. 从计算时间角度：
      - lissa-sample在二阶的基础上，进一步应用matrix sample技术，简化计算时间的复杂度

5. 达到二阶收敛速率了嘛？

   - lissa没有，线性的
   - 为什么连超线性都没有呢？
     - 因为他的限制，我s2限制，要我不能特别精确地估计，但是我觉着还是有好处的
     - lissa是一阶的，lissa-sample是二阶的，但是它没有明确地写出来，证明中的过程表示了

6. 对比一下Quasi-Newton和它？

   1. 在large scale的ml问题中，Quasi-Newton中实际上只有L-BFGS可以用到。
      - L-BFGS实际上是在BFGS的基础上限制了存储范围的大小。
      - Stochastics L-BFGS是可以用在ml中的
   2. 但是Stochastics L-BFGS是有一定缺陷的
      - 只有迭代点足够近的时候才能体现出优势，而实际上ml问题对精度的要求不高
      - 参数特别多，难以调参；并且不robust，甚至有时候不收敛（or 会溢出）
      - 只能解决convex的，对non-convex的，如nn就不行
   3. 注：Quasi-Newton本身是超线性收敛的

7. ***几点其他思考***

   1. 相对一阶方法而言，二阶方法能利用hessian的信息，应该更容易走出saddle

   2. 可以下载它在github上的代码，自己进一步考虑优化一下

      + [gitlink](https://github.com/brianbullins/lissa_code)

      + 比如，它没给lissa-sample的代码
      + 进一步，我还是怀疑它与acc的一阶算法之间的优越性，而且怀疑lissa-sample的二阶收敛性质，应该给出一些数值对比的
        + 如果真的这么好，为什么sample没给代码也没给数值结果啊。。。。。

   3. 但是我认为二阶算法方向的学习是很有必要的，而且要试图尽量解决non-convex的效果

   4. ***后续要学习什么？***

      + Allen-zhu写过一个也是linear time的non-convex：[Finding Approximate Local Minima Faster than Gradient Descent](https://arxiv.org/abs/1611.01146)
      + 里面应用的矩阵采样的方法：[Uniform Sampling for Matrix Approximation](https://arxiv.org/abs/1408.5099)
      + 我觉着还应该再看一些**加速的一阶方法**
      + 我还是想研究一下他的代码……[gitlink](https://github.com/brianbullins/lissa_code)

## 5. Partial Slides Outline

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
        - 范数小于1，正定
        - 论文中这个是通过scale f(x)，整体缩放hessian达到这个要求
   2. aj是第j项tylor，计算一下可以得到这个。
   3. 我们根据这个定义了一个estimator，从单位阵开始，逐次迭代
   4. 注意到，aj在j趋近于无穷的时候是逼近矩阵的逆的；因为我迭代的深度不同，会存在误差
5. **剩下的是自己讲的……**