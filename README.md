# 应用类型：soft-ddl+hard-ddl
# 多目标RL
1、spark_env:环境部分包含除各种神经网络和训练的代码：重点为生成不同类型的DAG，以及在env文件中任务调度和卸载的流程(step).
2、actor_agent---标量网络 
  actor_agent11-actor_agent13---分别与三个网络对应，向量网络
  actor_agent0---不加gnn的标量网络
3、train文件为训练，主函数在，得到的模型分别保存在model1,2,3中，数据在results中.
   train_3---多目标rl
   train_nognn---无图神经网络rl
   train_orig---标量rl
4、other_agents:保存对比算法，分别是ddos，greedy,heft和random,在参数文件中的scheme里可以直接改所需要测试的对比算法，
   如果选learn，则是测试训练后保存模型
