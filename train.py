import math
import os
import numpy as np
from matplotlib import pyplot as plt

from baseline1 import BaselineCalculator1
from baseline2 import BaselineCalculator2
from baseline3 import BaselineCalculator3
from test_along_with_train import TestManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import multiprocessing as mp
from param import *
from spark_env.env import Environment
from average_reward import *
from compute_gradients import *
from actor_agent11 import ActorAgent1
from actor_agent12 import ActorAgent2
from actor_agent13 import ActorAgent3
from tf_logger import TFLogger
import pandas as pd
from datetime import datetime  # 导入时间模块

def invoke_model(actor_agent11,actor_agent12,actor_agent13, obs, exp1,exp2,exp3,ep):
    # parse observation
    job_dags, frontier_nodes, free_mecs, \
    node_action_map, mec_action_map = obs
    if len(frontier_nodes) == 0 or len(free_mecs)==0:
        return None, None
    # invoking the learning model
    node_act_probs1, job_act_probs1, \
    node_inputs, job_inputs, \
    node_valid_mask, job_valid_mask,\
    gcn_mats, gcn_masks, summ_mats, \
    running_dags_mat, dag_summ_backward_map, \
    job_dags_changed = \
        actor_agent11.invoke_model(obs)

    node_act_probs2, job_act_probs2, \
    node_inputs, job_inputs, \
    node_valid_mask1, job_valid_mask1,\
    gcn_mats, gcn_masks, summ_mats, \
    running_dags_mat, dag_summ_backward_map, \
    job_dags_changed = \
        actor_agent12.invoke_model(obs)

    node_act_probs3, job_act_probs3, \
    node_inputs, job_inputs, \
    node_valid_mask, job_valid_mask, \
    gcn_mats, gcn_masks, summ_mats, \
    running_dags_mat, dag_summ_backward_map, \
    job_dags_changed = \
        actor_agent13.invoke_model(obs)
    with open("./spark_env/weights.txt", "r") as fw:
        lines = fw.readlines()
    w = lines[ep - 1].strip().split()
    w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
    eps = 1e-13
    node_logits1 = np.log(node_act_probs1 + eps)
    node_logits2 = np.log(node_act_probs2 + eps)
    node_logits3 = np.log(node_act_probs3 + eps)

    job_logits1 = np.log(job_act_probs1 + eps)
    job_logits2 = np.log(job_act_probs2 + eps)
    job_logits3 = np.log(job_act_probs3 + eps)

    combined_node_logits = w1 * node_logits1 + w2 * node_logits2 + w3 * node_logits3
    combined_job_logits = w1 * job_logits1 + w2 * job_logits2 + w3 * job_logits3

    # apply mask and Gumbel sampling on combined logits
    node_noise = np.random.uniform(size=combined_node_logits.shape)
    node_act_logits = combined_node_logits - np.log(-np.log(node_noise + eps)) + (node_valid_mask - 1) * 10000.0
    node_act = np.argmax(node_act_logits, axis=1)

    job_valid_mask_re = job_valid_mask.reshape(1, np.shape(job_act_probs1)[1], np.shape(job_act_probs1)[2])
    job_noise = np.random.uniform(size=combined_job_logits.shape)
    job_act_logits = combined_job_logits - np.log(-np.log(job_noise + eps)) + (job_valid_mask_re - 1) * 10000.0
    job_act = np.argmax(job_act_logits, axis=2)

    if not node_valid_mask[0, node_act[0]] == 1:
        print('报错node')
        print(node_act[0])
        print('node_valid_mask', node_valid_mask)
    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = node_action_map[node_act[0]]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)
    # parse job_dag action
    mec = mec_action_map[job_act[0,job_idx]]
    if not job_valid_mask[0, job_act[0, job_idx] + \
                          (args.num_mecs + 1) * job_idx] == 1:
        print('报错mec')
        print(job_act[0,job_idx])
        print('job_valid_mask_re', job_valid_mask_re)
    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
                          (args.num_mecs + 1) * job_idx] == 1

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs1.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs1.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp1['node_inputs'].append(node_inputs)
    exp1['job_inputs'].append(job_inputs)
    exp1['summ_mats'].append(summ_mats)
    exp1['running_dag_mat'].append(running_dags_mat)
    exp1['node_act_vec'].append(node_act_vec)
    exp1['job_act_vec'].append(job_act_vec)
    exp1['node_valid_mask'].append(node_valid_mask)
    exp1['job_valid_mask'].append(job_valid_mask)
    exp1['job_state_change'].append(job_dags_changed)

    exp2['node_inputs'].append(node_inputs)
    exp2['job_inputs'].append(job_inputs)
    exp2['summ_mats'].append(summ_mats)
    exp2['running_dag_mat'].append(running_dags_mat)
    exp2['node_act_vec'].append(node_act_vec)
    exp2['job_act_vec'].append(job_act_vec)
    exp2['node_valid_mask'].append(node_valid_mask)
    exp2['job_valid_mask'].append(job_valid_mask)
    exp2['job_state_change'].append(job_dags_changed)

    exp3['node_inputs'].append(node_inputs)
    exp3['job_inputs'].append(job_inputs)
    exp3['summ_mats'].append(summ_mats)
    exp3['running_dag_mat'].append(running_dags_mat)
    exp3['node_act_vec'].append(node_act_vec)
    exp3['job_act_vec'].append(job_act_vec)
    exp3['node_valid_mask'].append(node_valid_mask)
    exp3['job_valid_mask'].append(job_valid_mask)
    exp3['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp1['gcn_mats'].append(gcn_mats)
        exp1['gcn_masks'].append(gcn_masks)
        exp1['dag_summ_back_mat'].append(dag_summ_backward_map)

        exp2['gcn_mats'].append(gcn_mats)
        exp2['gcn_masks'].append(gcn_masks)
        exp2['dag_summ_back_mat'].append(dag_summ_backward_map)

        exp3['gcn_mats'].append(gcn_mats)
        exp3['gcn_masks'].append(gcn_masks)
        exp3['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, mec


def train_agent(agent_id, param_queue1,param_queue2,param_queue3,
                reward_queue1,reward_queue2, reward_queue3, adv_queue1,adv_queue2,adv_queue3,
                gradient_queue1,gradient_queue2,gradient_queue3,epoch_queue,time_queue):
    # model evaluation seed
    tf.set_random_seed(agent_id)

    # set up environment
    env = Environment()

    # # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    actor_agent11 = ActorAgent1(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
    actor_agent12 = ActorAgent2(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
    actor_agent13 = ActorAgent3(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))

    while True:

        (actor_params1, seed, entropy_weight) = \
            param_queue1.get()
        (actor_params2, seed, entropy_weight) = \
            param_queue2.get()
        (actor_params3, seed, entropy_weight) = \
            param_queue3.get()

        actor_agent11.set_params(actor_params1)
        actor_agent12.set_params(actor_params2)
        actor_agent13.set_params(actor_params3)

        # reset environment
        env.seed(seed)
        env.reset(max_time=197)
        exp1 = {'node_inputs': [], 'job_inputs': [],
                    'gcn_mats': [], 'gcn_masks': [],
                    'summ_mats': [], 'running_dag_mat': [],
                    'dag_summ_back_mat': [],
                    'node_act_vec': [], 'job_act_vec': [],
                    'node_valid_mask': [], 'job_valid_mask': [],
                    'reward': [], 'wall_time': [],
                    'job_state_change': []}

        exp2 = {'node_inputs': [], 'job_inputs': [],
                'gcn_mats': [], 'gcn_masks': [],
                'summ_mats': [], 'running_dag_mat': [],
                'dag_summ_back_mat': [],
                'node_act_vec': [], 'job_act_vec': [],
                'node_valid_mask': [], 'job_valid_mask': [],
                'reward': [], 'wall_time': [],
                'job_state_change': []}

        exp3 = {'node_inputs': [], 'job_inputs': [],
                'gcn_mats': [], 'gcn_masks': [],
                'summ_mats': [], 'running_dag_mat': [],
                'dag_summ_back_mat': [],
                'node_act_vec': [], 'job_act_vec': [],
                'node_valid_mask': [], 'job_valid_mask': [],
                'reward': [], 'wall_time': [],
                'job_state_change': []}


        try:
            # run experiment
            t_obs = 0.0
            t_input_state = 0.0
            t_obs = time.time()
            obs = env.observe()
            done = False
            # model_agent = ModelAgent(sess, actor_agent11, actor_agent13)

            # initial time ### 这个应该是一个值

            exp1['wall_time'].append(env.wall_time.curr_time)
            exp2['wall_time'].append(env.wall_time.curr_time)
            exp3['wall_time'].append(env.wall_time.curr_time)

            ep = epoch_queue.get()

            ## 这里应该指的是1个环境的（只要一个环境输出step)
            while not done:
                if env.wall_time == 0:
                    t_input_state = time.time()
                node, mec = invoke_model(actor_agent11,actor_agent12,actor_agent13,obs,exp1,exp2,exp3,ep)
                obs, reward1, reward2, reward3, done = env.step2(node, mec)

                #节点执行时间超过上限，done
                if node is not None:
                    # valid action, store reward and time
                    exp1['reward'].append(reward1)
                    exp2['reward'].append(reward2)
                    exp3['reward'].append(reward3)
                    exp1['wall_time'].append(env.wall_time.curr_time)
                    exp2['wall_time'].append(env.wall_time.curr_time)
                    exp3['wall_time'].append(env.wall_time.curr_time)

                elif len(exp1['reward']) > 0:
                    exp1['reward'][-1] += reward1
                    exp1['wall_time'][-1] = env.wall_time.curr_time
                    print('elif1')
                    print(node, mec)
                elif len(exp2['reward']) > 0:
                    exp2['reward'][-1] += reward2
                    exp2['wall_time'][-1] = env.wall_time.curr_time
                    print('elif2')
                    print(node, mec)
                elif len(exp3['reward']) > 0:
                    exp3['reward'][-1] += reward3
                    exp3['wall_time'][-1] = env.wall_time.curr_time
                    print('elif3')
                    print(node, mec)


            # report reward signals to master
            assert len(exp1['node_inputs']) == len(exp1['reward'])
            assert len(exp2['node_inputs']) == len(exp2['reward'])
            assert len(exp3['node_inputs']) == len(exp3['reward'])
            ## 这里到后面看看是存在这一个reward queue里面还是存在三个reward queue里面
            reward_queue1.put(
                [exp1['reward'], exp1['wall_time'],
                 len(env.finished_job_dags),
                 np.sum([j.completion_time - j.start_time \
                          for j in env.finished_job_dags])])
            reward_queue2.put(
                [exp2['reward'], exp2['wall_time'],
                len(env.failed_job_dags)])

            reward_queue3.put(
                [exp3['reward'], exp3['wall_time'],
                 len(env.finished_job_dags),
                 np.sum([max(0,j.completion_time-(j.ddl+j.start_time)) for
                         j in env.finished_job_dags if j.var =="soft"])])

            time_queue.put([t_obs, t_input_state])

            # get advantage term from master
            batch_adv1 = adv_queue1.get()
            batch_adv2 = adv_queue2.get()
            batch_adv3 = adv_queue3.get()

            if batch_adv1  is None:
                continue
            if batch_adv2  is None:
                continue
            if batch_adv3 is None:
                continue

            # compute gradients
            actor_gradient1, loss1, node_entropy1, job_entropy1 = compute_actor_gradients(
                actor_agent11, exp1, batch_adv1, entropy_weight)

            actor_gradient2, loss2, node_entropy2, job_entropy2 = compute_actor_gradients(
                actor_agent12, exp2, batch_adv2, entropy_weight)
            #
            actor_gradient3, loss3, node_entropy3, job_entropy3 = compute_actor_gradients(
                actor_agent13, exp3, batch_adv3, entropy_weight)

            # report gradient to master
            gradient_queue1.put([actor_gradient1, loss1, node_entropy1, job_entropy1])
            gradient_queue2.put([actor_gradient2, loss2, node_entropy2, job_entropy2])
            gradient_queue3.put([actor_gradient3, loss3, node_entropy3, job_entropy3])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            print('error')
            print('len(exp1)',len(exp1['node_inputs']),'len(reward1)',len(exp1['reward']))
            print('len(exp2)',len(exp2['node_inputs']),'len(reward2)',len(exp2['reward']))
            print('len(exp3)',len(exp3['node_inputs']),'len(reward3)',len(exp3['reward']))
            reward_queue1.put(None)
            reward_queue2.put(None)
            reward_queue3.put(None)

            # need to still get from adv_queue to
            # prevent blocking
            adv_queue1.get()
            adv_queue2.get()
            adv_queue3.get()

def write_tensorboard_log(epoch, var0, var1, var2, var3, var4, var5, var6, var7, var8, var9, sess, merged_summary, writer,
                         ph0, ph1, ph2, ph3, ph4, ph5, ph6, ph7, ph8, ph9):  # 新增占位符参数
    summary = sess.run(merged_summary, feed_dict={
        ph0: var0,
        ph1: var1,
        ph2: var2,
        ph3: var3,
        ph4: var4,
        ph5: var5,
        ph6: var6,
        ph7: var7,
        ph8: var8,
        ph9: var9
    })
    writer.add_summary(summary, global_step=epoch)
    writer.flush()

def main():
    # tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder1)
    create_folder_if_not_exists(args.model_folder2)
    create_folder_if_not_exists(args.model_folder3)

    # initialize communication queues
    params_queues1 = [mp.Queue(1) for _ in range(args.num_agents)]
    params_queues2 = [mp.Queue(1) for _ in range(args.num_agents)]
    params_queues3 = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues1 = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues2 = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues3 = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues1 = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues2 = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues3 = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues1 = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues2 = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues3 = [mp.Queue(1) for _ in range(args.num_agents)]
    epoch_queue = mp.Queue(1)
    time_queue = mp.Queue(1)
    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues1[i],params_queues2[i], params_queues3[i],
            reward_queues1[i], reward_queues2[i],reward_queues3[i],
            adv_queues1[i], adv_queues2[i],adv_queues3[i],
            gradient_queues1[i],gradient_queues2[i], gradient_queues3[i], epoch_queue, time_queue)))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)
    # set up actor agent
    actor_agent11 = ActorAgent1(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
    actor_agent12 = ActorAgent2(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
    actor_agent13 = ActorAgent3(sess,args.node_input_dim, args.job_input_dim, args.hid_dims,
                                args.output_dim, args.max_depth, range(0, args.num_mecs + 1))

    bs1 = BaselineCalculator1()
    bs2 = BaselineCalculator2()
    bs3 = BaselineCalculator3()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess.run(tf.global_variables_initializer())
    #画图--------------------------------------------------------
    # 主函数内创建占位符（局部变量）
    log0_ph = tf.placeholder(tf.float32, shape=(), name="log0_ph")
    log1_ph = tf.placeholder(tf.float32, shape=(), name="log1_ph")
    log2_ph = tf.placeholder(tf.float32, shape=(), name="log2_ph")
    log6_ph = tf.placeholder(tf.float32, shape=(), name="log6_ph")
    adv_loss1_ph = tf.placeholder(tf.float32, shape=(), name="adv_loss1_ph")
    adv_loss2_ph = tf.placeholder(tf.float32, shape=(), name="adv_loss2_ph")
    adv_loss3_ph = tf.placeholder(tf.float32, shape=(), name="adv_loss3_ph")
    node_entropy1_ph = tf.placeholder(tf.float32, shape=(), name="node_entropy1_ph")
    node_entropy2_ph = tf.placeholder(tf.float32, shape=(), name="node_entropy2_ph")
    node_entropy3_ph = tf.placeholder(tf.float32, shape=(), name="node_entropy3_ph")

    # 主函数内定义摘要和合并操作
    tf.summary.scalar("log0", log0_ph)
    tf.summary.scalar("log1", log1_ph)
    tf.summary.scalar("log2", log2_ph)
    tf.summary.scalar("log6", log6_ph)
    tf.summary.scalar('adv_loss1', adv_loss1_ph)
    tf.summary.scalar('adv_loss2', adv_loss2_ph)
    tf.summary.scalar('adv_loss3', adv_loss3_ph)
    tf.summary.scalar('node_entropy1', node_entropy1_ph)
    tf.summary.scalar('node_entropy2', node_entropy1_ph)
    tf.summary.scalar('node_entropy3', node_entropy3_ph)
    merged_summary = tf.summary.merge_all()

    # 主函数内创建日志写入器
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(f"./logs/run_{timestamp}")
    #-----------------------------------------------------------

    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # 定义带时间戳的Excel路径（每次训练文件名不同）
    excel_path = f"./result_data/training_metrics1_{timestamp}.xlsx"

    # 确保文件夹存在（当前目录，通常已存在，保险措施）
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    # 定义固定列名（表头）
    columns = ["Epoch", '加权奖励', "总服务时间", "实时逾期", "非实时惩罚", '开始传参到获得奖励',
               '计算基线优势', '计算损失梯度', '应用梯度', '开始观察状态', '开始输入状态']
    df = pd.DataFrame()

    #=============================================
    test = TestManager(f"./test_result/test_{timestamp}.xlsx")
    #=============================================

    # ---- start training process ----
    for ep in range(1, args.num_ep+1):
        print('training epoch', ep)
        epoch_queue.put(ep)
        # synchronize the model parameters for each training agent

        actor_params1 = actor_agent11.get_params()
        actor_params2 = actor_agent12.get_params()
        actor_params3 = actor_agent13.get_params()

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues1[i].put([
                actor_params1, args.seed + ep,
                entropy_weight])

            params_queues2[i].put([
                actor_params2, args.seed + ep,
                entropy_weight])

            params_queues3[i].put([
                actor_params3, args.seed + ep,
                entropy_weight])

        # storage for advantage computation ##怎么算的要会
        all_rewards1, all_diff_times1, all_times1, \
        all_num_finished_jobs, all_avg_job_duration, \
            = [], [], [], [], []

        all_rewards2, all_diff_times2, all_times2, \
        all_failed_rate, \
            = [], [], [], []
        #
        all_rewards3, all_diff_times3, all_times3, \
        all_num_finished_jobs,all_overdue_penalty, \
            = [], [], [], [],[]

        t1 = time.time()

        # get reward from agents
        any_agent_panic = False

        for i in range(args.num_agents):
            result1 = reward_queues1[i].get()
            result2 = reward_queues2[i].get()
            result3 = reward_queues3[i].get()

            t2 = time.time()
            print('开始传参到获得奖励', t2 - t1, 'seconds')

            if result1 is None:
                any_agent_panic = True
                print('result1 is None')
                continue
            if result2 is None:
                any_agent_panic = True
                print('result2 is None')
                continue
            if result3 is None:
                any_agent_panic = True
                print('result3 is None')
                continue
            else:
                batch_reward1, batch_time1, \
                num_finished_jobs, avg_job_duration \
                    = result1
                batch_reward2, batch_time2, \
                failed_rate \
                    = result2
                batch_reward3, batch_time3, \
                num_finished_jobs,overdue_penalty \
                    = result3

            diff_time1 = np.array(batch_time1[1:]) - \
                         np.array(batch_time1[:-1])
            diff_time2 = np.array(batch_time2[1:]) - \
                         np.array(batch_time2[:-1])
            diff_time3 = np.array(batch_time3[1:]) - \
                         np.array(batch_time3[:-1])

            all_rewards1.append(batch_reward1)
            all_rewards2.append(batch_reward2)
            all_rewards3.append(batch_reward3)

            all_diff_times1.append(diff_time1)
            all_diff_times2.append(diff_time2)
            all_diff_times3.append(diff_time3)

            all_times1.append(batch_time1[1:])
            all_times2.append(batch_time2[1:])
            all_times3.append(batch_time3[1:])

            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_failed_rate.append(failed_rate)
            all_overdue_penalty.append(overdue_penalty)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward1, diff_time1)
            avg_reward_calculator.add_list_filter_zero(
                batch_reward2, diff_time2)
            avg_reward_calculator.add_list_filter_zero(
                batch_reward3, diff_time3)

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues1[i].put(None)
                adv_queues2[i].put(None)
                adv_queues3[i].put(None)
            continue

        # compute differential reward
        all_cum_reward1,all_cum_reward2,all_cum_reward3 = [],[],[]
        avg_per_step_reward1 = avg_reward_calculator.get_avg_per_step_reward()
        avg_per_step_reward2 = avg_reward_calculator.get_avg_per_step_reward()
        avg_per_step_reward3 = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards1 = np.array([r - avg_per_step_reward1 * t for \
                                     (r, t) in zip(all_rewards1[i], all_diff_times1[i])])
                rewards2 = np.array([r - avg_per_step_reward2 * t for \
                                     (r, t) in zip(all_rewards2[i], all_diff_times2[i])])
                rewards3 = np.array([r - avg_per_step_reward3 * t for \
                                     (r, t) in zip(all_rewards3[i], all_diff_times3[i])])
            else:
                # regular reward
                rewards1 = np.array([r for \
                                     (r, t) in zip(all_rewards1[i], all_diff_times1[i])])
                rewards2 = np.array([r for \
                                     (r, t) in zip(all_rewards2[i], all_diff_times2[i])])
                rewards3 = np.array([r for \
                                     (r, t) in zip(all_rewards3[i], all_diff_times3[i])])

            cum_reward1 = discount(rewards1, args.gamma)
            cum_reward2 = discount(rewards2, args.gamma)
            cum_reward3 = discount(rewards3, args.gamma)

            all_cum_reward1.append(cum_reward1)
            all_cum_reward2.append(cum_reward2)
            all_cum_reward3.append(cum_reward3)

        flat_cum_reward1 = np.array([r for ep in all_cum_reward1 for r in ep], dtype=np.float32)
        flat_times1 = np.array([t for ep in all_times1 for t in ep], dtype=np.float32)

        flat_cum_reward2 = np.array([r for ep in all_cum_reward2 for r in ep], dtype=np.float32)
        flat_times2 = np.array([t for ep in all_times2 for t in ep], dtype=np.float32)

        flat_cum_reward3 = np.array([r for ep in all_cum_reward3 for r in ep], dtype=np.float32)
        flat_times3 = np.array([t for ep in all_times3 for t in ep], dtype=np.float32)

        # compute baseline
        baselines1 = bs1.get_baseline1(ep, flat_cum_reward1, flat_times1)
        baselines2 = bs2.get_baseline2(ep, flat_cum_reward2, flat_times2)
        baselines3 = bs3.get_baseline3(ep, flat_cum_reward3, flat_times3)

        # give worker back the advantage
        for i in range(args.num_agents):
            # ====== 直接对 raw advantage 做全局缩放 ======
            batch_adv1 = flat_cum_reward1 - baselines1
            batch_adv1 = batch_adv1.reshape(-1, 1)
            adv_queues1[i].put(batch_adv1)

            batch_adv2 = flat_cum_reward2 - baselines2
            batch_adv2 = batch_adv2.reshape(-1, 1)
            adv_queues2[i].put(batch_adv2)

            batch_adv3 = flat_cum_reward3 - baselines3
            batch_adv3 = batch_adv3.reshape(-1, 1)
            adv_queues3[i].put(batch_adv3)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients1 = []
        adv_loss1 = []
        n_entropy1 = []
        job_entropy1 = []

        actor_gradients2 = []
        adv_loss2 = []
        n_entropy2 = []
        job_entropy2 = []

        actor_gradients3 = []
        adv_loss3 = []
        n_entropy3 = []
        job_entropy3 = []

        for i in range(args.num_agents):
            (actor_gradient1, loss1, node_entropy1, job_entropy1) = gradient_queues1[i].get()
            (actor_gradient2, loss2, node_entropy2, job_entropy2) = gradient_queues2[i].get()
            (actor_gradient3, loss3, node_entropy3, job_entropy3) = gradient_queues3[i].get()

            actor_gradients1.append(actor_gradient1)
            actor_gradients2.append(actor_gradient2)
            actor_gradients3.append(actor_gradient3)

            adv_loss1 = loss1[0]
            adv_loss2 = loss2[0]
            adv_loss3 = loss3[0]
            n_entropy1 = node_entropy1
            n_entropy2 = node_entropy2
            n_entropy3 = node_entropy3


        bs1.update1(flat_cum_reward1, flat_times1)
        bs2.update2(flat_cum_reward2, flat_times2)
        bs3.update3(flat_cum_reward3, flat_times3)

        [t_observe, t_input_s] = time_queue.get()

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent11.apply_gradients(
            aggregate_gradients(actor_gradients1), args.lr)
        actor_agent12.apply_gradients(
            aggregate_gradients(actor_gradients2), args.lr)
        actor_agent13.apply_gradients(
            aggregate_gradients(actor_gradients3), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')
        #======================================================================================================================================
        # ew_scheduler.update(ep,job_entropy1,np.mean(all_avg_job_duration), adv_loss1)
             # entropy_weight = ew_scheduler.adjust(ep)
        # if ep % 50 == 0:
        #     ew.append(entropy_weight)
        #     if ep % 100 == 0:
        # decrease entropy weight
        #指数衰减
        # entropy_weight = 1.2 * (0.0001 ** (ep / 12000))
        # #余弦退火衰减
        # if ep <= 8000:
        #     entropy_weight = args.entropy_weight_min + (args.entropy_weight_init - args.entropy_weight_min) * (1 + math.cos(math.pi * ep / 8000)) / 2
        # else:
        #     entropy_weight = args.entropy_weight_min
        # ==== entropy weight piecewise decay ====
        raw = args.entropy_weight_init + (args.entropy_weight_init - args.entropy_weight_min) * ((np.exp(3 * ep / 4800)- 1) / (1 - np.exp(3)))
        entropy_weight = max(raw, args.entropy_weight_min)

        #
        # #线性衰减
        # entropy_weight = decrease_var(entropy_weight,
        #                               args.entropy_weight_min, 1/8000)
        #======================================================================================================================================

        log0 = np.mean(all_avg_job_duration)
        log1 = np.mean(all_failed_rate)
        log2 = np.mean(all_overdue_penalty)
        log3 = np.mean([cr[0] for cr in all_cum_reward1])
        log4 = np.mean([cr[0] for cr in all_cum_reward2])
        log5 = np.mean([cr[0] for cr in all_cum_reward3])
        log6 = log0 + log1 + log2
        print('all_avg_job_duration',all_avg_job_duration,'all_failed_rate',all_failed_rate,'log0',log0,'log1',log1,'log2',log2)

        #画图-----------------------------------------------------------------------
        write_tensorboard_log(
            ep,log0, log1, log2, log6, adv_loss1, adv_loss2, adv_loss3, n_entropy1, n_entropy2, n_entropy3,
            sess,merged_summary,writer,log0_ph, log1_ph,log2_ph, log6_ph, adv_loss1_ph, adv_loss2_ph, adv_loss3_ph,
            node_entropy1_ph, node_entropy2_ph, node_entropy3_ph)
        if ep % args.model_save_interval == 0 and ep >= 0 :
            actor_agent11.save_model1(args.model_folder1 + \
                                     "model_" + str(timestamp) + '_' + str(ep))
            actor_agent12.save_model2(args.model_folder2 + \
                                     "model_" + str(timestamp) + '_' + str(ep))
            actor_agent13.save_model3(args.model_folder3 + \
                                     "model_" + str(timestamp) + '_' + str(ep))
            if ep % 30000 == 0:
                 test.run_test(actor_agent11, actor_agent12, actor_agent13,ep)

        # ---------------------------------------------------------------------------------------
        # 训练循环中，按epoch保存数据（示例）
        # 假设当前epoch的指标（替换为实际数据）
        metrics_list = [ep, log6, log0, log1, log2, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t_observe, t_input_s]
        # 第一次写入时创建DataFrame（带表头），后续直接追加
        if ep == 0:  # 假设从ep=0开始训练，第一次循环创建文件
            df = pd.DataFrame([metrics_list], columns=columns)
        else:
            # 追加当前数据
            df = pd.concat([df, pd.DataFrame([metrics_list], columns=columns)], ignore_index=True)


    # 训练结束后，保存整个训练过程的指标到带时间戳的文件
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"本次训练数据已保存至：{excel_path}")
    # ---------------------------------------------------------------------------------------

    sess.close()
    writer.close()
    # sess2.close()


if __name__ == '__main__':
    main()
