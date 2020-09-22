import os
import pickle
import argparse
import numpy as np
from collections import deque


import time
from environment import Env
from PIL import ImageGrab

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, VDB
from train_model import train_actor_critic, train_vdb

expert_sample_size = 50


iteration = 12000
# tot_sample_size = 1000

parser = argparse.ArgumentParser(description='PyTorch VAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=50,
                    help='hidden unit size of actor, critic and vdb networks (default: 100)')
parser.add_argument('--z_size', type=int, default=4,
                    help='latent vector z unit size of vdb networks (default: 4)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--alpha_beta', type=float, default=1e-4,
                    help='step size to be used in beta term (default: 1e-4)')
parser.add_argument('--i_c', type=float, default=0.5, 
                    help='constraint for KL-Divergence upper bound (default: 0.5)')
parser.add_argument('--vdb_update_num', type=int, default=3, 
                    help='update number of variational discriminator bottleneck (default: 3)')
parser.add_argument('--ppo_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')

parser.add_argument('--total_sample_size', type=int, default=10,
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=iteration,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

tot_sample_size = 1


def main():
    expert_demo = pickle.load(open('./Expert dataset 1/expert_20x20_1.p', "rb"))
    demonstrations = np.array(expert_demo[0])

    print("demonstrations.shape", demonstrations.shape)

    print(expert_demo[1])
    print(expert_demo[0])
    print(np.array(expert_demo[0]).shape)

    # expert_x = int(expert_demo[1][0])
    # expert_y = int(expert_demo[1][1])

    expert_x = int(expert_demo[0][0])
    expert_y = int(expert_demo[0][1])


    env = Env(expert_x, expert_y)

    # env.seed(args.seed)
    # torch.manual_seed(args.seed)

    num_inputs = 6
    num_actions = 8
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    vdb = VDB(num_inputs + num_actions, args)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    vdb_optim = optim.Adam(vdb.parameters(), lr=args.learning_rate)
    
    # load demonstrations

    k = 1
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        vdb.load_state_dict(ckpt['vdb'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0
    train_discrim_flag = True



    for iter in range(args.max_iter_num):
        # expert_demo = pickle.load(open('./paper/{}.p'.format((iter+1)%expert_sample_size), "rb"))
        print(iter)
        expert_demo = pickle.load(open('./Expert dataset 1/expert_20x20_{}.p'.format(np.random.randint(1,50)), "rb"))
        tmp = expert_demo.pop(-1)

        demonstrations = np.array(expert_demo)

        print(demonstrations, demonstrations.shape)
        tot_sample_size = len(demonstrations) + 10
        ##########################

        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []

        # while steps < args.total_sample_size:

        while steps < tot_sample_size:
            # env.delete_graph()
            state = env.reset()
            # time.sleep(1)

            score = 0

            # state = running_state(state)
            state1 = state
            for _ in range((tot_sample_size+1)*2):
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action2 = np.argmax(get_action(mu, std)[0])
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action2)

                irl_reward = get_reward(vdb, state, action)

                # ###### 동영상 촬영용
                # if iter > 11500 :
                #     time.sleep(0.015)
                # #####
                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, irl_reward, mask])

                # next_state = running_state(next_state)
                state = next_state

                score += reward

                if done:
                    break
            ##########################
            env.draw_graph()
            env.render()
            ##########################
            episodes += 1
            scores.append(score)
        
        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train(), vdb.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_vdb(vdb, memory, vdb_optim, demonstrations, 0, args)
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'vdb': vdb.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

    ####
    score_avg = int(score_avg)

    model_path = os.path.join(os.getcwd(), 'save_model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    ckpt_path = os.path.join(model_path, 'ckpt_' + 'last_model' + '.pth.tar')

    save_checkpoint({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'vdb': vdb.state_dict(),
        'z_filter_n': running_state.rs.n,
        'z_filter_m': running_state.rs.mean,
        'z_filter_s': running_state.rs.sum_square,
        'args': args,
        'score': score_avg
    }, filename=ckpt_path)

if __name__=="__main__":
    main()

