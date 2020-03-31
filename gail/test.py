# If you want to test GAIL,
# Test the agent with the saved model ckpt_4000_gail.pth.tar
# in the mujoco/gail/save_model folder.
# $  python test.py --load_model ckpt_4000_gail.pth.tar

# If you want to Continue training from the saved checkpoint,
# $ python main.py --load_model ckpt_4000_gail.pth.tar

import os
from environment import Env
import torch
import argparse

from model import Actor, Critic
from utils.utils import get_action
# import utils.zfilter
from utils.zfilter import ZFilter
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="Hopper-v2",
                    help='name of Mujoco environement')
parser.add_argument('--iter', type=int, default=5,
                    help='number of episodes to play')
parser.add_argument("--load_model", type=str, default='ppo_max.tar',
                     help="if you test pretrained file, write filename in save_model folder")

parser.add_argument('--hidden_size', type=int, default=100,
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')


args = parser.parse_args()


if __name__ == "__main__":

    env = Env(20, 20)
    # env.seed(500)
    torch.manual_seed(500)

    num_inputs = 2
    num_actions = 8

    print("state size: ", num_inputs)
    print("action size: ", num_actions)

    actor = Actor(num_inputs, num_actions,args)
    critic = Critic(num_inputs,args)

    running_state = ZFilter((num_inputs,), clip=5)
    # running_state = ZFilter((100*100,), clip=5)

    # print(running_state)

    if args.load_model is not None:
        pretrained_model_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))

        pretrained_model = torch.load(pretrained_model_path)

        actor.load_state_dict(pretrained_model['actor'])
        critic.load_state_dict(pretrained_model['critic'])

        running_state.rs.n = pretrained_model['z_filter_n']
        running_state.rs.mean = pretrained_model['z_filter_m']
        running_state.rs.sum_square = pretrained_model['z_filter_s']

        print("Loaded OK ex. ZFilter N {}".format(running_state.rs.n))

    else:
        assert("Should write pretrained filename in save_model folder. ex) python3 test_algo.py --load_model ppo_max.tar")


    actor.eval(), critic.eval()
    for episode in range(args.iter):
        state = env.reset()
        steps = 0
        score = 0
        for _ in range(500):
            env.render()

            # mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            mu, std = actor(torch.Tensor(state).unsqueeze(0))
            action2 = np.argmax(get_action(mu, std)[0])
            action = get_action(mu, std)[0]
            print('mu, std :', mu, std)
            next_state, reward, done, _ = env.step(action2)
            print('1','next_state :', next_state)
            next_state = running_state(next_state) # ZFilter의 역할 : env 환경에 맞춰진 state 값을 반환
            print('2','next_state :', next_state)

            state = next_state
            score += reward

            if done:
                print("{} cumulative reward: {}".format(episode, score))
                break
    print(torch.Tensor(state).unsqueeze(0))
