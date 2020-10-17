import math

import mrl
from mrl.utils.misc import soft_update, flatten_state
from mrl.modules.model import PytorchModel

import numpy as np
import torch
import torch.nn.functional as F
import os


class QValuePolicy(mrl.Module):
    """ For acting in the environment"""

    def __init__(self):
        super().__init__(
            'policy',
            required_agent_modules=[
                'qvalue', 'env', 'replay_buffer'
            ],
            locals=locals())

    def _setup(self):
        self.use_qvalue_target = self.config.get('use_qvalue_target') or False

    def __call__(self, state, greedy=False):
        res = None

        # Initial Exploration
        if self.training:
            if self.config.get('initial_explore') and len(
                    self.replay_buffer) < self.config.initial_explore:
                res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            elif hasattr(self, 'ag_curiosity'):
                state = self.ag_curiosity.relabel_state(state)

        state = flatten_state(state)  # flatten goal environments
        if hasattr(self, 'state_normalizer'):
            state = self.state_normalizer(state, update=self.training)

        if res is not None:
            return res

        state = self.torch(state)

        if self.use_qvalue_target:
            q_values = self.numpy(self.qvalue_target(state))
        else:
            q_values = self.numpy(self.qvalue(state))

        if self.training and not greedy and np.random.random() < self.config.random_action_prob(
                steps=self.config.env_steps):
            # print("The action space is: " + str(self.env.action_space) + " with " + str(self.env.action_space.n) + " choices")
            action = np.random.randint(self.env.action_space.n, size=[self.env.num_envs])
        else:
            action = np.argmax(q_values, -1)  # Convert to int

        return action


class BaseQLearning(mrl.Module):
    """ Generic Discrete Action Q-Learning Algorithm"""

    def __init__(self):
        super().__init__(
            'algorithm',
            required_agent_modules=['qvalue', 'replay_buffer', 'env'],
            locals=locals())

    def _setup(self):
        """ Set up Q-value optimizers and create target network modules."""

        self.targets_and_models = []

        # Q-Value setup
        qvalue_params = []
        self.qvalues = []
        for module in list(self.module_dict.values()):
            name = module.module_name
            if name.startswith('qvalue') and isinstance(module, PytorchModel):
                self.qvalues.append(module)
                qvalue_params += list(module.model.parameters())
                # Create the target network with the same structure as the normal Q network
                target = module.copy(name + '_target')
                target.model.load_state_dict(module.model.state_dict())
                self.agent.set_module(name + '_target', target)
                self.targets_and_models.append((target.model, module.model))

        self.q_value_optimizer = torch.optim.Adam(
            qvalue_params,
            lr=self.config.qvalue_lr,
            weight_decay=self.config.qvalue_weight_decay)

        self.qvalue_params = qvalue_params

    def save(self, save_folder: str):
        path = os.path.join(save_folder, self.module_name + '.pt')
        torch.save({
            'qvalue_opt_state_dict': self.q_value_optimizer.state_dict(),
        }, path)

    def load(self, save_folder: str):
        path = os.path.join(save_folder, self.module_name + '.pt')
        checkpoint = torch.load(path)
        self.q_value_optimizer.load_state_dict(checkpoint['qvalue_opt_state_dict'])

    def _optimize(self):
        # If the replay buffer is out of "warm-up" mode
        if len(self.replay_buffer) > self.config.warm_up:
            # Sample some states/actions/gammas from the replay buffer
            states, actions, rewards, next_states, gammas, dones = self.replay_buffer.sample(
                self.config.batch_size, include_dones=True)

            # Run the subclass optimize method
            self.optimize_from_batch(states, actions, rewards, next_states, gammas, dones)

            # If we should update the target network, do it
            if self.config.opt_steps % self.config.target_network_update_freq == 0:
                for target_model, model in self.targets_and_models:
                    # Update all the parameters of the target model
                    # according to the following equation, there b is the update factor:
                    # Q_target <- (1 - b) * Q_target + b * ( TODO )
                    soft_update(target_model, model, self.config.target_network_update_frac)

    def optimize_from_batch(self, states, actions, rewards, next_states, gammas, dones):
        raise NotImplementedError('Subclass this!')


class DQN(BaseQLearning):
    def optimize_from_batch(self, states, actions, rewards, next_states, gammas, dones=None):
        """
        The optimization method that gets called from _optimize in the BaseQLearning module.

        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param gammas:
        :return: nothing
        """
        # The Q_next value is what the target q network applied to the next states gives us
        # We detach it since it's not relevant for gradient computations
        q_next = self.qvalue_target(next_states).detach()

        # Minimum modification to get double q learning to work
        # (Hasselt, Guez, and Silver, 2016: https://arxiv.org/pdf/1509.06461.pdf)
        if self.config.double_q:
            best_actions = torch.argmax(self.qvalue(next_states), dim=-1, keepdim=True)
            q_next = q_next.gather(-1, best_actions)
        else:
            q_next = q_next.max(-1, keepdims=True)[0]  # Assuming action dim is the last dimension

        # Set the target (y_j in Mnih et al.) to r_j + gamma * Q_target(next_states)
        # TODO: why aren't we taking the maximum over the actions?
        target = (rewards + gammas * q_next)

        # Optionally clip the targets--empirically, this seems to work better
        target = torch.clamp(target, *self.config.clip_target_range).detach()

        if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
            self.logger.add_histogram('Optimize/Target_q', target)

        # Get the actual Q function for the real network on the current states
        q = self.qvalue(states)

        # Index the rows of the q-values by the batch-list of actions
        # q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))
        q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))

        # Get the squared bellman error
        td_loss = F.mse_loss(q, target)

        # Clear previous gradients before the backward pass
        self.q_value_optimizer.zero_grad()

        # Run the backward pass
        td_loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            torch.nn.utils.clip_grad_norm_(self.qvalue_params, self.config.grad_norm_clipping)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.qvalue_params, self.config.grad_value_clipping)

        # Run the update
        self.q_value_optimizer.step()

        return


class DistributionalQNetwork(BaseQLearning):
    def __init__(self, num_atoms, v_max, v_min):
        """
        Creates a Distributional Q Network (https://arxiv.org/abs/1707.06887)

        :param num_atoms: the number of atoms. 5 for SoRB, 51 for Atari
        :param v_max: maximum value for value distribution estimation
        :param v_min: minimum value for value distribution estimation
        """
        super().__init__()

        self.num_atoms = num_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.value_range = torch.tensor(v_max - v_min)
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    def __project_next_state_value_distribution(self, states, actions, rewards, next_states, gammas, dones,
                                                next_distribution, best_actions):
        batch_size = states.size(0)
        projected_distribution = torch.zeros(next_distribution.size())

        for i in range(batch_size):
            for j in range(self.num_atoms):
                # Compute the projection of ˆTz_j onto the support {z_i}
                Tz_j = (rewards[i] + (1 - dones[i]) * gammas[i] * self.z[j]).clamp(min=self.v_min, max=self.v_max)
                b_j = (Tz_j - self.v_min) / self.delta_z
                l, u = math.floor(b_j), math.ceil(b_j)

                # Distribute probability of ˆTz_j
                projected_distribution[i][actions[i].long()][int(l)] += (next_distribution[i][best_actions[i][j]][j] * (
                        u - b_j)).squeeze(0)
                projected_distribution[i][actions[i].long()][int(u)] += (next_distribution[i][best_actions[i][j]][j] * (
                        b_j - l)).squeeze(0)

        return projected_distribution

    def optimize_from_batch(self, states, actions, rewards, next_states, gammas, dones):
        """
        The optimization method that gets called from _optimize in the BaseQLearning module.

        :param states: the states, of size [batch_size, num_values]
        :param actions: the actions, of size [batch_size]
        :param rewards: the rewards, of size [batch_size, 1]
        :param next_states: the next states, of size [batch_size, num_values]
        :param gammas: the discount factors, of size [batch_size, 1]
        :param dones: the done states, of size [batch_size, 1]
        :return: nothing
        """

        # Get some useful values
        batch_size = states.size(0)
        num_actions = self.env.action_space.n
        v_min = self.v_min
        v_max = self.v_max
        delta_z = self.delta_z
        num_atoms = self.num_atoms

        # Run the q network on the current states.
        q_val = self.qvalue(states).view(batch_size, num_actions, num_atoms)  # [batch_size, num_actions, num_atoms]

        # q_val = torch.stack(
        #     [q_val[i].index_select(0, actions[i].int().to(torch.int64)) for i in range(batch_size)]).squeeze(1)

        # First, we run the Q value network on our next states.
        q_next = self.qvalue_target(next_states).view(batch_size, num_actions,
                                                      num_atoms).detach()  # [batch_size, num_actions, num_atoms]

        # This gives us a matrix of size |A| x N.
        # That is, each row of the matrix represents a single action
        # and the values in that row contain each probability for the N atoms.

        # Now, we need to compute Q(x_{t+1}, a), which is the expectation of the
        # distribution Z(x_{t+1}, a). Using the first line of Algorithm 1 in
        # Bellemare, Dabney, Munos, we know that that is \sum_i z_i p_i (x_{t+1}, a).

        # We compute all of the Q(x_{t+1}, a) values (that is, for every action a)
        # using matrix multiplication: Q(x_{t+1}) = q_next * {z_0, z_1, ..., z_{N-1}}^T.
        # q_next = torch.matmul(q_next.double(), torch.tensor(np.array(self.z).reshape(-1, 1))).squeeze(
        #     2)  # [batch_size, num_actions]

        # Now get the best action a*:
        best_actions = q_next.argmax(dim=1)

        # Now, we do the for loop to project it onto our support.
        projected_dist = self.__project_next_state_value_distribution(states, actions, rewards, next_states, gammas,
                                                                      dones,
                                                                      q_next, best_actions)

        # get loss
        loss = projected_dist - q_val
        loss = torch.mean(loss)

        # Clear previous gradients before the backward pass
        self.q_value_optimizer.zero_grad()

        # Run the backward pass
        loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            torch.nn.utils.clip_grad_norm_(self.qvalue_params, self.config.grad_norm_clipping)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.qvalue_params, self.config.grad_value_clipping)

        # Run the update
        self.q_value_optimizer.step()
