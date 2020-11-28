import math
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import torch
import torch.nn.functional as F

import mrl
from mrl.modules.model import PytorchModel
from mrl.utils.misc import soft_update, flatten_state


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

        # if self.use_qvalue_target:
        #     q_values = self.numpy(self.qvalue_target(state.view(-1))).reshape(
        #         [4, self.config.other_args['max_episode_steps']])
        # else:
        #     q_values = self.numpy(self.qvalue(state.view(-1))).reshape([4, self.config.other_args['max_episode_steps']])

        q_values = self.agent.algorithm.get_expected_q_values_from_flat(state)[0]

        if self.training and not greedy and np.random.random() < self.config.random_action_prob(
                steps=self.config.env_steps):
            # print("The action space is: " + str(self.env.action_space) + " with " + str(self.env.action_space.n) + " choices")
            action = np.random.randint(self.env.action_space.n, size=[self.env.num_envs])
        else:
            d_q_v = [get_weighted_average_of_bins(q_values[i]) for i in range(q_values.shape[0])]
            action = [np.argmax(d_q_v, -1)]  # Convert to int

        return action


class RandomPolicy(mrl.Module):
    """ For acting in the environment"""

    def __init__(self):
        super().__init__(
            'policy',
            required_agent_modules=[
                'env'
            ],
            locals=locals())

    def _setup(self):
        pass

    def __call__(self, state):
        action = np.random.randint(self.env.action_space.n, size=[self.env.num_envs])

        return action


class SearchPolicy(mrl.Module):
    def __init__(self):
        """
         Args:
             rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
             pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                    rb_vec[i] to rb_vec[j]
             max_search_steps: (int)
             open_loop: if True, only performs search once at the beginning of the episode
             weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
             no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
         """

        super().__init__(
            'policy',
            required_agent_modules=[
                'qvalue', 'env', 'replay_buffer'
            ],
            locals=locals())

        # Initialize the replay buffer graph
        # and the pairwise distance matrix
        self.replay_states_in_graph = []
        self.pairwise_distances = None

        # Parameters (bugbug: read from passed arguments)
        self.aggregate = False  # How to aggregate the ensemble values
        self.max_dist = 1  # Maximum search distance
        self.open_loop = False  # Whether it's open loop (take steps before replanning)
        self.weighted_path_planning = False  # Whether to use weighted path planning

        self.no_waypoint_hopping = False  # Is hopping waypoint ok
        self.cleanup = False  # idk
        self.attempt_cutoff = 3 * self.max_dist  # idk
        self.reached_final_waypoint = False  # idk

        self.graph_update = 0  # Whether the graph has been initialized

        self.training_steps = 0

        self.reset_stats()

    def build_graph_on_top_of_replay_buffer(self):
        """
        Builds a graph on top of the current replay buffer.

        :return: none
        """

        # Create a directed graph
        graph = nx.DiGraph()

        # Sample everything from the replay buffer
        # (we build the graph on top of the whole thing so far)
        # states, actions, rewards, next_states, gammas = self.agent.replay_buffer.sample(len(
        #     self.replay_buffer))
        states, actions, rewards, next_states, gammas = self.agent.replay_buffer.sample(17)

        # In our replay buffer, we care about the states,
        # and not the goals that we had at that point in time
        next_states, _ = torch.chunk(next_states, 2, dim=-1)

        self.replay_states_in_graph = []
        for i in range(next_states.shape[0]):
            self.replay_states_in_graph.append(next_states[i])

        # Get the pairwise distances between each pair of points in the current replay buffer.
        # (This is a bit of a huge graph, which is one of the issues with SoRB)
        pairwise_distances = self.agent.algorithm.get_pairwise_dist(next_states.numpy(), aggregate=None)

        combined_pairwise_distances = np.max(pairwise_distances, axis=-1)

        # For each pair of points in the replay buffer
        for i, s_i in enumerate(self.replay_states_in_graph):
            for j, s_j in enumerate(self.replay_states_in_graph):
                # Get the distance between them
                length = combined_pairwise_distances[i, j]

                # If the length is less than the max_dist hyperparameter,
                # add it to the graph with an edge weight being the estimated distance.
                # Note that we don't actually make a graph out of "real states",
                # like observations--we just build the graph on top of the indices.
                # Thus, the graph just contains numbers, but the number i in the graph
                # corresponds to the ith state in the replay buffer, so it's isomorphic.
                if length < self.max_dist:
                    graph.add_edge(i, j, weight=length)

        # Save the graph to a class variable.
        self.replay_buffer_graph = graph

        # nx.draw(graph, with_labels=True)
        # plt.show()

        if not self.open_loop:
            print("Building F_W distances now yo")

            pdist2 = self.agent.algorithm.get_pairwise_dist(self.get_replay_states_in_graph_as_tensor(),
                                                            aggregate=self.aggregate,
                                                            max_dist=self.max_dist,
                                                            masked=True)
            self.replay_buffer_distances = scipy.sparse.csgraph.floyd_warshall(pdist2, directed=True)

            # plt.matshow(self.replay_buffer_distances)
            # plt.show()

        self.reset_stats()

    def get_replay_states_in_graph_as_tensor(self):
        return torch.cat([x.unsqueeze(0) for x in self.replay_states_in_graph])

    # def build_graph(self):
    #     self.pdist = torch.zeros([self.agent.config.batch_size, self.agent.config.batch_size])
    #
    #     self.build_rb_graph()
    #     if not self.open_loop:
    #         pdist2 = self.agent.algorithm.get_pairwise_dist(self.rb_vec,
    #                                               aggregate=self.aggregate,
    #                                               max_search_steps=self.max_search_steps,
    #                                               masked=True)
    #         self.rb_distances = floyd_warshall(np.split(pdist2, 4, axis=2)[0].squeeze(2), directed=True)  # bugbug currently running on only one batch-slice of the sample

    def _setup(self):
        self.use_qvalue_target = self.config.get('use_qvalue_target') or False

    def __call__(self, state, greedy=False):
        """
        Gets an action based on a state.

        :param state: the current state
        :param greedy:
        :return: the action to take
        """

        # 1. Initial Exploration

        random_action = None

        if self.training:
            # If the replay buffer has not been filled up with random samples
            if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
                # Sample a random action
                random_action = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            elif hasattr(self, 'ag_curiosity'):
                # If we need to relabel states, do so
                state = self.ag_curiosity.relabel_state(state)

        # If we're doing random stuff, return the random action
        if random_action is not None:
            return random_action

        # 2. No longer warm-up

        self.training_steps += 1

        # Occasionally update the graph
        self.graph_update += 1
        if self.graph_update % 10 == 0 or self.graph_update == 1:
            self.build_graph_on_top_of_replay_buffer()

        # Now, self.replay_buffer_graph is the graph over the replay buffer states.
        # print(self.replay_buffer_graph)

        visualize_graph = False
        if visualize_graph:
            nx.draw(self.replay_buffer_graph)
            plt.savefig("rb_graph.png")

        goal = state['desired_goal']
        dist_to_goal = self.agent.algorithm.get_dist_to_goal({k: v for k, v in state.items()})[0][0]

        # if dist_to_goal <= 0:
        #     dist_to_goal = 100000

        if self.open_loop or self.cleanup:
            if state.get('first_step', False): self.initialize_path(state)

            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.replay_buffer_graph.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state['desired_goal'] = waypoint
            dist_to_waypoint = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state['desired_goal'] = waypoint
                dist_to_waypoint = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

            dist_to_goal_via_waypoint = dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
        else:
            # closed loop, replan waypoint at each step
            waypoint, dist_to_goal_via_waypoint = self.get_closest_waypoint(state)

        if (self.no_waypoint_hopping and not self.reached_final_waypoint) or \
                (dist_to_goal_via_waypoint < dist_to_goal) or \
                (dist_to_goal > self.max_dist):
            state['desired_goal'] = waypoint.unsqueeze(0)

            if self.open_loop:
                self.waypoint_attempts += 1
        else:
            state['desired_goal'] = goal

        return self.agent.algorithm.select_action(state)

    def __str__(self):
        s = f'{self.__class__.__name__} (|V|={self.replay_buffer_graph.number_of_nodes()}, |E|={self.replay_buffer_graph.number_of_edges()})'
        return s

    def reset_stats(self):
        """
        Resets all the statistics of the search.

        :return: nothing
        """

        self.stats = dict(
            path_planning_attempts=0,
            path_planning_fails=0,
            graph_search_time=0,
            localization_fails=0,
        )

    def get_stats(self):
        """
        Gets the statistics of the search.

        :return: a dictionary of statistics, containing information such as
                *path_planning_attempts (bugbug explain what each of these means)
                *path_planning_fails
                *graph_search_time
                *localization_fails
        """
        return self.stats

    def set_cleanup(self, cleanup: bool):
        """
        Set the cleanup parameter.
        if True, will prune edges when search fails to reach a waypoint after `attempt_cutoff` steps.

        :param cleanup: whether to prune edges, as described above.
        :return: nothing
        """
        self.cleanup = cleanup

    def get_pairwise_distance_to_states_in_replay_graph(self, state, masked=True):
        """

        :param state: a dictionary.
                      state['observation'] contains the observations, of size [1, 9, 9, 4].
                      state['desired_goal'] contains the goals, of size [1, 9, 9, 4].
        :param masked: whether it's masked or not

        :return: a set of pairwise distances between the starting points and the replay buffer, and
                a set of pairwise distances between the replay buffer and the goal points, as a tuple.
        """

        replay_states = self.get_replay_states_in_graph_as_tensor()

        start_to_replay_buffer_distance = self.agent.algorithm.get_pairwise_dist(state['observation'],
                                                                                 replay_states,
                                                                                 aggregate='mean',
                                                                                 # bugbug get from args
                                                                                 max_dist=self.max_dist,
                                                                                 masked=masked)
        replay_buffer_to_goal_distance = self.agent.algorithm.get_pairwise_dist(replay_states,
                                                                                state['desired_goal'],
                                                                                aggregate='mean',
                                                                                # bugbug get from args
                                                                                max_dist=self.max_dist,
                                                                                masked=masked)
        return start_to_replay_buffer_distance, replay_buffer_to_goal_distance

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` between states in `rb_vec`

        :param state: the current state
        :return: the closest waypoint, and the search distance to it.
        """
        observations_to_replay_buffer_distances, replay_buffer_to_goal_distances = self.get_pairwise_distance_to_states_in_replay_graph(
            state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum([
            np.expand_dims(observations_to_replay_buffer_distances, 2),
            np.expand_dims(self.replay_buffer_distances, 0),
            np.expand_dims(np.transpose(replay_buffer_to_goal_distances), 1)
        ])  # elementwise sum

        # We assume a batch size of 1.
        min_search_dist = np.min(search_dist)
        waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
        waypoint = self.replay_states_in_graph[waypoint_index]

        return waypoint, min_search_dist

    def construct_planning_graph(self, state):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_distance_to_states_in_replay_graph(state)
        planning_graph = self.replay_buffer_graph.copy()

        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb_dist.flatten(), rb_to_goal_dist.flatten())):
            if dist_from_start < self.max_dist:
                planning_graph.add_edge('start', i, weight=dist_from_start)
            if dist_to_goal < self.max_dist:
                planning_graph.add_edge(i, 'goal', weight=dist_to_goal)

        if not np.any(start_to_rb_dist < self.max_dist) or not np.any(
                rb_to_goal_dist < self.max_dist):
            self.stats['localization_fails'] += 1

        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        try:
            self.stats['path_planning_attempts'] += 1
            graph_search_start = time.perf_counter()

            if self.weighted_path_planning:
                path = nx.shortest_path(g2, source='start', target='goal', weight='weight')
            else:
                path = nx.shortest_path(g2, source='start', target='goal')
        except:
            self.stats['path_planning_fails'] += 1
            raise RuntimeError(f'Failed to find path in graph (|V|={g2.number_of_nodes()}, |E|={g2.number_of_edges()})')
        finally:
            graph_search_end = time.perf_counter()
            self.stats['graph_search_time'] += graph_search_end - graph_search_start

        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])

        waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_indices = list(path)[1:-1]
        return waypoint_indices, waypoint_to_goal_dist[1:]

    def initialize_path(self, state):
        self.waypoint_indices, self.waypoint_to_goal_dist_vec = self.get_path(state)
        self.waypoint_counter = 0
        self.waypoint_attempts = 0
        self.reached_final_waypoint = False

    def get_current_waypoint(self):
        waypoint_index = self.waypoint_indices[self.waypoint_counter]
        waypoint = self.replay_states_in_graph[waypoint_index]
        return waypoint, waypoint_index

    def get_waypoints(self):
        waypoints = [self.replay_states_in_graph[i] for i in self.waypoint_indices]
        return waypoints

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        return dist_to_waypoint < self.max_dist


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
                    # print("Updating target")

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
        q_next = self.qvalue_target(next_states.view(self.agent.config.batch_size, -1)).detach()

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
        q = self.qvalue(states.view(self.agent.config.batch_size, -1))

        # Index the rows of the q-values by the batch-list of actions
        # q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))
        q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))

        # Get the squared bellman error
        td_loss = -F.mse_loss(q, target)

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
        projected_distribution = np.zeros(next_distribution.size())

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


def squeeze_dict(dict):
    for key in dict.keys():
        dict[key] = dict[key].squeeze()

    return dict


def get_state_illustration(state):
    state = squeeze_dict(state)

    final_matrix = np.zeros((state['observation'].shape[0], state['observation'].shape[1]))

    dim1, dim2, dim3 = state['observation'].shape

    for i in range(dim1):
        for j in range(dim2):
            if state['observation'][i][j][1] == 1:  # it's the agent
                final_matrix[i][j] = 200
            elif state['observation'][i][j][2] == 1:  # it's a wall
                final_matrix[i][j] = 100

            if state['goal'][i][j][1] == 1:  # it's the goal!
                if final_matrix[i][j] == 200:  # reached it
                    final_matrix[i][j] = 0
                else:
                    final_matrix[i][j] = 50

    return final_matrix


global count
count = 0


def illustrate(state, q_values, directional_q_values):
    global count

    # First, draw what's happening in terms of location and goals
    fig, axs = plt.subplots(3, 3)

    axs[0, 0].axis('off')
    axs[0, 2].axis('off')
    axs[2, 0].axis('off')
    axs[2, 2].axis('off')
    axs[1, 1].matshow(get_state_illustration(state))

    axs[1, 0].set_title("E: " + str(round(float(directional_q_values[0]), 2)))
    axs[1, 0].bar(range(len(q_values[0])), -q_values[0].numpy())  # Left

    axs[2, 1].set_title("E: " + str(round(float(directional_q_values[1]), 2)))
    axs[2, 1].bar(range(len(q_values[1])), -q_values[1].numpy())  # Down

    axs[1, 2].set_title("E: " + str(round(float(directional_q_values[2]), 2)))
    axs[1, 2].bar(range(len(q_values[2])), -q_values[2].numpy())  # Right

    axs[0, 1].set_title("E: " + str(round(float(directional_q_values[3]), 2)))
    axs[0, 1].bar(range(len(q_values[3])), -q_values[3].numpy())  # Up

    fig.suptitle("Non-random step " + str(count))

    plt.savefig("step_" + str(count).zfill(3) + ".png")
    # plt.show()
    plt.close(fig)


def get_weighted_average_of_bins(bin_distribution):
    # sm = F.softmax(-bin_distribution, dim=0)
    sm = bin_distribution

    sum = 0

    for i in range(len(sm)):
        sum += i * sm[i]

    if isinstance(sm, np.ndarray):
        return sum / np.sum(sm)
    else:
        return sum / torch.sum(sm)

    # torch.range(0, -len(bin_distribution), step=-1).dot(bin_distribution)


class SorbDDQN(BaseQLearning):
    def __init__(self, num_atoms, batch_size):
        """
        Creates a Distributional Q Network (https://arxiv.org/abs/1707.06887)

        :param num_atoms: the number of atoms. 5 for SoRB, 51 for Atari
        :param v_max: maximum value for value distribution estimation
        :param v_min: minimum value for value distribution estimation
        """
        super().__init__()

        print("Hi")

        self.num_atoms = num_atoms
        self.batch_size = batch_size
        # self.v_max = v_max
        # self.v_min = v_min
        # self.value_range = torch.tensor(v_max - v_min)
        # self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        # self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    # def __project_next_state_value_distribution(self, states, actions, rewards, next_states, gammas, dones,
    #                                             next_distribution, best_actions):
    #     batch_size = states.size(0)
    #     projected_distribution = torch.zeros(next_distribution.size())
    #
    #     for i in range(batch_size):
    #         for j in range(self.num_atoms):
    #             # Compute the projection of ˆTz_j onto the support {z_i}
    #             Tz_j = (rewards[i] + (1 - dones[i]) * gammas[i] * self.z[j]).clamp(min=self.v_min, max=self.v_max)
    #             b_j = (Tz_j - self.v_min) / self.delta_z
    #             l, u = math.floor(b_j), math.ceil(b_j)
    #
    #             # Distribute probability of ˆTz_j
    #             projected_distribution[i][actions[i].long()][int(l)] += (next_distribution[i][best_actions[i][j]][j] * (
    #                     u - b_j)).squeeze(0)
    #             projected_distribution[i][actions[i].long()][int(u)] += (next_distribution[i][best_actions[i][j]][j] * (
    #                     b_j - l)).squeeze(0)
    #
    #     return projected_distribution

    def get_loss(self, q, q_next, rewards, actions, dones):
        if not isinstance(q, list):
            current_q_list = [q]
            target_q_list = [q_next]
        else:
            current_q_list = q
            target_q_list = q_next

        self.use_distributional_rl = True

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            if self.use_distributional_rl:
                # Index q by actions
                current_q = q.gather(1, actions.unsqueeze(-1).repeat([1, 20]).unsqueeze(1).to(torch.int64))

                # Compute distributional td targets
                target_q_probs = F.softmax(target_q, dim=-1)

                # self.get_weighted_average_of_bins()

                # torch.range(0, -len(bin_distribution), step=-1).dot(bin_distribution)
                max_indices = torch.argmin(
                    torch.sum(torch.range(0, 19)[None, None, :].repeat([2, 4, 1]) * target_q_probs, dim=-1),
                    dim=-1)
                target_q_probs.gather(1, max_indices[:, None, None].repeat([1, 1, 20]))

                # target_q_probs = target_q_probs.gather(1, target_q.mean(2).argmin(1).unsqueeze(-1).repeat([1, 20]).unsqueeze(1))
                # target_q_probs = target_q_probs.mean(dim=-)

                batch_size = target_q_probs.shape[0]
                action_dim = target_q_probs.shape[1]
                one_hot = torch.zeros(batch_size, action_dim, self.num_atoms)
                one_hot[:, :, 0] = 1

                # Calculate the shifted probabilities
                # Fist column: Since episode didn't terminate, probability that the
                # distance is 1 equals 0.
                col_1 = torch.zeros((batch_size, action_dim, 1))
                # Middle columns: Simply the shifted probabilities.
                col_middle = target_q_probs[:, :, :-2]
                # Last column: Probability of taking at least n steps is sum of
                # last two columns in unshifted predictions:
                col_last = torch.sum(target_q_probs[:, :, -2:], dim=-1, keepdim=True)
                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=-1)
                assert one_hot.shape == shifted_target_q_probs.shape

                td_targets = torch.where(dones.bool(), one_hot, shifted_target_q_probs).detach()

                critic_loss = torch.mean(-torch.sum(td_targets * torch.log_softmax(current_q, dim=-1),
                                                    dim=-1))  # https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
            else:
                critic_loss = super().critic_loss(current_q, target_q, rewards, dones)
                raise NotImplementedError()

            critic_loss_list.append(critic_loss)

        critic_loss = torch.mean(torch.stack(critic_loss_list))

        return critic_loss

    def optimize_from_batch(self, states, actions, rewards, next_states, gammas,
                            dones=None):  # bugbug currently dqn optimizer
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
        q_next = self.qvalue_target(next_states.view(self.batch_size, -1)).detach() \
            .view(self.batch_size, 4, self.num_atoms)

        # # Minimum modification to get double q learning to work
        # # (Hasselt, Guez, and Silver, 2016: https://arxiv.org/pdf/1509.06461.pdf)
        # if self.config.double_q:
        #     best_actions = torch.argmax(self.qvalue(next_states), dim=-1, keepdim=True)
        #     q_next = q_next.gather(-1, best_actions)
        # else:
        # q_next = q_next.max(-1, keepdims=True)[0]  # Assuming action dim is the last dimension

        # Get the actual Q function for the real network on the current states
        q = self.qvalue(states.view(self.batch_size, -1)) \
            .view(self.batch_size, 4, self.num_atoms)


        # # Index the rows of the q-values by the batch-list of actions
        # # q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))
        # q = q.gather(-1, actions.unsqueeze(-1).to(torch.int64))
        # q.gather(1, actions.unsqueeze(-1).repeat([1, 10]).unsqueeze(1).to(torch.int64))

        # Get the loss
        loss = self.get_loss(q, q_next, rewards, actions, dones)

        # Clear previous gradients before the backward pass
        self.q_value_optimizer.zero_grad()

        # Run the backward pass
        loss.backward()

        # Grad clipping
        # if self.config.grad_norm_clipping > 0.:
        #     torch.nn.utils.clip_grad_norm_(self.qvalue_params, self.config.grad_norm_clipping)
        # if self.config.grad_value_clipping > 0.:
        #     torch.nn.utils.clip_grad_value_(self.qvalue_params, self.config.grad_value_clipping)

        # Run the update
        self.q_value_optimizer.step()

        return

    def optimize_from_batch_old(self, states, actions, rewards, next_states, gammas,
                                dones=None):  # bugbug currently dqn optimizer
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
        q_next = self.qvalue_target(next_states.view(self.agent.config.batch_size, -1)).detach()

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
        q = self.qvalue(states.view(self.agent.config.batch_size, -1))

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

    def select_action(self, state):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state['observation']),
                goal=torch.FloatTensor(state['desired_goal']),
            )

            q_values = self.get_expected_q_values(state).view(4, 10)  # bugbug get from parameters

            # print(q_values)

            directional_q_values = [self.get_weighted_average_of_bins(q_values[i]) for i in range(q_values.shape[0])]
            # print("Q values for each direction: " + str(directional_q_values))

            global count
            count += 1
            if count > 2000:
                print("Illustrating")
                illustrate(state, q_values, directional_q_values)

            return [int(np.argmin(directional_q_values))]

            # return np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            # return [int(np.argmax(self.get_expected_q_values(state)))]
            # return self.actor(state).cpu().detach().numpy().flatten()

    def get_expected_q_values(self, state, aggregate='mean'):
        first_dim = state['observation'].shape[0]

        # print("Getting expected q values...")

        # Get a list of q values.
        # We use an ensemble of networks, each of which returns a certain set of q values.
        # The q_values_list is a list of the predictions over all the networks.

        q_values = []
        # for network in [self.qvalue, self.qvaluee, self.qvalueee]:
        #     q_value = network(torch.cat((state['observation'], state['goal']), dim=-1)
        #                       .squeeze(1).squeeze(0)
        #                       .view(first_dim, -1))
        #
        #     q_values.append(q_value)

        q_value = self.qvalue(torch.cat((state['observation'], state['goal']), dim=-1)
                              .squeeze(1).squeeze(0)
                              .view(first_dim, -1))
        q_values.append(q_value)

        # If we only have one network in the ensemble, its prediction is the entire list.
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        del q_values

        self.use_distributional_rl = True  # bugbug get from config

        expected_q_values_list = []
        for q_values in q_values_list:
            if self.use_distributional_rl:
                q_probs = F.softmax(q_values, dim=1)
                batch_size = q_probs.shape[0]
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                bin_range = torch.arange(1, self.num_atoms + 1, dtype=torch.float)
                neg_bin_range = -1.0 * bin_range
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat([batch_size, 1])
                assert q_probs.shape == tiled_bin_range.shape
                # Take the inner product between these two tensors
                expected_q_values = torch.sum(q_probs * tiled_bin_range, dim=1, keepdim=True)
                expected_q_values_list.append(expected_q_values)
            else:
                expected_q_values_list.append(-q_values)
                raise NotImplementedError()

        expected_q_values = torch.stack(expected_q_values_list)

        aggregate = 'mean'  # bugbug get from args

        if aggregate is not None:
            if aggregate == 'mean':
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == 'min':
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            else:
                raise ValueError

        if not self.use_distributional_rl:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_q_value = -1.0 * self.num_atoms
            max_q_value = 0.0
            expected_q_values = torch.clamp(expected_q_values, min_q_value, max_q_value)

        return expected_q_values

    def get_expected_q_values_from_flat(self, state, aggregate='mean'):
        first_dim = state.shape[0]

        # print("Getting expected q values...")

        # Get a list of q values.
        # We use an ensemble of networks, each of which returns a certain set of q values.
        # The q_values_list is a list of the predictions over all the networks.

        q_values = []
        # for network in [self.qvalue, self.qvaluee, self.qvalueee]:
        #     q_value = network(torch.cat((state['observation'], state['goal']), dim=-1)
        #                       .squeeze(1).squeeze(0)
        #                       .view(first_dim, -1))
        #
        #     q_values.append(q_value)

        q_value = self.qvalue(state.view(first_dim, -1)).view(first_dim, 4, self.num_atoms)
        q_values.append(q_value)

        # If we only have one network in the ensemble, its prediction is the entire list.
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        del q_values

        self.use_distributional_rl = True  # bugbug get from config

        expected_q_values_list = []
        for q_values in q_values_list:
            if self.use_distributional_rl:
                q_probs = F.softmax(q_values, dim=1)
                batch_size = q_probs.shape[0]
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                bin_range = torch.arange(1, self.num_atoms + 1, dtype=torch.float)
                neg_bin_range = -1.0 * bin_range
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat([batch_size, 4, 1])
                assert q_probs.shape == tiled_bin_range.shape
                # Take the inner product between these two tensors
                expected_q_values = torch.sum(q_probs * tiled_bin_range, dim=0, keepdim=True)
                expected_q_values_list.append(expected_q_values)
            else:
                expected_q_values_list.append(-q_values)
                raise NotImplementedError()

        expected_q_values = torch.stack(expected_q_values_list)

        aggregate = 'mean'  # bugbug get from args

        if aggregate is not None:
            if aggregate == 'mean':
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == 'min':
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            else:
                raise ValueError

        if not self.use_distributional_rl:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_q_value = -1.0 * self.num_atoms
            max_q_value = 0.0
            expected_q_values = torch.clamp(expected_q_values, min_q_value, max_q_value)

        return expected_q_values

    # def critic_loss(self, current_q, target_q, reward, done):
    #     if not isinstance(current_q, list):
    #         current_q_list = [current_q]
    #         target_q_list = [target_q]
    #     else:
    #         current_q_list = current_q
    #         target_q_list = target_q
    #
    #     critic_loss_list = []
    #     for current_q, target_q in zip(current_q_list, target_q_list):
    #         if self.use_distributional_rl:
    #             # Compute distributional td targets
    #             target_q_probs = F.softmax(target_q, dim=1)
    #             batch_size = target_q_probs.shape[0]
    #             one_hot = torch.zeros(batch_size, self.num_bins)
    #             one_hot[:, 0] = 1
    #
    #             # Calculate the shifted probabilities
    #             # Fist column: Since episode didn't terminate, probability that the
    #             # distance is 1 equals 0.
    #             col_1 = torch.zeros((batch_size, 1))
    #             # Middle columns: Simply the shifted probabilities.
    #             col_middle = target_q_probs[:, :-2]
    #             # Last column: Probability of taking at least n steps is sum of
    #             # last two columns in unshifted predictions:
    #             col_last = torch.sum(target_q_probs[:, -2:], dim=1, keepdim=True)
    #             shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=1)
    #             assert one_hot.shape == shifted_target_q_probs.shape
    #             td_targets = torch.where(done.bool(), one_hot, shifted_target_q_probs).detach()
    #
    #             critic_loss = torch.mean(-torch.sum(td_targets * torch.log_softmax(current_q, dim=1),
    #                                                 dim=1))  # https://github.com/tensorflow/tensorflow/issues/21271
    #         else:
    #             critic_loss = super().critic_loss(current_q, target_q, reward, done)
    #         critic_loss_list.append(critic_loss)
    #     critic_loss = torch.mean(torch.stack(critic_loss_list))
    #     return critic_loss

    def get_pairwise_dist(self, obs_vec, goal_vec=None, aggregate='mean', max_dist=7, masked=False):
        """Estimates the pairwise distances.
          obs_vec: Array containing observations
          goal_vec: (optional) Array containing a second set of observations. If
                    not specified, computes the pairwise distances between obs_tensor and
                    itself.
          aggregate: (str) How to combine the predictions from the ensemble. Options
                     are to take the minimum predicted q value (i.e., the maximum distance),
                     the mean, or to simply return all the predictions.
          max_search_steps: (int)
          masked: (bool) Whether to ignore edges that are too long, as defined by
                  max_search_steps.
        """

        # If there's no second vector, compute the pairwise distances
        # between the elements of the first vector and itself.
        if goal_vec is None:
            goal_vec = obs_vec

        # The size of the observation tensor is [batch_size, 9, 9, 4].
        # The size of the goal tensor is also [batch_size, 9, 9, 4].    bugbug could be not square
        # Let's get the batch dimension, so we can use it later:
        obs_batch_size = obs_vec.shape[0]
        goal_batch_size = goal_vec.shape[0]
        # Thus, in order to efficiently compute all the distances,
        # we make a tensor of size [batch_size * batch_size, 9, 9, 8],
        # which contains every pairwise combination (so we need to call
        # the q-value function only once).
        # First, we obtain the tiled (e.g., copied and pasted) version
        # of the observation vector:
        obs_tiled = np.tile(obs_vec, (goal_batch_size, 1, 1, 1))  # size: [batch_size * batch_size, 9, 9, 4]
        # Next, we obtain the repeated (e.g., elementwise expanded)
        # version of the goal vector:
        goal_repeated = np.repeat(goal_vec, obs_batch_size, axis=0)  # size: [batch_size * batch_size, 9, 9, 4]
        # Finally, we get the efficient state tensor:
        state = {'observation': obs_tiled, 'desired_goal': goal_repeated}
        # Which gets us the q-values super quickly:
        dist = self.get_dist_to_goal(state, aggregate=aggregate)
        # Now, since we did the weird thing where we make
        # a batch_size^2-length tensor, we need to convert
        # it back to [batch_size, batch_size, &c].
        dist = dist.reshape(obs_batch_size, goal_batch_size, -1)
        # Now, the distances tensor has shape [batch_size, batch_size, 4], as desired.

        # Old code (Ben)
        # dist_matrix = []
        # for obs_index in range(len(obs_vec)):
        #     obs = np.expand_dims(obs_vec[obs_index], 0)
        #     # obs_repeat_tensor = np.ones_like(goal_vec) * np.expand_dims(obs, 0)
        #     obs_repeat_tensor = np.repeat(obs, len(goal_vec), axis=0)
        #     state = {'observation': obs_repeat_tensor, 'desired_goal': goal_vec}
        #     dist = self.get_dist_to_goal(state, aggregate=aggregate)
        #     dist_matrix.append(dist)
        #
        # pairwise_dist = np.stack(dist_matrix)

        pairwise_dist = dist

        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])
        else:
            # pairwise_dist = np.expand_dims(pairwise_dist, 0)
            pairwise_dist = np.mean(pairwise_dist, axis=-1)

        if masked:
            mask = (pairwise_dist > max_dist)
            return np.where(mask, np.full(pairwise_dist.shape, np.inf), pairwise_dist)
        else:
            return pairwise_dist

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state['observation']),
                goal=torch.FloatTensor(state['desired_goal']),
            )
            q_values = self.get_expected_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy()

    def reset_stats(self):
        self.stats = dict(
            path_planning_attempts=0,
            path_planning_fails=0,
            graph_search_time=0,
            localization_fails=0,
        )

    def get_stats(self):
        return self.stats

    def set_cleanup(self, cleanup):  # if True, will prune edges when fail to reach waypoint after `attempt_cutoff`
        self.cleanup = cleanup

    def build_rb_graph(self):
        g = nx.DiGraph()
        pdist_combined = np.max(self.pdist, axis=0)
        for i, s_i in enumerate(self.rb_vec):
            for j, s_j in enumerate(self.rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        self.g = g

    def get_pairwise_dist_to_rb(self, state, masked=True):
        start_to_rb_dist = self.agent.get_pairwise_dist([state['observation']],
                                                        self.rb_vec,
                                                        aggregate=self.aggregate,
                                                        max_dist=self.max_search_steps,
                                                        masked=masked)
        rb_to_goal_dist = self.agent.get_pairwise_dist(self.rb_vec,
                                                       [state['goal']],
                                                       aggregate=self.aggregate,
                                                       max_dist=self.max_search_steps,
                                                       masked=masked)
        return start_to_rb_dist, rb_to_goal_dist

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum([
            np.expand_dims(obs_to_rb_dist, 2),
            np.expand_dims(self.rb_distances, 0),
            np.expand_dims(np.transpose(rb_to_goal_dist), 1)
        ])  # elementwise sum

        # We assume a batch size of 1.
        min_search_dist = np.min(search_dist)
        waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
        waypoint = self.rb_vec[waypoint_index]

        return waypoint, min_search_dist

    def construct_planning_graph(self, state):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        planning_graph = self.g.copy()

        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb_dist.flatten(), rb_to_goal_dist.flatten())):
            if dist_from_start < self.max_search_steps:
                planning_graph.add_edge('start', i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps:
                planning_graph.add_edge(i, 'goal', weight=dist_to_goal)

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
                rb_to_goal_dist < self.max_search_steps):
            self.stats['localization_fails'] += 1

        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        try:
            self.stats['path_planning_attempts'] += 1
            graph_search_start = time.perf_counter()

            if self.weighted_path_planning:
                path = nx.shortest_path(g2, source='start', target='goal', weight='weight')
            else:
                path = nx.shortest_path(g2, source='start', target='goal')
        except:
            self.stats['path_planning_fails'] += 1
            raise RuntimeError(f'Failed to find path in graph (|V|={g2.number_of_nodes()}, |E|={g2.number_of_edges()})')
        finally:
            graph_search_end = time.perf_counter()
            self.stats['graph_search_time'] += graph_search_end - graph_search_start

        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])

        waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_indices = list(path)[1:-1]
        return waypoint_indices, waypoint_to_goal_dist[1:]

    def initialize_path(self, state):
        self.waypoint_indices, self.waypoint_to_goal_dist_vec = self.get_path(state)
        self.waypoint_counter = 0
        self.waypoint_attempts = 0
        self.reached_final_waypoint = False

    def get_current_waypoint(self):
        waypoint_index = self.waypoint_indices[self.waypoint_counter]
        waypoint = self.rb_vec[waypoint_index]
        return waypoint, waypoint_index

    def get_waypoints(self):
        waypoints = [self.rb_vec[i] for i in self.waypoint_indices]
        return waypoints

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        return dist_to_waypoint < self.max_search_steps
