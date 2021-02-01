import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    #env = gym.make('CartPole-v1')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=5, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    # Play n episodes
    for i in range(n_games):
        done = False
        # Get 1st state (observation) from gym env
        observation = env.reset()

        score = 0

        # Play episode until reaching a terminal state
        while not done:
            # Take an epsilon-greedy approach to select an action
            action = agent.choose_action(observation)
            # Get next state, reward, done flag and info from environment
            observation_, reward, done, info = env.step(action)
            # Increment score for performance monitoring
            score += reward

            if not load_checkpoint:
                # Store transition in replay buffer
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                # Execute a training step for the agent
                agent.learn()

            # Set state to next state
            observation = observation_

            n_steps += 1

        # Append score, n_steps and compute avg_score at the end of an episode
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])

        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        # Save main and target models if avg_score improved
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)
