import numpy as np
import random
from collections import deque
import yfinance as yf # Import yfinance

# --- 1. Environment Definition ---
# A market environment using real historical stock data.
# The agent's goal is to maximize its portfolio value.

class MarketEnvironment:
    def __init__(self, stock_symbol='AAPL', start_date='2020-01-01', end_date='2021-01-01',
                 initial_cash=10000, transaction_cost_rate=0.001): # 0.1% transaction cost
        """
        Initializes the market environment with real stock data.

        Args:
            stock_symbol (str): Ticker symbol for the stock.
            start_date (str): Start date for historical data.
            end_date (str): End date for historical data.
            initial_cash (float): Starting cash in the portfolio.
            transaction_cost_rate (float): Percentage cost per transaction (buy/sell).
        """
        self.stock_symbol = stock_symbol
        self.initial_cash = initial_cash
        self.transaction_cost_rate = transaction_cost_rate

        # Fetch historical data
        print(f"Fetching historical data for {stock_symbol} from {start_date} to {end_date}...")
        self.data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        if self.data.empty:
            raise ValueError(f"No data found for {stock_symbol} between {start_date} and {end_date}. "
                             "Please check the symbol and dates.")
        self.prices = self.data['Close'].values
        self.num_steps_available = len(self.prices) - 1 # Number of trading days available in data

        self.current_step = 0
        self.cash = initial_cash
        self.shares_held = 0
        self.stock_price = self.prices[0] # Current day's closing price
        self.previous_stock_price = self.prices[0] # Used for calculating price change
        self.portfolio_value = self.cash + (self.shares_held * self.stock_price)

        # State discretization parameters (tuned for this simplified example)
        # These are now dynamic based on potential portfolio growth
        self.cash_bin_size = 5000 # Larger bin size for larger initial cash
        self.shares_bin_size = 10 # Larger shares bin size
        self.price_change_range = 0.05 # Max price change considered for binning (+/- 5%)
        self.price_change_bin_count = 11 # Number of bins for price change (e.g., -5% to +5% in 1% steps)

    def reset(self, num_steps_per_episode):
        """
        Resets the environment to a random starting point for a new episode.
        This helps the agent learn from various market conditions.

        Args:
            num_steps_per_episode (int): The number of steps an episode will run for.
        Returns:
            tuple: Initial state (discretized cash, shares_held, price_change).
        """
        # Start at a random point in the historical data, leaving enough steps for an episode
        # Ensure there's enough data for at least `num_steps_per_episode` from the start_idx
        max_start_idx = len(self.prices) - 1 - num_steps_per_episode
        if max_start_idx < 0:
            max_start_idx = 0 # If data is too short, start from beginning

        start_idx = random.randint(0, max_start_idx)
        self.current_step = start_idx
        self.cash = self.initial_cash
        self.shares_held = 0
        self.stock_price = self.prices[self.current_step]
        self.previous_stock_price = self.prices[self.current_step] # Initialize previous price
        self.portfolio_value = self.cash + (self.shares_held * self.stock_price)
        return self._get_state()

    def _get_state(self):
        """
        Returns the current discretized state of the environment.
        State representation: (discretized_cash, discretized_shares_held, discretized_price_change)
        """
        # Discretize cash
        cash_bin = int(self.cash // self.cash_bin_size)
        # Cap cash bin to prevent excessively large Q-table dimensions
        # Assuming max cash could reach 200,000 for 100k initial -> 200000 / 5000 = 40 bins
        # Let's set a slightly higher cap to allow for growth
        cash_bin = min(cash_bin, 60) # Assuming max cash bin of 60 (e.g., up to $300,000)

        # Discretize shares held
        shares_bin = int(self.shares_held // self.shares_bin_size)
        # Cap shares bin
        # If stock price is 200, 100k cash buys 500 shares. 500 / 10 = 50 bins.
        shares_bin = min(shares_bin, 60) # Assuming max shares bin of 60 (e.g., up to 600 shares)

        # Calculate and discretize price change
        if self.previous_stock_price == 0: # Avoid division by zero at very start if price is 0
            price_change_percent = 0.0
        else:
            price_change_percent = (self.stock_price - self.previous_stock_price) / self.previous_stock_price

        # Normalize price change to a range [-price_change_range, price_change_range]
        # Then map to an integer bin
        normalized_change = np.clip(price_change_percent, -self.price_change_range, self.price_change_range)
        # Map to 0 to price_change_bin_count - 1
        price_change_bin = int(((normalized_change + self.price_change_range) / (2 * self.price_change_range)) * (self.price_change_bin_count - 1))
        price_change_bin = max(0, min(price_change_bin, self.price_change_bin_count - 1)) # Ensure within bounds

        return (cash_bin, shares_bin, price_change_bin)

    def step(self, action):
        """
        Applies an action to the environment and returns the new state, reward, and done flag.

        Args:
            action (int): The action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            tuple: (next_state, reward, done, info_dict).
        """
        prev_portfolio_value = self.portfolio_value
        self.previous_stock_price = self.stock_price # Store current price as previous for next step

        self.current_step += 1
        done = self.current_step >= len(self.prices) # End of historical data
        if done:
            # If done, no further price movement, just return current state
            # This handles the case where the episode ends exactly at the last data point
            reward = 0 # No further reward
            return self._get_state(), reward, done, {}

        # Move to the next day's price
        self.stock_price = self.prices[self.current_step]

        # Execute action
        if action == 1:  # Buy
            # Try to buy a fixed number of shares (e.g., 10 shares, adjusted for larger capital)
            shares_to_trade = 10
            cost = shares_to_trade * self.stock_price
            transaction_fee = cost * self.transaction_cost_rate
            total_cost = cost + transaction_fee

            if self.cash >= total_cost:
                self.cash -= total_cost
                self.shares_held += shares_to_trade
            # else: action is effectively a hold if not enough cash
        elif action == 2:  # Sell
            # Try to sell a fixed number of shares (e.g., 10 shares)
            shares_to_trade = 10
            if self.shares_held >= shares_to_trade:
                revenue = shares_to_trade * self.stock_price
                transaction_fee = revenue * self.transaction_cost_rate
                self.cash += (revenue - transaction_fee)
                self.shares_held -= shares_to_trade
            # else: action is effectively a hold if not enough shares
        # Action 0: Hold (do nothing)

        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares_held * self.stock_price)

        # Calculate reward: Change in portfolio value
        reward = self.portfolio_value - prev_portfolio_value

        return self._get_state(), reward, done, {}

# --- Replay Buffer for Experience Replay (Key for DQN) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Ensure we don't try to sample more than available
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

    def __len__(self):
        return len(self.buffer)

# --- 2. Agent Definition (Q-Learning Agent with Replay Buffer - conceptual step towards DQN) ---

class QLearningAgent:
    def __init__(self, state_space_dimensions, action_space_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.999, replay_buffer_capacity=10000, batch_size=64):
        """
        Initializes the Q-Learning Agent.

        Args:
            state_space_dimensions (tuple): Dimensions for the Q-table (e.g., (cash_bins, shares_bins, price_change_bins)).
            action_space_size (int): Number of possible actions (e.g., 3 for Hold, Buy, Sell).
            learning_rate (float): Alpha (how much new information overrides old).
            discount_factor (float): Gamma (importance of future rewards).
            exploration_rate (float): Epsilon (probability of taking a random action).
            min_exploration_rate (float): Minimum value for epsilon.
            exploration_decay_rate (float): Rate at which epsilon decays per step.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            batch_size (int): Number of experiences to sample from the buffer for learning.
        """
        self.state_space_dimensions = state_space_dimensions
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.batch_size = batch_size

        # Initialize Q-table with zeros. Q[state_idx_tuple][action_idx] = value
        self.q_table = np.zeros(state_space_dimensions + (action_space_size,))

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): Current state of the environment (discretized).

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_space_size) # Explore: choose random action
        else:
            # Exploit: choose action with max Q-value for current state
            state_indices = tuple(min(s, dim - 1) for s, dim in zip(state, self.state_space_dimensions))
            return np.argmax(self.q_table[state_indices])

    def store_experience(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Updates the Q-table (or Q-network) using a batch of experiences from the replay buffer.
        This is where the learning happens.
        """
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough experiences to learn yet

        # Sample a batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)

        # For Q-table, iterate through the batch
        for i in range(len(states)): # Iterate over the actual batch size sampled
            state_idx = tuple(min(s, dim - 1) for s, dim in zip(states[i], self.state_space_dimensions))
            next_state_idx = tuple(min(s, dim - 1) for s, dim in zip(next_states[i], self.state_space_dimensions))

            current_q = self.q_table[state_idx + (actions[i],)]
            max_next_q = np.max(self.q_table[next_state_idx])

            # Bellman equation update
            # If done, future reward is 0
            target_q = rewards[i] + self.discount_factor * max_next_q * (1 - int(dones[i]))
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_table[state_idx + (actions[i],)] = new_q

    def decay_exploration_rate(self):
        """Decays the exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

# --- 3. Training Loop ---
# The main loop where the agent interacts with the environment and learns.

def train_agent(env, agent, num_episodes=5000, steps_per_episode=50):
    """
    Trains the RL agent.

    Args:
        env (MarketEnvironment): The market environment.
        agent (QLearningAgent): The RL agent.
        num_episodes (int): Number of episodes to train for.
        steps_per_episode (int): Number of steps (trading days) in each episode.
    """
    rewards_per_episode = []
    portfolio_values_per_episode = []

    print(f"Starting training for {num_episodes} episodes with {steps_per_episode} steps per episode...")

    for episode in range(num_episodes):
        state = env.reset(steps_per_episode) # Reset environment to a random starting point in historical data
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done) # Store experience
            agent.learn() # Learn from a batch of experiences
            state = next_state
            episode_reward += reward
            step_count += 1

            # If the environment runs out of data before steps_per_episode, mark as done
            if env.current_step >= env.num_steps_available:
                done = True

        agent.decay_exploration_rate() # Decay exploration rate after each episode

        rewards_per_episode.append(episode_reward)
        portfolio_values_per_episode.append(env.portfolio_value)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_portfolio_value = np.mean(portfolio_values_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Avg Portfolio Value (last 100): {avg_portfolio_value:.2f} | Epsilon: {agent.exploration_rate:.4f} | Buffer Size: {len(agent.replay_buffer)}")

    print("Training complete!")
    return rewards_per_episode, portfolio_values_per_episode

# --- 4. Simulation/Evaluation ---
# Evaluate the trained agent's performance.

def evaluate_agent(env, agent, num_eval_episodes=100, steps_per_episode=50):
    """
    Evaluates the performance of the trained agent.

    Args:
        env (MarketEnvironment): The market environment.
        agent (QLearningAgent): The trained RL agent.
        num_eval_episodes (int): Number of episodes to evaluate for.
        steps_per_episode (int): Number of steps (trading days) in each evaluation episode.
    """
    total_portfolio_values = []
    agent.exploration_rate = 0 # Turn off exploration for evaluation

    print(f"\nStarting evaluation for {num_eval_episodes} episodes with {steps_per_episode} steps per episode...")

    for episode in range(num_eval_episodes):
        state = env.reset(steps_per_episode)
        done = False
        step_count = 0
        while not done and step_count < steps_per_episode:
            action = agent.choose_action(state) # Agent acts greedily
            state, _, done, _ = env.step(action)
            step_count += 1
            if env.current_step >= env.num_steps_available:
                done = True # Ensure termination if data runs out

        total_portfolio_values.append(env.portfolio_value)

    avg_final_portfolio_value = np.mean(total_portfolio_values)
    print(f"Evaluation complete!")
    print(f"Average final portfolio value over {num_eval_episodes} episodes: {avg_final_portfolio_value:.2f}")
    print(f"Maximum final portfolio value: {np.max(total_portfolio_values):.2f}")
    print(f"Minimum final portfolio value: {np.min(total_portfolio_values):.2f}")

# --- Main Execution ---
if __name__ == "__main__":
    # Define environment parameters
    stock_symbol = 'AAPL' # Defaulting to AAPL for this simulation
    start_date = '2020-01-01'
    end_date = '2024-01-01' # More recent data
    initial_cash = 100000 # User's requested initial cash
    transaction_cost_rate = 0.001 # 0.1%

    # --- State Discretization Parameters ---
    # These define the dimensions of the Q-table. Adjusted for larger initial cash.
    cash_max_bin = 60 # Represents cash up to 60 * cash_bin_size (e.g., 60 * 5000 = $300,000)
    shares_max_bin = 60 # Represents shares up to 60 * shares_bin_size (e.g., 60 * 10 = 600 shares)
    price_change_bins = 11 # From MarketEnvironment: -5% to +5% in 1% steps (11 bins)

    state_space_dimensions = (cash_max_bin + 1, shares_max_bin + 1, price_change_bins)
    action_space_size = 3 # 0: Hold, 1: Buy, 2: Sell

    # Initialize environment and agent
    env = MarketEnvironment(stock_symbol=stock_symbol, start_date=start_date, end_date=end_date,
                            initial_cash=initial_cash, transaction_cost_rate=transaction_cost_rate)

    agent = QLearningAgent(state_space_dimensions, action_space_size,
                           learning_rate=0.001,
                           discount_factor=0.99,
                           exploration_rate=1.0,
                           min_exploration_rate=0.01,
                           exploration_decay_rate=0.9995, # Slightly slower decay for more training
                           replay_buffer_capacity=100000, # Larger buffer for more experiences
                           batch_size=128)

    # Train the agent
    # Use a fixed number of steps per episode to ensure consistent episode length
    steps_per_episode_train = 100 # Longer episodes for more learning per episode
    num_episodes_train = 20000 # Increased episodes for better learning with larger state space
    rewards, portfolio_values = train_agent(env, agent, num_episodes=num_episodes_train, steps_per_episode=steps_per_episode_train)

    # Evaluate the trained agent
    steps_per_episode_eval = 100
    num_eval_episodes = 1000 # More evaluation episodes for better average
    evaluate_agent(env, agent, num_eval_episodes=num_eval_episodes, steps_per_episode=steps_per_episode_eval)

    # You can plot rewards and portfolio values over time to visualize learning
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(rewards)
    # plt.title('Rewards per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(portfolio_values)
    # plt.title('Final Portfolio Value per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Portfolio Value')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
