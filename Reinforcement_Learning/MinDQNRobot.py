import numpy as np
import torch
from torch_py.MinDQNRobot import MinDQNRobot
import time

class Robot(MinDQNRobot):
    def __init__(self, maze):
        """
        Initialize the Robot class.
        :param maze: Maze object
        """
        super(Robot, self).__init__(maze)
        self.maze = maze
        self.epsilon = 0  # Exploration rate
        self.maze.set_reward(reward={
            "hit_wall": 10.,  # Penalty for hitting a wall
            "destination": -self.maze.maze_size ** 2 * 4.,  # Reward for reaching the destination
            "default": 1.,  # Default reward for other actions
        })
        self.memory.build_full_view(maze=maze)  # Build the full view of the maze
        self.train()  # Train the robot and store the loss values

    def train(self):
        """
        Train the robot until it can solve the maze.
        :return: List of loss values during training
        """
        loss_list = []
        batch_size = len(self.memory)  # Size of the memory batch
        start = time.time()  # Start time for training

        while True:
            loss = self._learn(batch=batch_size)  # Learn from the batch
            loss_list.append(loss)  # Append the loss to the list
            self.reset()  # Reset the robot's state
            if self._is_training_complete():  # Check if training is complete
                print('Training time: {:.2f} s'.format(time.time() - start))  # Print the training time
                return loss_list  # Return the list of loss values

    def _is_training_complete(self):
        """
        Check if the training is complete by testing the robot.
        :return: Boolean indicating if training is complete
        """
        for _ in range(self.maze.maze_size ** 2 - 1):
            _, reward = self.test_update()  # Test the robot's update
            if reward == self.maze.reward["destination"]:  # Check if the robot reached the destination
                return True
        return False

    def train_update(self):
        """
        Update the robot's state and action during training.
        :return: Action taken and reward received
        """
        state = self.sense_state()  # Sense the current state
        action = self._choose_action(state)  # Choose an action based on the state
        reward = self.maze.move_robot(action)  # Move the robot and get the reward
        return action, reward

    def test_update(self):
        """
        Update the robot's state and action during testing.
        :return: Action taken and reward received
        """
        state = np.array(self.sense_state(), dtype=np.int16)  # Sense the current state
        state = torch.from_numpy(state).float().to(self.device)  # Convert state to tensor
        self.eval_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()  # Get Q-values from the model
        action = self.valid_action[np.argmin(q_value).item()]  # Choose the action with the minimum Q-value
        reward = self.maze.move_robot(action)  # Move the robot and get the reward
        return action, reward

if __name__ == "__main__":
    from Maze import Maze
    from Runner import Runner

    epoch = 10  # Number of training epochs
    maze_size = 5  # Size of the maze
    training_per_epoch = int(maze_size * maze_size * 1.5)  # Number of training steps per epoch

    maze = Maze(maze_size=maze_size)  # Create a maze
    robot = Robot(maze)  # Create a robot
    runner = Runner(robot)  # Create a runner
    runner.run_training(epoch, training_per_epoch)  # Run the training

    # Generate a gif of the training process
    runner.generate_gif(filename="results/dqn_size10.gif")
    runner.plot_results()  # Plot the training results

    robot.reset()  # Reset the robot
    for _ in range(25):
        action, reward = robot.test_update()  # Test the robot's update
        print("action:", action, "reward:", reward)
        if reward == maze.reward["destination"]:  # Check if the robot reached the destination
            print("success")
            break