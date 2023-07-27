from unityagents import UnityEnvironment


class NavigationEnv:

    def __init__(self, file_path: str):

        # Load the environment
        self.unityEnv = UnityEnvironment(file_name=file_path)

        # get the default brain
        self.brain_name = self.unityEnv.brain_names[0]
        self.brain = self.unityEnv.brains[self.brain_name]

        # Get sizes
        self.action_size = self.brain.vector_action_space_size

        env_info = self.unityEnv.reset(train_mode=True)[self.brain_name]
        self.state_size = len(env_info.vector_observations[0])

    def step(self, action: int) -> (list, float, bool, dict):
        # take the action to the environment
        env_info = self.unityEnv.step(action)[self.brain_name]

        # get the current state
        next_state = env_info.vector_observations[0]

        # get the reward
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        return next_state, reward, done, env_info

    def reset(self, train_mode: bool = True) -> (list, float):
        # Reset the environment
        env_info = self.unityEnv.reset(train_mode=train_mode)[self.brain_name]

        # get the current state
        state = env_info.vector_observations[0]
        reward = 0

        return state, reward

