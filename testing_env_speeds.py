import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    enviroment_name = "CartPole-v1"

    def make_env():
        return gym.make(enviroment_name)

    # Function to benchmark training and evaluate rewards
    def benchmark(env, device, total_timesteps=500):
        print(f"Running training on {type(env).__name__} with device: {device}...")
        start_time = time.time()
        model = PPO('MlpPolicy', env, device=device, verbose=0)  # Set device (CPU or GPU)
        model.learn(total_timesteps=total_timesteps)  # Train the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
        env.close()
        end_time = time.time()
        return mean_reward, std_reward, end_time - start_time

    # 1. Test with DummyVecEnv
    print("Testing with DummyVecEnv...")
    dummy_env = DummyVecEnv([lambda: make_env()])
    mean_reward_dummy, std_reward_dummy, time_dummy = benchmark(dummy_env, device="cuda", total_timesteps=500)

    # 2. Test with SubprocVecEnv 4 cores
    print("Testing with SubprocVecEnv 4 instances...")
    subproc_env_4 = SubprocVecEnv([lambda: make_env() for _ in range(4)])  # Use 4 parallel environments
    mean_reward_subproc_4, std_reward_subproc_4, time_subproc_4 = benchmark(subproc_env_4, device="cuda", total_timesteps=500)
    
    # 3. Test with SubprocVecEnv 10 cores
    print("Testing with SubprocVecEnv 10 instances...")
    subproc_env_10 = SubprocVecEnv([lambda: make_env() for _ in range(10)])  # Use 10 parallel environments
    mean_reward_subproc_10, std_reward_subproc_10, time_subproc_10 = benchmark(subproc_env_10, device="cuda", total_timesteps=500)

    # Compare rewards and times
    print(f"Mean Reward with DummyVecEnv: {mean_reward_dummy:.2f} ± {std_reward_dummy:.2f} (Time: {time_dummy:.2f} seconds)")
    print(f"Mean Reward with SubprocVecEnv 4 instances: {mean_reward_subproc_4:.2f} ± {std_reward_subproc_4:.2f} (Time: {time_subproc_4:.2f} seconds)")
    print(f"Mean Reward with SubprocVecEnv 10 instances: {mean_reward_subproc_10:.2f} ± {std_reward_subproc_10:.2f} (Time: {time_subproc_10:.2f} seconds)")
