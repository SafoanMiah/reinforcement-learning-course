import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import warnings

warnings.filterwarnings("ignore")

model = PPO.load('projects\models\sub_8_model.zip')

enviroment_name = 'CarRacing'
env = DummyVecEnv([lambda: gym.make(enviroment_name, render_mode="human")])

for episode in range(5):
    obs = env.reset()
    score = 0
    done = False
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        score += reward
        
    print(f"Episode: {episode + 1}, Score: {score}")