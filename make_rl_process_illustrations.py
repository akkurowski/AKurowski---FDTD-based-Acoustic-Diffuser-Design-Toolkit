from _imports import *
import pandas as pd
import matplotlib.pyplot as plt
import os

# konfiguracja procedur bazujących na AI
CONFIG_PATH_AI  = '_settings/ai_default.ini'
settings_ai     = read_config(CONFIG_PATH_AI)

os.system('cls')

_, consolidated_data = obtain_replay_folder_contents(settings_ai)
agents       = get_replay_agent_fingerprints(consolidated_data)
agents_names = [agent[1] for agent in agents]
agents_names = sorted(agents_names)

diff_df   = {}
reward_df = {}
min_reward    =  np.inf
min_diffusion =  np.inf
max_reward    = -np.inf
max_diffusion = -np.inf
for agent_name in agents_names:
    agent_diffusions, _, mean_reward = extract_agents_data(agent_name, consolidated_data)
    diff_df.update({agent_name:agent_diffusions})
    reward_df.update({agent_name:mean_reward})
    
    if np.min(mean_reward) < min_reward: min_reward = np.min(mean_reward)
    if np.max(mean_reward) > max_reward: max_reward = np.max(mean_reward)
    
    if np.min(agent_diffusions) < min_diffusion: min_diffusion = np.min(agent_diffusions)
    if np.max(agent_diffusions) > max_diffusion: max_diffusion = np.max(agent_diffusions)

# ---------------------------------------------------

diff_df = pd.DataFrame.from_dict(diff_df, orient='index').transpose()
reward_df = pd.DataFrame.from_dict(reward_df, orient='index').transpose()

diff_joint_It2Smaller0C_and0D = diff_df['It2Smaller0C_rnd']
rwrd_joint_It2Smaller0C_and0D = reward_df['It2Smaller0C_rnd']
n_episodes = len(diff_joint_It2Smaller0C_and0D)

diff_It2Smaller0C = diff_joint_It2Smaller0C_and0D[0:n_episodes-1:2].reset_index(drop=True)
diff_It2Smaller0D = diff_joint_It2Smaller0C_and0D[1:n_episodes-1:2].reset_index(drop=True)

rwrd_It2Smaller0C = rwrd_joint_It2Smaller0C_and0D[0:n_episodes-1:2].reset_index(drop=True)
rwrd_It2Smaller0D = rwrd_joint_It2Smaller0C_and0D[1:n_episodes-1:2].reset_index(drop=True)

diff_df = diff_df.drop(columns=['It2Smaller0C_rnd'])
diff_df = diff_df.assign(It2Smaller0C_rnd = diff_It2Smaller0C)
diff_df = diff_df.assign(It2Smaller0D_rnd = diff_It2Smaller0D)

reward_df = reward_df.drop(columns=['It2Smaller0C_rnd'])
reward_df = reward_df.assign(It2Smaller0C_rnd = rwrd_It2Smaller0C)
reward_df = reward_df.assign(It2Smaller0D_rnd = rwrd_It2Smaller0D)

# ---------------------------------------------------

diff_joint_It2Smaller0C_and0D = diff_df['It2Smaller0C']
rwrd_joint_It2Smaller0C_and0D = reward_df['It2Smaller0C']
n_episodes = len(diff_joint_It2Smaller0C_and0D)

diff_It2Smaller0C = diff_joint_It2Smaller0C_and0D[0:n_episodes-1:2].reset_index(drop=True)
diff_It2Smaller0D = diff_joint_It2Smaller0C_and0D[1:n_episodes-1:2].reset_index(drop=True)

rwrd_It2Smaller0C = rwrd_joint_It2Smaller0C_and0D[0:n_episodes-1:2].reset_index(drop=True)
rwrd_It2Smaller0D = rwrd_joint_It2Smaller0C_and0D[1:n_episodes-1:2].reset_index(drop=True)

diff_df = diff_df.drop(columns=['It2Smaller0C'])
diff_df = diff_df.assign(It2Smaller0C = diff_It2Smaller0C)
diff_df = diff_df.assign(It2Smaller0D = diff_It2Smaller0D)

reward_df = reward_df.drop(columns=['It2Smaller0C'])
reward_df = reward_df.assign(It2Smaller0C = rwrd_It2Smaller0C)
reward_df = reward_df.assign(It2Smaller0D = rwrd_It2Smaller0D)

# ---------------------------------------------------

for agent_name in diff_df.columns:  
    print(f"visualizing: {agent_name}")
    plt.figure(figsize=(10,5.5))
    ax     = plt.gca()
    sec_ax = ax.twinx()

    ax.plot(diff_df[agent_name], label='episodic diffusion coefficient [-]', color='navy')
    ax.set_ylabel('best episodic diffusion coefficient [-]', color='navy', fontsize=16)
    ax.set_ylim([min_diffusion,1])
    
    sec_ax.plot(reward_df[agent_name], label='episodic reward [-]', color='darkgreen')
    sec_ax.set_ylabel('episodic reward [-]', color='darkgreen', fontsize=14)
    sec_ax.set_ylim([min_reward,max_reward])
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=15)
    sec_ax.tick_params(axis='y', labelsize=15)
    plt.grid()
    ax.set_xlabel('episode number [-]', fontsize=15)
    plt.savefig(f"trn_progress_{agent_name}.png")
    plt.tight_layout()
    plt.close()