# DIAYN_VCL
## Acknowledgments
Our implementation of DIAYN is based on the code provided by skywalker0803r in @GitHub https://github.com/skywalker0803r/DIAYN, and 
Yunzhong Hou in @GitHub https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/hierarchical_agents/DIAYN.py. 
Based on their works, we implemented the DIAYN and DIAYN_VCL through desiging gym_style wrapper(seeing env_warpper.py). Here we sincerely thanks to their wonderful job.

The VCL algorithm(seeing ./discriminator/vcc_nn.py) we merged is provided by NixGD et al. in @GitHub  https://github.com/NixGD/variational-continual-learning and 
also referred the original implementation from Siddharth Swaroop @GitHub  https://github.com/nvcuong/variational-continual-learning. 

The urlb based DIAYN_VCL is implemented based on the DIAYN provided by them. Seeing https://github.com/rll-research/url_benchmark
## Environment setting
Here we use anaconda to manage our environment. 
Firstly, you can create a new environment using command：
```bash
conda create -f environment.yml
```

Then we need to activate it:
```bash
conda activate mmaml_final
```
 If you find there are some packages missing, you can follow the error messages and install them by pip or conda.
## Experiments
In this gym based environments, we mainly did two sets of experiments. The one is to train the agents firstly, and then 
visualize the learned skills in form of videos and plotted statistical figures. The other is to pre-train the agents in the 
environments without rewards, and then fine-tuning them with extrinsic rewards. Here we will introduce several command  line 
to run the environments and get the results.
### Training
Experiment 1 in ./experiment01.py. The trained models are stored in folder: ./trainedmodel
- Arguments description 
    - --mode: 'train', 'train_vcl', 'eval', 'view'
      - -- 'train': train the diayn agent: eg: 
        ```bash 
        python experiment01.py --mode train -- env_name Ant-v2 --total_timestep 1000000
        ```
      - -- 'train_vcl': train the _vcl agent: eg: 
        ```bash 
        python experiment01.py --mode train_vcl -- env_name Ant-v2 --total_timestep 1000000
        ```
      - -- 'eval': print the episode rewards of the learned skills： eg: 
        ```bash  
        python experiment01.py --mode eval --env_name Hopper-v3 --agent_path trainedmodel/Hopper-v3-500000.zip
        ```
      - -- 'view'： show the video of learned skills:  
        ```bash 
        python experiment01.py --mode view --env_name Hopper-v3 --agent_path trainedmodel/Hopper-v3-1000000.zip
        ```
    - --env_name: the environments chosen to play:['MountainCarContinuous-v0','InvertedPendulum-v2','Hopper-v3',‘HalfCheetah-v2’,‘Ant-v2’]
    - --total_timestep: total training steps
    - --agent_path：the learned models' path: ex: trainedmodel/Hopper-v3-500000.zip
    
      
### Visilization of learned skills
These results are got by running python file: visiualization.py

- Arguments description 
    - --mode:  'stack', 'hist', 'video', 'training_curve'
      - -- 'stack': get the training dynamic figure: eg: 
        ```bash 
        python visiualization.py --mode stack --env_name Hopper-v3 --agent DIAYN_VCL
        ```
      - -- 'hist': get the distribution of learned skills: eg: 
        ```bash 
        python visiualization.py --mode stack --env_name Hopper-v3 --agent DIAYN_VCL
        ```
      - -- 'video': record the video： eg: 
        ```bash  
        python visiualization.py --mode video  --env_name MountainCarContinuous-v0 --training_steps 200000
        ```
      - -- 'training_curve'： show the training curve:  
        ```bash 
        python visiualization.py --mode training_curve  --env_name hopper
        ```
    - --env_name: the environments chosen to play:['MountainCarContinuous-v0','InvertedPendulum-v2','Hopper-v3',‘HalfCheetah-v2’,‘Ant-v2’]
    - --total_timestep: total training steps
    - --agent_path：the learned models' path: ex: trainedmodel/Hopper-v3-500000.zip
    - --agent：DIAYN or DIAYN_VCL

### Some videos
#### MountainCarContinuous
<p float="left">
  <img src="/results/video/MountainCarContinuous-v0_0_400000.gif" width="200" />
  <img src="/results/video/MountainCarContinuous-v0_0_50000.gif" width="200" /> 
  <img src="/results/video/MountainCarContinuous-v0_10.gif" width="200" />
</p>

#### InvertedPendulum
<p float="left">
  <img src="/results/video/InvertedPendulum-v2_10_200000.gif" width="200" />
  <img src="/results/video/InvertedPendulum-v2_15_200000.gif" width="200" /> 
  <img src="/results/video/InvertedPendulum-v2_20_200000.gif" width="200" />
</p>


#### Hopper
<p float="left">
  <img src="/results/video/Hopper-v3_1_500000.gif" width="200" />
  <img src="/results/video/Hopper-v3_23_500000.gif" width="200" /> 
  <img src="/results/video/Hopper-v3_4_500000.gif" width="200" />
</p>

#### HalfCheetah
<p float="left">
  <img src="/results/video/HalfCheetah-v2_0_750000.gif" width="200" />
  <img src="/results/video/HalfCheetah-v2_14_750000.gif" width="200" /> 
  <img src="/results/video/HalfCheetah-v2_39_750000.gif" width="200" />
</p>

#### Ant
<p float="left">
  <img src="/results/video/Ant-v2_27_1000000.gif" width="200" />
  <img src="/results/video/Ant-v2_27_500000.gif" width="200" /> 
  <img src="/results/video/Ant-v2_7_1000000.gif" width="200" />
</p>

### Pre-training and fine-tuning 
The experiments of this setup are run in ./experiment02.py. In this experiment, we first pre-train the agent and store its
model, then load the trained models and continually train them.
- Arguments description 
    - --mode:  'train', 'train_vcl',  'train_random', 'train_choosed_skill'
      - -- 'train': pre-train and fine-tuning on DIAYN: eg: 
        ```bash 
        python experiment02.py --mode train --env_name HalfCheetah-v2 --total_pretraining_timesteps 1000000 --total_training_timesteps 200000
        ```
      - -- 'train_vcl': pre-train and fine-tuning on DIAYN_VCL: eg: 
        ```bash 
        python experiment02.py --mode train_vcl --env_name HalfCheetah-v2 --total_pretraining_timesteps 1000000 --total_training_timesteps 200000
        ```
      - -- 'train_random': fine-tuning on agents with random initialization： eg: 
        ```bash  
        python experiment02.py --mode train_random --env_name Hopper-v3  --total_pretraining_timesteps 1000 --total_training_timesteps 200000
        ```
      - -- 'train_choosed_skill'：manually choose skill and fine-tuning:  
        ```bash 
        python experiment02.py --mode train_choosed_skill --env_name HalfCheetah-v2 --total_pretraining_timesteps 100000  --total_training_timesteps 500000 --agent DIAYN_VCL --choosed_skill 9
        ```
    - --env_name: the environments chosen to play:['MountainCarContinuous-v0','InvertedPendulum-v2','Hopper-v3',‘HalfCheetah-v2’,‘Ant-v2’]
    - --total_pretraining_timesteps: total pre-training steps
    - --total_training_timesteps: total fine-tuning steps  
    - --agent_path：the learned models' path: ex: trainedmodel/Hopper-v3-500000.zip
    - --agent：DIAYN or DIAYN_VCL
    - --num_skills: scale of skill space
    - --choosed_skill: manually choose skill that perform well