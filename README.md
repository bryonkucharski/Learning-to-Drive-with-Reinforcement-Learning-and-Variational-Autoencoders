# Learning to Drive with Reinforcement Learning and Variational Autoencoders
Using the DuckieTown self driving car environment, the agent learns policies to drive in a straight line using the DDPG reinforcement learning algorithm. I experimented with two different state representations: one with raw images directly from the simulator, and the second with a VAE compressed version of the image to a single dimensional vector. The idea was that if the agent learns from a lower dimensional state representation, it may learn better policies or learn policies quicker than using raw images. My results showed that this was not true, and using raw images outperformed the latent representations.
<p align="center">
<img src="images/rl_vae.gif" width="40%">
</p>

Idea came from the paper "Learning to Drive In A Day" by Wayve
<br/>
Blog Post: https://wayve.ai/blog/learning-to-drive-in-a-day-with-reinforcement-learning <br/>
Paper: https://arxiv.org/pdf/1807.00412.pdf 
