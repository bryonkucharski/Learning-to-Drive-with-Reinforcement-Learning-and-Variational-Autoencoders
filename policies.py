from models import *
import torch

class FCPolicy():
    def __init__(self,num_states, num_actions, actor_weight_scale,critic_weight_scale ,hidden_1 = None, hidden_2 = None,use_bn = True):

        self.actor =  ActorFC(
                                num_states = num_states,
                                num_actions = num_actions,
                                hidden_1 = hidden_1,
                                hidden_2 = hidden_2,
                                weight_scale = actor_weight_scale,
                                use_bn = use_bn
                            )

        self.actor_target = ActorFC(

                                num_states=num_states,
                                num_actions=num_actions,
                                hidden_1=hidden_1,
                                hidden_2=hidden_2,
                                weight_scale=actor_weight_scale,
                                use_bn=use_bn
                                )
        self.critic = CriticFC(

                            num_states=num_states,
                            num_actions=num_actions,
                            hidden_1=hidden_1,
                            hidden_2=hidden_2,
                            weight_scale=actor_weight_scale,
                            use_bn=use_bn
                            )

        self.critic_target = CriticFC(

                            num_states=num_states,
                            num_actions=num_actions,
                            hidden_1=hidden_1,
                            hidden_2=hidden_2,
                            weight_scale=actor_weight_scale,
                            use_bn=use_bn
                            )

    def save(self, PATH):
        torch.save({
            'actor_state_dict':  self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
        }, PATH)

    def load(self,PATH):
        checkpoint = torch.load(PATH)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])


class CNNPolicy( ):
    def __init__(self, num_actions, hidden_actor,hidden_critic_1,hidden_critic_2, filter_size,kernal_size,stride_size,padding, use_bn):


        self.actor = ActorCNN( num_actions,
                               hidden_actor,
                               filter_size,
                               kernal_size,
                               stride_size,
                               padding,
                               use_bn)


        self.actor_target= ActorCNN( num_actions,
                               hidden_actor,
                               filter_size,
                               kernal_size,
                               stride_size,
                               padding,
                               use_bn)


        self.critic =        CriticCNN(num_actions, hidden_critic_1,hidden_critic_2, filter_size,kernal_size,stride_size,padding, use_bn)
        self.critic_target = CriticCNN(num_actions, hidden_critic_1,hidden_critic_2, filter_size,kernal_size,stride_size,padding, use_bn)