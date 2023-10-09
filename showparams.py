import torch
from torchkeras import summary
from agent.maddpg.maddpg import Actor
checkpoint = torch.load('model/maddpg/model_0001000.pkl')
agent = Actor([5, 100, 100])
agent.load_state_dict(checkpoint[0])
# print(agent.parameters())

opt_para = ['module.classifier.weight', 'module.classifier.bias']
# for name, param in agent.named_parameters():
#     if name not in opt_para:
#         param.requires_grad = False
#     if name in opt_para:
#         param.requires_grad = True
cnt = 0
for name, param in agent.named_parameters():
    if param.requires_grad:
        print("-----{}:{}".format(name, param))
    if cnt > 3:
        break
    cnt += 1
