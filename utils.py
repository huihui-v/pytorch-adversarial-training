import torch


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
    
    def report(self):
        return (self.sum / self.count)

def collect(x, device, mode='mean'):
    xt = torch.tensor([x]).to(device)
    torch.distributed.all_reduce(xt, op=torch.distributed.ReduceOp.SUM)
    # print(xt.item())
    xt = xt.item()
    if mode == 'mean':
        xt /= torch.distributed.get_world_size()
    return xt
    
def get_device_id():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args.local_rank