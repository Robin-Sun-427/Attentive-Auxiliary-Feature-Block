from math import log10
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import torch
from datasets import datasets_train
from data_utilis import get_trainning_set,get_eval_set
from AAF_Network import AAF


parser = argparse.ArgumentParser(description='AAF pytorch implemet')
parser.add_argument('--upscale_factor',type=int,required=True,help='upscale_factor')
parser.add_argument('--batch_size',type=int,default=1,required=True,help='batch size')
parser.add_argument('--eval_batch_size',type = int,default=2,required=True,help='test batch_size')
parser.add_argument('--epochs',type=int,default=6,required=True,help='epochs ')
parser.add_argument('--lr',type=float,default=0.0001,required=True,help='learning rate you set')
parser.add_argument('--threads',type = int,default=6,help='how may threads you choose')
parser.add_argument('--seed',type = int,default=123,help='random seed to use,Default=123')
parser.add_argument('--step',type = int,default=10,help='step learning rate decay')
parser.add_argument('--base_filter',type=int,default=64,help='set base_filters numbers')
parser.add_argument('--cuda',action='store_true',help='use cuda?')
parser.add_argument('--patch_size',type=int,default=40,help='set patch_size')

opt = parser.parse_args()

def main():
    use_cuda = opt.cuda
    if use_cuda and not torch.cuda.is_available():
        raise Exception("No gpu to use .please run without --cuda")

    torch.manual_seed(opt.seed)

    train_sets = get_trainning_set(opt.patch_size, opt.upscale_factor)
    eval_sets  = get_eval_set(opt.patch_size, opt.upscale_factor)

    training_set_Loader = DataLoader(dataset=train_sets,num_workers=opt.threads,batch_size=opt.batch_size,
                                      shuffle=True)
    eval_set_Loader = DataLoader(dataset=eval_sets,num_workers=opt.threads,batch_size=opt.eval_batch_size,
                                      shuffle=False)

    AAF_networks = AAF(num_channels=1,base_filter=opt.base_filter,num_stages=4,scale_factor=opt.upscale_factor)
    criterion = nn.MSELoss()

    if use_cuda:
        AAF_networks.cuda()
        criterion = criterion.cuda()

    def adjust_learning_rate(epoch):
        lr = opt.lr * (0.1 ** (epoch // opt.step))
        return lr

    optimizer = optim.Adam(AAF_networks.parameters(),lr=opt.lr)

    def train_process(epoch):
        lr = adjust_learning_rate(epoch-1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

        epoch_loss = 0
        for iter,batch in enumerate(training_set_Loader,1):
            input,target = Variable(batch[0]),Variable(batch[1])

            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            model_output = AAF_networks(input)

            # print("model_output size",model_output.shape)

            loss = criterion(model_output,target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iter, len(training_set_Loader), loss.item()))
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_set_Loader)))

    def eval_process():
        average_loss = 0
        for batch in eval_set_Loader:
            input,target = Variable(batch[0]),Variable(batch[1])
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            prediction = AAF_networks(input)
            mse = criterion(prediction,target)
            PSNR = 10 *log10(1.0/mse.item())
            average_loss += PSNR
        print("===> Avg. PSNR: {:.4f} dB".format(average_loss / len(eval_set_Loader)))

    def checkpoint(epoch):
        model_out_path ='./model/model__{}.pkl'.format(epoch)
        torch.save(AAF_networks.state_dict(),model_out_path)
        print("Checkpoint saved into {}".format(model_out_path))

    for epoch in range(opt.epochs+1):
        train_process(epoch)
        eval_process()
        if( epoch %100 ==0):
            checkpoint(epoch)

if __name__ =='__main__':
    main()
















