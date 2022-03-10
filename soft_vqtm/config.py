import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of Code')
    parser.add_argument('--data', default='../data/',help='path to dataset')

    #path to Glove embeddings
    parser.add_argument('--glove', default='/research2/tools/glove',help='directory with Glove embeddings')
    parser.add_argument('--save', default='checkpoints/',help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='model_',help='Name to identify experiment')
    parser.add_argument('--expno', type=int, default=0,help='Name to identify experiment')

    # model arguments
    parser.add_argument('--in_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--numbr_concepts', type=int, default=90)

    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--decay', type=float, default=0.999)

    # training arguments
    parser.add_argument('--epochs', default=5000, type=int,help='number of total epochs to run')
    parser.add_argument('--print_every', type=int, default=100, help = 'Print after given # of iterations')
    parser.add_argument('--batchsize', default=128, type=int,help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.0002, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--optim', default='adam',help='optimizer (default: adam)')
    parser.add_argument('--shuffle', action='store_true')

    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_false')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
