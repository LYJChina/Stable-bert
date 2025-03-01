import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for sentiment analysis model")
    
    # Dataset and model parameters
    parser.add_argument('--pretrained_model_name', type=str,    default="bert-base-uncased", help="Pre-trained BERT model name")
    parser.add_argument('--max_len', type=int,                  default=80, help="Maximum sequence length")
    parser.add_argument('--batch_size', type=int,               default=32, help="Batch size for training and evaluation")
    parser.add_argument('--dropout_prob', type=float,           default=0.1, help="Dropout probability")
    parser.add_argument('--num_classes', type=int,              default=3, help="Number of classes for classification")
    parser.add_argument('--device', type=str,                   default="cuda:2", help="Device to train the model (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument('--epochs', type=int,                   default=10, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float,          default=5e-5, help="Learning rate")
    parser.add_argument('--adamw_correct_bias', type=bool,      default=True, help="Whether to apply bias correction in AdamW optimizer")
    parser.add_argument('--num_warmup_steps', type=int,         default=1, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--num_runs', type=int,                 default=10, help="Number of different runs for the experiment")
    parser.add_argument('--random_seeds', type=list,            default=[8,9,5,8,42,8,9,5,8,42], help="List of random seeds for different runs")
    # parser.add_argument('--random_seeds', type=list,            default= list(range(10)), help="List of random seeds for different runs")
    # 2015
    parser.add_argument('--image_dir', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015_images", help="Directory for image data")
    parser.add_argument('--train_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/train.tsv", help="Path to the train dataset")
    parser.add_argument('--dev_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/dev.tsv", help="Path to the dev dataset")
    parser.add_argument('--test_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/test.tsv", help="Path to the test dataset")
    # parser.add_argument('--captions_json', type=str, default="/home/yinxx23/yxx_noise/captions/twitter2015_images.json", help="Path to the image captions JSON file")
    
    # 2017
    # parser.add_argument('--image_dir', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2017_images", help="Directory for image data")
    # parser.add_argument('--train_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2017/train.tsv", help="Path to the train dataset")
    # parser.add_argument('--dev_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2017/dev.tsv", help="Path to the dev dataset")
    # parser.add_argument('--test_tsv', type=str, default="/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2017/test.tsv", help="Path to the test dataset")
    # parser.add_argument('--captions_json', type=str, default="/home/yinxx23/yxx_noise/captions/twitter2017_images.json", help="Path to the image captions JSON file")
    # # stable
    parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')
    # for pow
    parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')
    parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')
    parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')
    parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
    parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
    parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')
    # for first step
    parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')
    parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochp', type=list, default=2, help="Begin epoch of stableLearning")
    parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
    parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')
    parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
    # Dataset file paths

    return parser.parse_args()

