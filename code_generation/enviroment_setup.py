import os
import logging
from logging import handlers
import time
import torch
from pytz import timezone, utc
from datetime import datetime
import random
import numpy
from subprocess import call


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_filename, level='info', when='D', backCount=3, time_zone="localtime"):
        self.logger = logging.getLogger(log_filename)
        log_format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - '
                                           '%(levelname)s - %(module)s : %(message)s')  # Set format
        console_format_str = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')  # Set format

        if timezone is not "localtime":
            # https://stackoverflow.com/questions/32402502/
            def customTime(*args):
                utc_dt = utc.localize(datetime.utcnow())
                my_tz = timezone(time_zone)
                converted = utc_dt.astimezone(my_tz)
                return converted.timetuple()

            log_format_str.converter = customTime
            console_format_str.converter = customTime

        self.logger.setLevel(self.level_relations.get(level))  # Set log level
        sh = logging.StreamHandler()  # Print to console
        sh.setFormatter(console_format_str)
        # Write to file
        th = handlers.TimedRotatingFileHandler(filename=log_filename, when=when, backupCount=backCount, encoding='utf-8')
        # Create a processor that automatically generates files at specified intervals
        # 'backupCount' is the number of backup files. If it exceeds this number, it will be deleted automatically.
        # 'when' is the time unit of the interval
        th.setFormatter(log_format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def set_logger(args, save_path_name="outputs", log_level="debug", time_zone="Asia/Shanghai"):
    '''
    Set logging file

    :param args: parameters
    :param save_path_name: set the save path
    :param log_level: {debug, info, warning, error, crit}
    :return: {args.timestamp, args.model_save_path, args.log_file, args.logger}
    '''
    if args.if_load_from_checkpoint:
        timestamp = args.checkpoint_name
    else:
        timestamp = str(int(time.time()))
    args.timestamp = timestamp
    args.model_save_path = '{}/{}/'.format(save_path_name, args.timestamp)

    if not os.path.exists(args.model_save_path):
        # print("{} is not exists and is now created".format(args.model_save_path))
        os.makedirs(args.model_save_path)  # Create the output path
    args.log_file = args.model_save_path + time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
    args.log = Logger(args.log_file, level=log_level, time_zone=time_zone)
    args.logger = args.log.logger
    args.logger.info("Model save path:{}, log:{}".format(args.model_save_path, args.log_file.split('/')[-1]))


def set_gpu(args):
    if torch.cuda.is_available():
        # call(["CUDA_LAUNCH_BLOCKING=1"])
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.n_gpu = torch.cuda.device_count()
        # print('Available devices ', torch.cuda.device_count())
        # print('Current cuda device ', torch.cuda.current_device())
        if args.n_gpu > 1:
            torch.distributed.init_process_group(backend="nccl")
            # args.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        else:
            args.device = torch.device("cuda")
        args.logger.info("You are now using {} GPU - {} local_rank - {}".format(args.n_gpu, torch.cuda.current_device()
                                                                                ,args.local_rank))
    else:
        args.logger.warning("CUDA is not avaliable, so now in CPU mode!")
        args.device = torch.device("cpu")



def set_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def show_hparams(args, print_params=None):
    options = vars(args)
    print_str = "Print params info: \n"
    if print_params is None:
        print_params = options.keys()
    for item in print_params:
        if item in options:
            print_str += "{}:{}\n".format(item, options[item])
    print_str += '-' * 15
    args.logger.info(print_str)

    # print(options)




