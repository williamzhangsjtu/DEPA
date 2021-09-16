from sklearn.utils import axis0_safe_slice
import torch
import h5py
import fire
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import datetime
from pprint import pformat
import kaldi_io
from sklearn.preprocessing import StandardScaler

import utils
import models
import losses
from dataloader import create_dataloader

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint,\
    EarlyStopping, global_step_from_engine
from ignite.metrics import RunningAverage, Loss

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(device)

class Runner(object):

    def __init__(self, seed=0):
        super(Runner, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
        self.seed = seed
    
    @staticmethod
    def _forward(model, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        return model(inputs), targets

    def train(self, config, debug=False):
        config = utils.parse_config(config)
        outputdir = os.path.join(config['outputdir'], 
                "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m')))
        os.makedirs(outputdir, exist_ok=True)
        logger = utils.genlogger(os.path.join(outputdir, 'logging.txt'))
        logger.info(f'Output directory is: {outputdir}')
        torch.save(config, os.path.join(outputdir, 'run_config.d'))

        train, dev, test = utils.dataset_split(
            config['input'], debug=debug, random_state=self.seed)

        logger.info('Conducting Normalization')
        scaler, _ = utils.normalization(config['input'], train,
            config['normalization'], **config['normalization_args'])

        transform_fn = (lambda x: x) if config['transform'] else\
            utils.get_transform(**config['transform_args'])

        train_dataloader = create_dataloader(
            config['input'], train, scaler=scaler,
            transform_fn=transform_fn,
            dataloader_args=config['dataloader_args'],
            sample_args=config['sample_args'])
        dev_dataloader = create_dataloader(
            config['input'], dev, scaler=scaler,
            dataloader_args=config['dataloader_args'],
            sample_args=config['sample_args'])
        test_dataloader = create_dataloader(
            config['input'], test, scaler=scaler,
            dataloader_args=config['dataloader_args'],
            sample_args=config['sample_args'])

        model = getattr(models, config['model'])(
            **config['model_args'])
        model = model.to(device)
        for line in pformat(model).split('\n'):
            logger.info(line)
        criterion = getattr(losses, config['criterion'])(
            **config['criterion_args'])
        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])
        scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(
            optimizer, **config['scheduler_args'])

        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                output, targets = Runner._forward(model, batch)
                optimizer.zero_grad()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                output, targets = Runner._forward(model, batch)
            return output, targets

        trainer, evaluator, testor = Engine(_train), Engine(_inference), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Loss(criterion).attach(evaluator, 'Loss')
        Loss(criterion).attach(testor, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(trainer, output_transform=lambda x: {'loss': x})

        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            logger.info(f'<==== Epoch {trainer.state.epoch} ====>')
            evaluator.run(dev_dataloader)
            train_loss = engine.state.metrics['Loss']
            val_loss = evaluator.state.metrics['Loss']
            logger.info(f'Training Loss: {train_loss}')
            logger.info(f'Validation Loss: {val_loss}')
            scheduler.step(val_loss)
        
        @trainer.on(Events.COMPLETED)
        def test(engine):
            params = torch.load(
                glob(os.path.join(outputdir, 'eval_best*.pt'))[0], map_location=DEVICE)
            model.load_state_dict(params)
            testor.run(test_dataloader)
            test_loss = testor.state.metrics['Loss']
            logger.info(f'Test Loss: {test_loss}')

        
        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        
        # ModelCheckpoint only supports to store object that has state_dict
        # file name is: {prefix}_{name}_{score_name}={score}.pt
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, {'model': model})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)

        trainer.run(train_dataloader, max_epochs=config['n_epochs'])

        return outputdir

    @staticmethod
    def encoding(model_path, input_file, output_file):
        params = torch.load(
            glob(os.path.join(model_path, 'eval_best*.pt'))[0], map_location='cpu')
        config = torch.load(
            glob(os.path.join(model_path, 'run_config.d'))[0])
        model = getattr(models, config['model'])(**config['model_args'])
        model.load_state_dict(params)
        model.eval().to(DEVICE)
        
        scaler = None
        if config['normalization']:
            scaler = StandardScaler(**config['normalization_args'])
            with h5py.File(input_file, 'r') as input:
                for key in tqdm(input.keys(), desc="Calculating mean, std: "):
                    for i in range(len(input[key])):
                        scaler.partial_fit(input[key][str(i)][()])

        with h5py.File(input_file, 'r') as input,\
                open(output_file, 'wb') as output,\
                torch.no_grad():
            for key in tqdm(input.keys(), desc="Extracting Progress: "):
                feats = []
                for i in range(len(input[key])):
                    feat = input[key][str(i)][()]
                    if scaler is not None:
                        feat = scaler.transform(feat)
                    feat = torch.from_numpy(
                        feat).unsqueeze(0).to(DEVICE)
                    out = model.extract_embedding(feat)
                    feats.append(out.squeeze(0).cpu())
                # output[key] = np.concatenate(feat, axis=0)
                kaldi_io.write_mat(
                    output, np.concatenate(feats, axis=0), key=key)
            



if __name__ == '__main__':
    fire.Fire(Runner)
