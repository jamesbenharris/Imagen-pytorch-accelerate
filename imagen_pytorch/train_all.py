from PIL import Image
import torch as th
from imagen_pytorch.download import load_checkpoint
from imagen_pytorch.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
import argparse
from imagen_pytorch.resample import create_named_schedule_sampler
import os

from imagen_pytorch import logger
from imagen_pytorch.dataset import get_loader
from imagen_pytorch.train_utils import TrainLoop
from imagen_pytorch.get_webdataset_loader import WebdatasetReader

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_folder', type=str, default='', help='Input folder')
  parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
  parser.add_argument('--path_for_chaeckpoints', type=str, default='', help='path_for_chaeckpoints')
  parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
  parser.add_argument('--save_interval', type=int, default=200, help='batch_size')
  args = parser.parse_args()
  print('num cuda', th.cuda.device_count())
  
  options = model_and_diffusion_defaults()
  options['use_fp16'] = False
  options['t5_name'] = 't5-3b'
  model, diffusion = create_model_and_diffusion(**options)
  model.load_state_dict(load_checkpoint('base', 'cpu'), strict=False)
  reader = WebdatasetReader(
        None,
        args.input_folder,
        args.batch_size,
        2,
        enable_text=True,
        enable_image=True,
        enable_metadata=True,
    )
  data = reader.get_loader()
  logger.configure()

  logger.log("creating model and diffusion...")
  schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
  TrainLoop(
        model=model,
        diffusion=diffusion,
        data=reader,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=args.save_interval,
        resume_checkpoint=False,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.01,
        lr_anneal_steps=0,
        save_dir='/home/cene655/checkpoints',
  ).run_loop()
if __name__ == '__main__':
    main()
