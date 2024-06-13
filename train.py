import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter

import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from einops import rearrange
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from safetensors.torch import load_file as load_safetensors

import init_proj_path
from vwm.util import instantiate_from_config, isheatmap

MULTINODE_HACKS = True


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir"
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir"
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. "
             "Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`",
        default=list()
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train"
    )
    parser.add_argument(
        "--no_test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test"
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything"
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name"
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data"
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate"
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path"
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12"
    )
    parser.add_argument(
        "--no_base_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="no config name"
    )
    if version.parse(pl.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from"
        )
    parser.add_argument(
        "--n_devices",
        type=int,
        default=8,
        help="number of gpus in training"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default="ckpts/pytorch_model.bin",
        help="path to checkpoint to finetune from"
    )
    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print("Available last checkpoints:", ckpt)
    if len(ckpt) > 1:
        print("Got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            # version confusion but not bad
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


def save_img_seq_to_video(out_path, img_seq, fps):
    # img_seq: np array
    writer = imageio.get_writer(out_path, fps=fps)
    for img in img_seq:
        writer.append_data(img)
    writer.close()


class SetupCallback(Callback):
    def __init__(
            self,
            resume,
            now,
            logdir,
            ckptdir,
            cfgdir,
            config,
            lightning_config,
            debug,
            ckpt_name=None
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if not self.debug and trainer.global_rank == 0:
            # print("Summoning checkpoint")
            # if self.ckpt_name is None:
            #     ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            # else:
            #     ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            # trainer.save_checkpoint(ckpt_path)
            print("Exiting")

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now))
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now))
            )
        else:
            # ModelCheckpoint callback created log directory, remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
            self,
            batch_frequency,
            clamp=True,
            increase_log_steps=True,
            rescale=True,
            disabled=False,
            log_on_batch_idx=False,
            log_first_step=False,
            log_images_kwargs=None,
            log_before_first_step=False,
            enable_autocast=True,
            num_frames=25
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else dict()
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step
        self.num_frames = num_frames

    @rank_zero_only
    def log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx
    ):
        root = os.path.join(save_dir, "images", split)
        for log_type in images:
            if isheatmap(images[log_type]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[log_type].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_epoch{:03}_batch{:06}_step{:06}.png".format(
                    log_type, current_epoch, batch_idx, global_step
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, log_type, filename)
                plt.savefig(path)
                plt.close()
            elif "mp4" in log_type:
                dir_path = os.path.join(root, log_type)
                os.makedirs(dir_path, exist_ok=True)
                img_seq = images[log_type]
                if self.rescale:
                    img_seq = (img_seq + 1.0) / 2.0
                img_seq = rearrange(img_seq, "(b t) c h w -> b t h w c", t=self.num_frames)
                B, T = img_seq.shape[:2]
                for b_i in range(B):
                    cur_img_seq = img_seq[b_i].numpy()  # [t h w c]
                    cur_img_seq = (cur_img_seq * 255).astype(np.uint8)  # [t h w c]
                    filename = "{}_epoch{:02}_batch{:04}_step{:06}.mp4".format(
                        log_type, current_epoch, batch_idx, global_step
                    )
                    save_img_seq_to_video(os.path.join(root, log_type, filename), cur_img_seq, fps=10)
            else:
                grid = torchvision.utils.make_grid(images[log_type], nrow=int(images[log_type].shape[0] ** 0.5))
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_epoch{:02}_batch{:04}_step{:06}.png".format(
                    log_type, current_epoch, batch_idx, global_step
                )
                dir_path = os.path.join(root, log_type)
                os.makedirs(dir_path, exist_ok=True)
                path = os.path.join(dir_path, filename)
                img = Image.fromarray(grid)
                img.save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
                self.check_frequency(check_idx)
                and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
                and callable(pl_module.log_images)
        ) or split == "test":
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled()
            }

            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for log_type in images:
                if isinstance(images[log_type], torch.Tensor):
                    images[log_type] = images[log_type].detach().float().cpu()
                    if self.clamp and not isheatmap(images[log_type]):
                        images[log_type] = torch.clamp(images[log_type], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq == 0 or check_idx in self.log_steps) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        else:
            return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")

    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="test")


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode, postfix, resume
    # `--key value` arguments are interpreted as arguments to the trainer
    # `nested.key=value` arguments are interpreted as config parameters
    # configs are merged from left-to-right followed by command line parameters

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: train.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmd line)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python train.py`
    # (in particular `train.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f"Resuming from checkpoint `{ckpt}`")
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(
                        opt.base[0]
                    )[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                               os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                               + 1:
                               ]  # cut away the first one (we assert all configs are in "configs")
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        if opt.no_date:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        else:
            nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    # move before model init, in case a torch.compile(...) is called somewhere
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to gpu
        trainer_config["accelerator"] = "gpu"

        standard_args = default_trainer_args()
        for k in standard_args:
            if getattr(opt, k) != standard_args[k]:
                trainer_config[k] = getattr(opt, k)

        n_devices = getattr(opt, "n_devices", None)
        if n_devices is not None:
            assert isinstance(n_devices, int) and n_devices > 0
            devices = [str(i) for i in range(n_devices)]
            trainer_config["devices"] = ",".join(devices) + ","
        else:
            assert "devices" in trainer_config, "Must specify either n_devices or devices"

        ckpt_resume_path = opt.resume_from_checkpoint

        if "devices" not in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # use pretrained model
        if not opt.resume or opt.finetune:
            if not opt.finetune or not os.path.exists(opt.finetune):
                default_ckpt = "ckpts/svd_xt.safetensors"
                print(f"Loading pretrained model from {default_ckpt}")
                svd = load_safetensors(default_ckpt)
                for k in list(svd.keys()):
                    if "time_embed" in k:  # duplicate a new timestep embedding from the pretrained weights
                        svd[k.replace("time_embed", "cond_time_stack_embed")] = svd[k]
            else:
                ckpt_path = opt.finetune
                print(f"Loading pretrained model from {ckpt_path}")
                if ckpt_path.endswith("ckpt"):
                    svd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                elif ckpt_path.endswith("bin"):  # for deepspeed merged checkpoints
                    svd = torch.load(ckpt_path, map_location="cpu")
                    for k in list(svd.keys()):  # remove the prefix
                        if "_forward_module" in k:
                            svd[k.replace("_forward_module.", "")] = svd[k]
                        del svd[k]
                elif ckpt_path.endswith("safetensors"):
                    svd = load_safetensors(ckpt_path)
                else:
                    raise NotImplementedError
            missing, unexpected = model.load_state_dict(svd, strict=False)

            # avoid empty weights when resuming from EMA weights
            for miss_k in missing:
                ema_name = miss_k.replace(".", "").replace("modeldiffusion_model", "model_ema.diffusion_model")
                if ema_name in svd:
                    svd[miss_k] = svd[ema_name]
                    print("Fill", miss_k, "with", ema_name)
            missing, unexpected = model.load_state_dict(svd, strict=False)

            if len(missing) > 0:
                if not opt.finetune or not os.path.exists(opt.finetune):
                    model.reinit_ema()
                missing = [model_key for model_key in missing if "model_ema" not in model_key]
                # print(f"Missing keys: {missing}")
            print(f"Missing keys: {missing}")
            # if len(unexpected) > 0:
            #     print(f"Unexpected keys: {unexpected}")
            print(f"Unexpected keys: {unexpected}")

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["csv"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # use TrainResult/EvalResult(checkpoint_on=metric) to specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:02}",
                "verbose": True,
                "save_last": True,
                "save_top_k": -1
            }
        }
        # if hasattr(model, "monitor"):
        #     print(f"Monitoring {model.monitor} as checkpoint metric")
        #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
        # default to ddp if not further specified
        default_strategy_config = {"target": "pytorch_lightning.strategies.DDPStrategy"}

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {
                "find_unused_parameters": True
            }
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print(
            f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ "
        )
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "train.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name
                }
            },
            "image_logger": {
                "target": "train.ImageLogger",
                "params": {
                    "batch_frequency": 1000,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step"
                }
            }
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        # if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        #     print(
        #         "WARNING: saving checkpoints every n train steps without deleting, this might require some free space"
        #     )
        #     default_metrics_over_trainsteps_ckpt_dict = {
        #         "metrics_over_trainsteps_checkpoint": {
        #             "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        #             "params": {
        #                 "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
        #                 "filename": "{epoch:06}-{step:09}",
        #                 "verbose": True,
        #                 "save_top_k": -1,
        #                 "every_n_train_steps": 10000,
        #                 "save_weights_only": True
        #             }
        #         }
        #     }
        #     default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over
        # config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }
        trainer = Trainer(**trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir

        # data
        data = instantiate_from_config(config.data)
        # NOTE: according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary, but it is
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        # data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print("Datasets not yet initialized")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate
            )
        if cpu:
            ngpu = 1
        else:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to "
                "{:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batch_size) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                # print("Summoning checkpoint")
                # if melk_ckpt_name is None:
                #     ckpt_path = os.path.join(ckptdir, "last.ckpt")
                # else:
                #     ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
                # trainer.save_checkpoint(ckpt_path)
                print("Exiting")


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            trainer.fit(model, data, ckpt_path=ckpt_resume_path)
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as error:
        # if MULTINODE_HACKS:
        #     import datetime
        #     import os
        #     import socket
        #
        #     import requests
        #
        #     device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        #     hostname = socket.gethostname()
        #     ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        #     resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
        #     print(
        #         f"ERROR at {ts} "
        #         f"on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
        #         flush=True
        #     )
        raise error
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        # if trainer.global_rank == 0:
        #    print(trainer.profiler.summary())
