import torch
import time
import numpy as np
import os
from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from PIL import Image
from train import Trainer_seg

class Inferencer:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   mode='validation')

        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device))
        self.model.eval()

        self.metric = self._init_metric(self.args.task, self.args.n_classes)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []

    def start_inference_segmentation(self):
        f1_list, acc_list, sen_list, mcc_list, miou_list = [], [], [], [], []

        for batch_idx, (img, target) in enumerate(self.loader_form.Loader):
            with torch.no_grad():
                x_in, img_id = img
                target, origin_size = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)

                output = self.model(x_in)

                if isinstance(output, (tuple, list)):  # Deep supervision desteği
                    output = output[0]

                metric_result = self.post_process(output, target, x_in, img_id)
                f1_list.append(metric_result["f1"])
                acc_list.append(metric_result["acc"])
                sen_list.append(metric_result["sen"])
                mcc_list.append(metric_result["mcc"])
                miou_list.append(metric_result["iou"])

        print(f"mean mIoU: {np.mean(miou_list):.4f}")
        print(f"mean F1 score: {np.mean(f1_list):.4f}")
        print(f"mean Accuracy: {np.mean(acc_list):.4f}")
        print(f"mean Sensitivity: {np.mean(sen_list):.4f}")
        print(f"mean MCC: {np.mean(mcc_list):.4f}")

    def post_process(self, output, target, x_img, img_id):
        x_img = x_img.squeeze(0).cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = (x_img * self.image_std + self.image_mean) * 255.0
        x_img = x_img.astype(np.uint8)

        output = utils.remove_center_padding(output)
        target = utils.remove_center_padding(target)

        output_argmax = (output > 0.5).cpu().numpy().astype(np.uint8) * 255

        path, fn = os.path.split(img_id[0])
        img_id, _ = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        save_dir = os.path.join(dir_path, "InferenceSave")

        os.makedirs(save_dir, exist_ok=True)

        Image.fromarray(x_img).save(os.path.join(save_dir, f"{img_id}.png"), quality=100)
        Image.fromarray(output_argmax.squeeze()).save(
            os.path.join(save_dir, f"{img_id}_argmax.png"), quality=100
        )

        metric_result = metrics.metrics_np(
            output_argmax[None, :], target.squeeze(0).cpu().numpy(), b_auc=False
        )
        print(f"{img_id} \t Done !!")

        return metric_result

    def __init_model(self, model_name):
        model_dict = {
            "UNet": model_implements.UNet,
            "UNet2P": model_implements.UNet2P,
            "UNet3P_Deep": model_implements.UNet3P_Deep,
            "ResUNet": model_implements.ResUNet,
            "ResUNet2P": model_implements.ResUNet2P,
            "SAUNet": model_implements.SAUNet,
            "ATTUNet": model_implements.ATTUNet,
            "DCSAU_UNet": model_implements.DCSAU_UNet,
            "AGNet": model_implements.AGNet,
            "R2UNet": model_implements.R2UNet,
            "ConvUNeXt": model_implements.ConvUNeXt,
            "FRUNet": model_implements.FRUNet,
            "FSGNet": model_implements.FSGNet,
        }

        if model_name not in model_dict:
            raise ValueError(f"Geçersiz model adı: {model_name}")

        model = model_dict[model_name](n_classes=1, in_channels=self.args.input_channel)
        return torch.nn.DataParallel(model)

    def __init_data_loader(self, x_path, y_path, batch_size, mode):
        if self.args.dataloader != "Image2Image_zero_pad":
            raise ValueError(f"Geçersiz dataloader adı: {self.args.dataloader}")

        return dataloader_hub.Image2ImageDataLoader_zero_pad(
            x_path=x_path,
            y_path=y_path,
            batch_size=batch_size,
            num_workers=self.args.worker,
            pin_memory=self.args.pin_memory,
            mode=mode,
            args=self.args,
        )

    def _init_metric(self, task_name, num_class):
        if task_name != "segmentation":
            raise ValueError(f"Geçersiz görev adı: {task_name}")

        return metrics.StreamSegMetrics_segmentation(num_class)
