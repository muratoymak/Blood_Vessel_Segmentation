import torch
import time
import numpy as np
import os
from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from PIL import Image
from torchvision import transforms
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

                # Check if patch-based prediction should be used
                if hasattr(self.args, 'use_patches') and self.args.use_patches and hasattr(self.args,
                                                                                           'patch_predict') and self.args.patch_predict:
                    # Load original image
                    original_image = Image.open(img_id[0]).convert('RGB')
                    original_target = Image.open(origin_size[0]).convert('L')

                    # Use patch-based prediction
                    output = self._predict_with_patches(original_image)

                    # Convert target to numpy array
                    target_np = np.array(original_target)
                    target_np = np.where(target_np < 128, 0, 1)  # Binary conversion

                    # Process the output for metrics calculation and saving
                    output_tensor = torch.from_numpy(output).unsqueeze(0).unsqueeze(0).float()
                    target_tensor = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).long().to(self.device)

                    metric_result = self.post_process(output_tensor, target_tensor, x_in, img_id)
                else:
                    # Regular prediction
                    x_in = x_in.to(self.device)
                    target = target.long().to(self.device)

                    output = self.model(x_in)

                    if isinstance(output, (tuple, list)):  # Deep supervision support
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

    def _predict_with_patches(self, image):
        """
        Make predictions by dividing the image into patches and then combine the results
        """
        # Get image dimensions
        w, h = image.size
        patch_size = self.args.patch_size
        stride = self.args.patch_stride if hasattr(self.args, 'patch_stride') else patch_size // 2

        # Initialize result image and counter
        result = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        # Normalization transform
        normalize = transforms.Normalize(mean=self.loader_form.image_loader.image_mean,
                                         std=self.loader_form.image_loader.image_std)

        # Process patches
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Crop patch
                patch = image.crop((j, i, j + patch_size, i + patch_size))

                # Convert to tensor and normalize
                patch_tensor = transforms.ToTensor()(patch)

                if self.args.input_space == 'RGB':
                    patch_tensor = normalize(patch_tensor)
                elif self.args.input_space == 'GR':
                    patch_tensor_r = patch_tensor[0].unsqueeze(0)
                    patch_tensor_grey = transforms.ToTensor()(transforms.Grayscale()(patch))
                    patch_tensor = torch.cat((patch_tensor_r, patch_tensor_grey), dim=0)

                patch_tensor = patch_tensor.unsqueeze(0).to(self.device)

                # Model prediction
                with torch.no_grad():
                    output = self.model(patch_tensor)

                if isinstance(output, tuple) or isinstance(output, list):
                    output = output[0]

                # Apply sigmoid and convert to numpy
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                # Add to result image
                result[i:i + patch_size, j:j + patch_size] += prediction
                count[i:i + patch_size, j:j + patch_size] += 1

        # Calculate average (avoid division by zero)
        mask = count > 0
        result[mask] = result[mask] / count[mask]

        return result

    def post_process(self, output, target, x_img, img_id):
        # Handle the case where output is a numpy array from patch prediction
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0).to(self.device)

        # Handle x_img for visualization
        if isinstance(x_img, torch.Tensor):
            x_img = x_img.squeeze(0).cpu().numpy()
            x_img = np.transpose(x_img, (1, 2, 0))
            x_img = (x_img * self.image_std + self.image_mean) * 255.0
            x_img = x_img.astype(np.uint8)
        else:
            # If x_img is not a tensor (in case patch prediction was used)
            x_img = np.array(Image.open(img_id[0]).convert('RGB'))

        # Remove padding if necessary
        if hasattr(self.args, 'use_patches') and self.args.use_patches and hasattr(self.args,
                                                                                   'patch_predict') and self.args.patch_predict:
            # For patch-based prediction, we don't need to remove padding
            output_processed = output
            target_processed = target
        else:
            output_processed = utils.remove_center_padding(output)
            target_processed = utils.remove_center_padding(target)

        # Convert output to binary mask
        output_argmax = (output_processed > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Prepare paths for saving
        path, fn = os.path.split(img_id[0])
        img_id_str, _ = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        save_dir = os.path.join(dir_path, "InferenceSave")

        os.makedirs(save_dir, exist_ok=True)

        # Save images
        Image.fromarray(x_img).save(os.path.join(save_dir, f"{img_id_str}.png"), quality=100)
        Image.fromarray(output_argmax.squeeze()).save(
            os.path.join(save_dir, f"{img_id_str}_argmax.png"), quality=100
        )

        # Calculate metrics
        metric_result = metrics.metrics_np(
            output_argmax, target_processed.squeeze(0).cpu().numpy(), b_auc=False
        )
        print(f"{img_id_str} \t Done !!")

        return metric_result

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

    def __init_model(self, model_name):
        """
        This method is kept for backward compatibility but not used
        since we're using Trainer_seg.init_model instead
        """
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