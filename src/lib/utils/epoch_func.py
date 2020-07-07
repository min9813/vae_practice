import os
import time
import numpy as np
import torch
import torchvision
import lib.utils.average_meter as average_meter
from tqdm import tqdm
try:
    from apex import amp
except ImportError:
    pass

def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))

def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    meter = average_meter.AverageMeter()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        meter.add_value("time_data", time.time() - since)
        input_x = data["data"]
        since = time.time()
        
        kl_loss, mse_loss = wrappered_model(input_x)
        meter.add_value("time_f", time.time()-since)

        kl_loss = kl_loss
        mse_loss = mse_loss

        since = time.time()
        loss = kl_loss + mse_loss
        if args.TRAIN.fp16:
            with amp.scale_loss(loss, [optimizer]) as scaled_loss:
                scaled_loss.backward()
        # loss.backward()
        else:
            loss.backward()
        optimizer.step()

        meter.add_value("time_b", time.time()-since)

        meter.add_value("loss_total", loss)
      
        meter.add_value("loss_kl", kl_loss)
        meter.add_value("loss_rec", mse_loss)

        if args.LOG.train_print and (batch_idx+1) % args.LOG.train_print_iter == 0:
            # current training accuracy
            time_cur = (time.time() - iter_since)
            meter.add_value("time_iter", time_cur)
            iter_since = time.time()

            msg = f"Epoch [{epoch}] [{batch_idx+1}]/[{iter_num}]\t"
            summary = meter.get_summary()
            for name, value in summary.items():
                msg += " {}:{:.6f} ".format(name, value)
            logger.info(msg)

        if args.debug:
            if batch_idx >= 10:
                break
        since = time.time()
    train_info = meter.get_summary()

    return train_info


def valid_epoch(wrappered_model, train_loader, epoch, args, logger=None):
    meter = average_meter.AverageMeter()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since
    
    with torch.no_grad():
        save_num = 64
        now_num = 0
        infer_data = []
        for batch_idx, data in tqdm(enumerate(train_loader), total=iter_num):

            meter.add_value("time_data", time.time() - since)
            input_x = data["data"]
            since = time.time()
            
            kl_loss, mse_loss = wrappered_model(input_x)
            meter.add_value("time_f", time.time()-since)

            kl_loss = kl_loss
            mse_loss = mse_loss

            since = time.time()
            loss = kl_loss + mse_loss

            meter.add_value("time_b", time.time()-since)

            meter.add_value("loss_total", loss)
        
            meter.add_value("loss_kl", kl_loss)
            meter.add_value("loss_rec", mse_loss)
            if now_num < save_num:
                infer_data.append(input_x)
            now_num += save_num

            if args.debug:
                if batch_idx >= 10:
                    break
            since = time.time()
        infer_data = torch.cat(infer_data, dim=0)[:save_num]
    train_info = meter.get_summary()
    inference(wrappered_model.model, infer_data, args, epoch)

    return train_info


def inference(model, data, args, epoch):
    # data = (B, C, W, H)
    B, C, W, H = data.size()
    with torch.no_grad():
        data = data.to(args.device)
        output, mean, val = model(data)
        output = output.cpu()
        nrow = int(np.floor(np.sqrt(B)))
        show_output = output[:nrow*nrow]
    make_directory(args.image_save_dir)
    save_path = os.path.join(args.image_save_dir, "epoch_{:0>5}.jpg".format(epoch))
    torchvision.utils.save_image(show_output, save_path, nrow, normalize=True, range=(-1., 1.))
