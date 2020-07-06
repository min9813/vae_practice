import time
import lib.utils.average_meter as average_meter
from apex import amp


def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    meter = average_meter.AverageMeter()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        meter.add_value("data_time", time.time() - since)
        input_x = data["data"]
        
        kl_loss, mse_loss = wrappered_model(input_x)

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

        meter.add_value("backward_time", time.time()-since)

        meter.add_value("total_loss", loss)
      
        meter.add_value("train_kl_loss", kl_loss)
        meter.add_value("train_rec_loss", mse_loss)

        if args.train_print and (batch_idx+1) % args.train_print_iter == 0:
            # current training accuracy
            time_cur = (time.time() - iter_since)
            meter.add_value("iter_time", time_cur)
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
