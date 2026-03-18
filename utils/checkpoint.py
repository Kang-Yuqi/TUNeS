import os
import torch
import csv


def save_checkpoint(model, optimizer, scheduler, epoch, eval_loss, best_eval_loss,
                    checkpoint_dir, is_best=False, early_counter=0):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_eval_loss": best_eval_loss,
        "early_counter": early_counter,
    }

    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))

    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_checkpoint.pt"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))



def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
    return model, optimizer, scheduler, start_epoch, best_eval_loss


# def log_loss_csv(log_path, epoch, train_loss, eval_loss=None, stats=None):
#     import csv
#     fieldnames = ['epoch', 'train_loss', 'eval_loss']

#     if stats is not None:
#         for k in stats.keys():
#             if k not in fieldnames:
#                 fieldnames.append(k)

#     file_exists = os.path.exists(log_path)
#     with open(log_path, 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()

#         row = {'epoch': epoch, 'train_loss': train_loss}
#         if eval_loss is not None:
#             row['eval_loss'] = eval_loss
#         if stats is not None:
#             row.update(stats)
#         writer.writerow(row)


def log_loss_csv(log_path, epoch, train_loss, eval_loss=None, stats=None):
    import csv
    mode = 'w' if epoch == 0 else 'a'
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "eval_loss"] + (list(stats.keys()) if stats else [])
        )
        # 如果是新文件 或者重新开始训练，需要写 header
        if mode == 'w' or not file_exists:
            writer.writeheader()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss if eval_loss is not None else "",
        }
        if stats:
            row.update({k: float(v) for k, v in stats.items()})
        writer.writerow(row)

