import os, sys, time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler,
                 eval_fn, device, rank=0, world_size=1,
                 accum_steps=1, use_amp=True, amp_dtype="fp16",
                 normalize_pos: bool = True, normalize_disp: bool = True):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.accum_steps = max(1, int(accum_steps))
        self.use_amp = bool(use_amp)

        amp_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.amp_dtype = amp_map.get(amp_dtype, torch.float16)

        model = model.to(device)
        if world_size > 1:
            self.model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            self.is_ddp = True
        else:
            self.model = model
            self.is_ddp = False

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_fn = eval_fn

        self.scaler = torch.amp.GradScaler('cuda',enabled=self.use_amp and self.amp_dtype==torch.float16)

        self.normalize_pos = bool(normalize_pos)
        self.normalize_disp = bool(normalize_disp)

    def _prep_batch(self, batch):
        """
        Convert batch dict -> tensors on device, apply normalization:
          pos_norm = pos / L
          dpos_norm = dpos / L
          vel_std = (vel - mean) / std (optional)
        """
        pos = batch["pos_ini"].to(self.device, non_blocking=True)   # (M,3) or (B,M,3)
        vel = batch["vel_ini"].to(self.device, non_blocking=True)
        y   = batch["dpos"].to(self.device, non_blocking=True)

        z_ini = batch["z_ini"].to(self.device, non_blocking=True)
        z_fin = batch["z_fin"].to(self.device, non_blocking=True)

        # box_size may be tensor scalar, python float, or (B,)
        if "box_size" in batch:
            L = batch["box_size"].to(self.device, non_blocking=True).to(torch.float32)
        else:
            L = None

        # normalize position & displacement for Fourier features stability
        if self.normalize_pos:
            if L is None:
                raise RuntimeError("normalize_pos=True but batch has no 'box_size'")
            pos = pos / L
            pos = pos.clamp(0.0, 1.0 - 1e-6)  # keep inside [0,1)

        if self.normalize_disp:
            if L is None:
                raise RuntimeError("normalize_disp=True but batch has no 'box_size'")
            y = y / L

        return pos, vel, z_ini, z_fin, y
    
    def _forward_loss(self, batch):
        pos, vel, z_ini, z_fin, y = self._prep_batch(batch)

        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            pred = self.model(pos, vel, z_ini, z_fin)    # <-- changed
            loss = self.loss_fn(pred, y)
        return loss, pred

    def train_one_epoch(self, epoch, dataloader):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        steps = len(dataloader)

        pbar = tqdm(total=steps, desc=f"[Epoch {epoch}] Rank {self.rank}", file=sys.__stdout__) if self.rank==0 else None
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            # DDP + accum: disable sync on non-step boundary
            if self.is_ddp and ((step + 1) % self.accum_steps != 0):
                no_sync_ctx = self.model.no_sync()
            else:
                # dummy context
                no_sync_ctx = torch.enable_grad()

            with no_sync_ctx:
                loss, _ = self._forward_loss(batch)
                loss = loss / self.accum_steps
                self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.accum_steps
            if pbar is not None:
                pbar.set_postfix(loss=f"{(total_loss/(step+1)):.6f}")
                pbar.update(1)

        # tail step
        if (steps % self.accum_steps) != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, steps)
        if self.is_ddp:
            avg_loss_t = torch.tensor([avg_loss], device=self.device)
            torch.distributed.all_reduce(avg_loss_t, op=torch.distributed.ReduceOp.SUM)
            avg_loss = (avg_loss_t / self.world_size).item()

        if pbar is not None:
            pbar.close()
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, steps = 0.0, 0

        for batch in dataloader:
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                loss, _ = self._forward_loss(batch)
            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        if self.is_ddp:
            t = torch.tensor([avg_loss], device=self.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            avg_loss = (t / self.world_size).item()

        metrics = {"eval_loss": avg_loss}
        if self.eval_fn is not None:
            extra = self.eval_fn(self.model, dataloader, self.device)
            if isinstance(extra, dict):
                metrics.update(extra)
        return metrics["eval_loss"], metrics

    def run(self, train_loader, train_cfg, val_loader=None):
        from utils.checkpoint import save_checkpoint, log_loss_csv

        ckpt_dir = train_cfg["checkpoint_dir"]; os.makedirs(ckpt_dir, exist_ok=True)
        log_path = os.path.join(ckpt_dir, "loss_log.csv")

        start_epoch  = int(train_cfg.get("start_epoch", 0))
        final_epoch  = int(train_cfg["epochs"])
        save_every   = int(train_cfg.get("save_every", 1))
        eval_every   = int(train_cfg.get("eval_every", 1))
        early_pat    = int(train_cfg.get("early_stopping_patience", 100))
        resume       = bool(train_cfg.get("resume", False))

        best_eval = float("inf"); early_cnt = 0

        # resume
        if resume:
            ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint.pt")
            if os.path.exists(ckpt_path):
                if self.rank == 0: print(f"[Rank {self.rank}] Resume: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # DDP: load 到 .module
                target = self.model.module if self.is_ddp else self.model
                target.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self.scheduler and "scheduler_state_dict" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", start_epoch) + 1
                best_eval   = ckpt.get("best_eval_loss", best_eval)
                early_cnt   = ckpt.get("early_counter", early_cnt)
            elif self.rank == 0:
                print(f"[Rank {self.rank}] No checkpoint found; fresh start.")

        for epoch in range(start_epoch, final_epoch):
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch, train_loader)
            if self.scheduler:
                self.scheduler.step()
            train_time = time.time() - t0

            eval_loss = None
            if (val_loader is not None) and (epoch % eval_every == 0):
                eval_loss, eval_metrics = self.evaluate(val_loader)
            else:
                eval_metrics = {}

            if self.rank == 0:
                print(f"[Epoch {epoch}] train={train_loss:.6f}  eval={eval_loss}  time={train_time:.2f}s")
                log_loss_csv(log_path, epoch, train_loss, eval_loss)

                is_best = (eval_loss is not None) and (eval_loss < best_eval)
                if is_best:
                    best_eval = eval_loss
                    print(f"★ New best eval: {best_eval:.6f}")

                target = self.model.module if self.is_ddp else self.model
                if epoch % save_every == 0 or is_best:
                    save_checkpoint(
                        target, self.optimizer, self.scheduler,
                        epoch, eval_loss, best_eval,
                        checkpoint_dir=ckpt_dir, is_best=is_best, early_counter=early_cnt
                    )

                if eval_loss is not None:
                    if is_best: early_cnt = 0
                    else:
                        early_cnt += 1
                        if early_cnt >= early_pat:
                            print("Early stopping.")
                            break
