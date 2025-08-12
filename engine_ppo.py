import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import torch.nn.functional as F

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ppo_std_this_iter', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for focus_step_index in range(args.seq_l):
        metric_logger.add_meter(f'pos_x_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for focus_step_index in range(args.seq_l):
        metric_logger.add_meter(f'pos_x_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    mse_loss = torch.nn.MSELoss()

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        if args.ppo_std_decay == 'cosine':
            if epoch < args.warmup_epochs:
                ppo_std_this_iter = args.ppo_std_start
            else:
                ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end
                    ) * lr_schedule_values[it] / args.lr
        elif args.ppo_std_decay == 'linear':
            ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end
                ) * (1 - it / len(lr_schedule_values))
        else:
            raise NotImplementedError


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        with torch.cuda.amp.autocast():

            expected_outputs = model(samples, seq_l=args.seq_l, ppo_std_this_iter=ppo_std_this_iter)

            outputs_reg_focus_net = expected_outputs['outputs_reg_focus_net']
            loss_reg_focus_net = criterion(outputs_reg_focus_net, targets)
            loss_reg_focus_net = args.loss_reg_focus_net_weight * loss_reg_focus_net


            out_teacher = expected_outputs['x_focus'][-1].detach()

            loss_KD = F.kl_div(
                F.log_softmax(expected_outputs['x_glance'][-1] / args.kd_temp, dim=1),
                F.softmax(out_teacher / args.kd_temp, dim=1), 
                reduction='batchmean'
                ) * (args.kd_temp ** 2)

            loss_KD = loss_KD + sum(
                [
                    F.kl_div(
                        F.log_softmax(_x_focus / args.kd_temp, dim=1),
                        F.softmax(out_teacher / args.kd_temp, dim=1), 
                        reduction='batchmean'
                        ) * (args.kd_temp ** 2)
                        for _x_focus in expected_outputs['x_focus'][:-1]
                    ]
            )

            loss_glance = criterion(expected_outputs['x_glance'][-1], targets)
            loss_focus = sum(
                [criterion(_x_focus, targets) for _x_focus in expected_outputs['x_focus']]
            )
            loss = loss_reg_focus_net + loss_focus + loss_glance + loss_KD * args.kd_alpha
            loss_value = loss.item()
                


        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)


        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, 
                                model=model, skip_backbones=False, skip_policy_net=True, 
                                clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

        if data_iter_step == 0:
            ppo_update_collect_dict = []
            targets_collect_dict = []

        ppo_update_collect_dict.append(expected_outputs)
        targets_collect_dict.append(targets)

        if (data_iter_step + 1) % args.update_policy_freq == 0:

            expected_outputs_ppo = {
                'x_glance': [], 
                'x_focus': [], 
                'actions': [], 
                'actions_logprobs': [], 
                '_state_values': [], 
                'states': [],
            }
            for key in ['x_glance', 'x_focus', 'actions', 'actions_logprobs', '_state_values', 'states']:
                for index in range(len(ppo_update_collect_dict[0][key])):
                    expected_outputs_ppo[key].append(
                        torch.cat(
                            [ppo_update_collect_dict[m][key][index] for m in range(args.update_policy_freq)], dim=0
                        )
                    )
            targets = torch.cat(targets_collect_dict, dim=0)
            ppo_update_collect_dict = []
            targets_collect_dict = []

            for name, param in model.named_parameters():
                if 'policy_net_patch' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            _list_all_outputs = expected_outputs_ppo['x_glance'] + expected_outputs_ppo['x_focus']
            list_to_compute_reward = [((
                    F.cross_entropy(_list_all_outputs[focus_step_index].detach(), targets, reduction='none') \
                    - F.cross_entropy(_list_all_outputs[focus_step_index + 1].detach(), targets, reduction='none'))).unsqueeze(-1)
                for focus_step_index in range(args.seq_l)]
            # single step reward
            mb_rewards = torch.cat(list_to_compute_reward, dim=1) # B, seq_l
            
            
            old_states = expected_outputs_ppo['states']
            old_actions = expected_outputs_ppo['actions']
            old_actions_logprobs = torch.cat(expected_outputs_ppo['actions_logprobs'], dim=1).detach() # B, seq_l
            old_state_values = torch.cat(expected_outputs_ppo['_state_values'], dim=1).detach() # B, seq_l

            
            # GAE
            mb_values = old_state_values
            mb_advs = torch.zeros_like(mb_rewards)
            lastgaelam = torch.zeros((mb_advs.shape[0], ), device=model.device)
            for t in reversed(range(args.seq_l)):
                if t == args.seq_l - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0
                else:
                    nextnonterminal = 1.0
                    nextvalues = mb_values[:, t+1]
                delta = mb_rewards[:, t] + args.gamma * nextvalues * nextnonterminal - mb_values[:, t]
                mb_advs[:, t] = lastgaelam = delta + args.gamma * args.ppo_lam * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values # as value function update target
            
            if args.adv_normalization:
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
            
            advantages = mb_advs
            
            num_ppo_update_iters = args.num_ppo_update_iters
            ppo_total_batch_size = args.batch_size * args.update_policy_freq
            mini_batch_size = int(ppo_total_batch_size / num_ppo_update_iters)
            
            for _ in range(args.ppo_update_steps):
                inds = torch.randperm(ppo_total_batch_size)
                for i in range(num_ppo_update_iters):
                    
                    batch_index = inds[torch.arange(i*mini_batch_size, (i+1)*mini_batch_size)]
                    # Evaluating old actions and values
                    with torch.cuda.amp.autocast():
                        logprobs, state_values, dist_entropy = model(
                            old_states=old_states, old_actions=old_actions, seq_l=args.seq_l,
                            flag='evaluate_policy_net', batch_index=batch_index, ppo_std_this_iter=ppo_std_this_iter
                            )
                        # Finding the ratio (pi_theta / pi_theta__old)
                        ratios = torch.exp(logprobs - old_actions_logprobs[batch_index].detach())

                        # Finding Surrogate Loss  
                        surr1 = ratios * advantages[batch_index]
                        surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages[batch_index]

                        # final loss of clipped objective PPO
                        loss = - torch.min(surr1, surr2).mean() + mse_loss(state_values, mb_returns[batch_index].detach()) - 0.01 * dist_entropy.mean()
                        
                    # take gradient step
                    optimizer.zero_grad()
                    __ = loss_scaler(loss, optimizer, model=model, skip_backbones=True, skip_policy_net=False,
                                        clip_grad=args.ppo_clip_grid,
                                        parameters=model.parameters())

            for name, param in model.named_parameters():
                if 'policy_net_patch' in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

        torch.cuda.synchronize()


        metric_logger.update(loss=loss_value)
        for focus_step_index in range(args.seq_l):
            exec(f"metric_logger.update(pos_x_std_{focus_step_index}=expected_outputs['pos_std'][focus_step_index][1])")
            exec(f"metric_logger.update(pos_y_std_{focus_step_index}=expected_outputs['pos_std'][focus_step_index][0])")
            exec(f"metric_logger.update(pos_x_mean_{focus_step_index}=expected_outputs['pos_mean'][focus_step_index][1])")
            exec(f"metric_logger.update(pos_y_mean_{focus_step_index}=expected_outputs['pos_mean'][focus_step_index][0])")
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(ppo_std_this_iter=ppo_std_this_iter)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()


    for batch in metric_logger.log_every(data_loader, 10, header):
        
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # x_glance, x_focus, actions_logprobs, _state_values, pos_std, pos_mean
            expected_outputs = model(images, seq_l=args.seq_l)
            output = expected_outputs['x_focus'][-1]
            output_glance = expected_outputs['x_glance'][-1]
            loss = criterion(output, target)
            loss_glance = criterion(output_glance, target)


        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_glance, acc5_glance = accuracy(output_glance, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        for focus_step_index in range(args.seq_l):
            acc1_this_step, acc5_this_step = accuracy(expected_outputs['x_focus'][focus_step_index], target, topk=(1, 5))
            metric_logger.meters[f'acc1_step_{focus_step_index}'].update(acc1_this_step.item(), n=batch_size)
            metric_logger.meters[f'acc5_step_{focus_step_index}'].update(acc5_this_step.item(), n=batch_size)
        
        metric_logger.update(loss_glance=loss_glance.item())
        metric_logger.meters['acc1_glance'].update(acc1_glance.item(), n=batch_size)
        metric_logger.meters['acc5_glance'].update(acc5_glance.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    for focus_step_index in range(args.seq_l):
        exec(f"print(metric_logger.acc1_step_{focus_step_index}.global_avg)")

    print('* Glance Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1_glance, top5=metric_logger.acc5_glance, losses=metric_logger.loss_glance))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
