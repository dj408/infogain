"""
Function to train/fine-tune a PyTorch model, 
using an Accelerator wrapper for CPU-GPU(s)
device support.

Ref:
https://huggingface.co/docs/accelerate/en/package_reference/accelerator
"""

import utilities as u
import nn_utilities as nnu
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any
)


def train_model(
    args,
    model: nn.Module,
    data_container: Dict[str, DataLoader] | Data,
    optimizer: optim.Optimizer,
    validate_every: int = 1,
    grad_track_param_names: Optional[Tuple[str]] = None,
    snapshot_path: Optional[str] = None,
    save_states: bool = False,
    save_final_model_state: bool = False,
    return_best: bool = True,
    use_acc_print: bool = False,
    using_pytorch_geo: bool = False,
    fold_i: Optional[int] = None,
    verbosity: int = 0
) -> Tuple[
        Optional[nn.Module], # final model
        Optional[List[Dict[str, Any]]], # train history records
        Optional[nnu.EpochCounter] # EpochCounter object
]:
    """
    A general training function for base_module.BaseModule models, and
    potentially other pytorch models.

    Features:
    - Pass an optional `snapshot_path` to load an accelerate model snapshot and 
      resume training; else leave `None`.
    
    Notes:
    - The loss function must be an attrib. of the model, to get placed on the 
    device properly with Accelerate.
    - It's possible to keep training to the max number of epochs if
    validation loss kept improving by an arbitrarily small
    amount, but return a final model from a much earlier epoch, if 
    'args.MAIN_METRIC_REL_IMPROV_THRESH' is 1.0 or not None. That is,
    the model returned will be that with the weights where the valid
    loss last improved by this thresholded ratio.
    
    Args:
        args: ArgsTemplate object containing experiment arguments.
        model: the torch.nn.Module model object to train. Must have 
            'forward', 'loss', and 'update_metrics' methods that
            take dictionaries, as done in base_module.BaseModule.
        data_container: dictionary of Dataloaders by set, or pytorch_geometric 
            Data object containing the training data and set masks.
        optimizer: torch optimizer object, such as Adam or AdamW.
        validate_every: run validation phase after this many epochs
            (default is 1 = every epoch).
        grad_track_param_names: tuple of strings of parameter names
            for which their gradients and weights will be tracked.
            See 'nnu.log_parameter_grads_weights' for details.
        snapshot_path: directory path string from which to load a
            previously trained model, e.g. for resuming training.
        save_states: if True, save (overwrite) the best model obtained
            (by a new best main metric validation set score) as 'best' 
            in the 'snapshot_path' directory, each time a new best is 
            obtained ('checkpointing'). If False, the best model can still 
            be returned (not saved to disk) if 'return_best' is True.
        save_final_model_state: if True, the last (and likely not best)
            state of the model with weights from the last epoch reached
            will be saved (as 'final', not 'best').
        return_best: whether to return the trained model object with
            its best weights, plus the training records and epoch counter 
            objects.
        use_acc_print: if True, use Accelerator's print method, instead
            of base python's 'print'.
        using_pytorch_geo: boolean indicating whether a pytorch_geometric
            data container object is being used, instead of torch's
            base DataLoaders.
        fold_i: if this function is being used within k-folds cross
            validation, this variable sets its awareness of which fold
            its in.
        verbosity: controls volume of print output
            as the function runs. >1 prints epoch-
            by-epoch loss and time summaries.
    Returns:
        3-tuple of (model, records, epoch_ctr) if 'return_best' is True, 
        or else (None, None, None) if error (e.g. no model weights were 
        created) or 'return_best' is False. 
    """

    """
    INNER FUNCTIONS/CLASSES
    """
    def _save_snapshot(name):
        if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ''):
            snapshot_path = f'{args.MODEL_SAVE_DIR}/{name}'
            acc.save_state(snapshot_path)

    def _log_output(out):
        if use_acc_print:
            acc.print(out)
        else:
            with open(args.PRINT_DIR, 'a') as f:
                f.write(out + '\n')

            
    """
    INITIALIZE DIRS, WEIGHTS, METRICS
    """
    if verbosity > 0:
        print('save_states:', save_states)
    best_model_wts = copy.deepcopy(model.state_dict())
    if (args.MODEL_SAVE_DIR is not None) and (args.MODEL_SAVE_DIR != ""):
        os.makedirs(args.MODEL_SAVE_DIR, exist_ok=True)
        
    # store metrics by epoch in list of dicts 
    # (i.e. 'records' -> easy to convert to pd.DataFrame)
    records = []
    
    # initialize EpochCounter
    epoch_ctr = nnu.EpochCounter(0, args.MAIN_METRIC)
    best_epoch = 1


    """
    ACCELERATOR WRAPPER
    """
    acc = Accelerator(
        device_placement=args.DEVICE, 
        cpu=args.ON_CPU
    )

    # wrap training objects
    model, optimizer = acc.prepare(model, optimizer)
    if type(data_container) is dict:
        data_container['train'], data_container['valid'] = acc.prepare(
             data_container['train'], 
             data_container['valid']
         )
    else:
        data_container = acc.prepare(data_container)
    
    # custom objects must be 'registered for checkpointing'
    acc.register_for_checkpointing(epoch_ctr)
    acc_state = AcceleratorState(cpu=args.ON_CPU)
    num_devices, device_type, distr_type = (
        acc_state.num_processes, 
        acc_state.device, 
        acc_state.distributed_type
    )

    """
    log key settings
    """
    out = nnu.get_model_settings_str(args)
    _log_output(out)

    # log training hardware info
    # print(f'AcceleratorState device: {acc_state.device}')
    distributed_str = distr_type.split('.')[0]
    out = f'Training on {num_devices} x {device_type}' \
        + f' device (distributed: {distributed_str})'
    if verbosity > 0:
        print(out)
    _log_output(out)

    
    """
    OPTIONAL: LOAD MODEL SNAPSHOT
    - to resume training from saved model state
    """
    if snapshot_path is not None:
        acc.load_state(snapshot_path)
        out = f'...resuming training from snapshot at epoch {epoch_ctr.n}'
        _log_output(out)

    
    """
    TRAINING LOOP
    """
    time_0 = time.time()
    ul_str = '-' * 12
    num_epochs_no_vl_improv = 0
    best_main_metric_score = -1
    last_epoch_flag = False

    # save initial model weights
    if grad_track_param_names is not None:
        nnu.log_parameter_grads_weights(
            args=args,
            model=model,
            grad_track_param_names=grad_track_param_names,
            epoch_i=-1, 
            batch_i=-1,
            save_grads=False
        )

    # classification task: print class 1 preds proportion for
    # each epoch and phase; here, init empty counters container
    if (verbosity > 0) and ('class' in args.TASK):
        class1_preds_ctr = nnu.Class1PredsCounter()

    # loop through (marginal) epochs
    last_ctr_epoch = epoch_ctr.n + args.N_EPOCHS
    for epoch in range(epoch_ctr.n + 1, last_ctr_epoch + 1):
        time_epoch_0 = time.time()
        epoch_ctr += 1
        is_validation_epoch = (epoch % validate_every == 0)
        out = f'\nEpoch {epoch}/{args.N_EPOCHS}\n{ul_str}'
        _log_output(out)
        if verbosity > 0:
            print(out)      

        # each epoch has a training and maybe a validation phase
        phases = ('train', 'valid') if is_validation_epoch else ('train', )
        for phase in phases:
            training = (phase == 'train')
            model.train() if training else model.eval()

            with torch.set_grad_enabled(training):

                if using_pytorch_geo:
                    # if 'data_container' is a dictionary (with 'train' and 'valid'
                    # sets), loop through batches
                    if isinstance(data_container, dict):
                        for batch_i, batch in enumerate(data_container[phase]):
                            optimizer.zero_grad()

                            # PATCH: batches of size 1 cause errors in mfcn's forward()
                            # but using 'drop_last=True' in DataLoader also errors
                            if batch.num_graphs > 1:
                                output_dict = model(batch)
                                # print("output_dict['preds'].shape", output_dict['preds'].shape)
                                input_dict = {'target': batch.y}
                                loss_dict = model.loss(input_dict, output_dict)
                                if torch.isnan(loss_dict['loss']):
                                    raise Exception("Loss function returned NaN!")
    
                                # classification task: update class 1 preds counts, each batch
                                if (verbosity > 0) and ('class' in args.TASK):
                                    class1_preds_ctr.update(output_dict, phase)

                    # elif 'data_container' is a torch_geometric.data.Data object,
                    # there are no batches; during loss calc, use 'train' and 'val' mask 
                    # attributes;
                    # note that in a node-level task, we assume we have full knowledge 
                    # of the ($k$-NN) graph structure and signals, and only withhold 
                    # valid/test-set node targets at loss calculation and evaluation time
                    elif isinstance(data_container, Data):
                        batch_i = 0
                        optimizer.zero_grad()
                        output_dict = model(data_container)
                        preds = output_dict['preds']
                        target = data_container.y[data_container.train_mask] \
                            if training else data_container.y[data_container.val_mask]
                        input_dict = {'target': target}
                        output_dict['preds'] = preds[data_container.train_mask] \
                            if training else preds[data_container.val_mask]
                        loss_dict = model.loss(input_dict, output_dict)
                        if torch.isnan(loss_dict['loss']):
                            raise Exception("function returned NaN!")

                        # classification task: update class 1 preds counts, once
                        if (verbosity > 0) and ('class' in args.TASK):
                            class1_preds_ctr.update(output_dict, phase)

                    # for both dicts of DataLoaders and single Data objects with masks:
                    # train phase only: backward pass and optimizer step
                    if training:
                        acc.backward(loss_dict['loss'])
                        if grad_track_param_names is not None:
                            nnu.log_parameter_grads_weights(
                                args,
                                model,
                                grad_track_param_names,
                                epoch - 1, 
                                batch_i
                            )
                        optimizer.step()
                        
                    # update batch loss (test and valid) and metrics (valid only)
                    model.update_metrics(
                        phase, 
                        loss_dict, 
                        input_dict, 
                        output_dict
                    )

                # not using a pytorch geometric DataLoaders or Data object
                else: 
                    for batch_i, input_dict in enumerate(data_container[phase]):
                        optimizer.zero_grad()
                        output_dict = model(input_dict) # calls model.forward
                        loss_dict = model.loss(input_dict, output_dict)

                        # train phase only: backward pass and optimizer step
                        if training:
                            acc.backward(loss_dict['loss'])
                            if grad_track_param_names is not None:
                                nnu.log_parameter_grads_weights(
                                    args,
                                    model, 
                                    grad_track_param_names,
                                    epoch - 1, 
                                    batch_i
                                )
                            optimizer.step()
                            
                        # update batch loss (test and valid) and metrics (valid only)
                        model.update_metrics(
                            phase, 
                            loss_dict, 
                            input_dict, 
                            output_dict
                        )

                        # classification task: update class 1 preds counts, each batch
                        if (verbosity > 0) and ('class' in args.TASK):
                            class1_preds_ctr.update(output_dict, phase)

        
        # after both train and valid sets are complete
        # calc epoch losses/metrics
        epoch_hist_d = model.calc_metrics(epoch, is_validation_epoch, input_dict)

        # classification task: print class 1 preds counts
        if is_validation_epoch and (verbosity > 1) and ('class' in args.TASK):
            class1_preds_ctr.print_preds_counts()
            class1_preds_ctr.reset()

        # log/print losses
        train_loss = epoch_hist_d['loss_train']
        valid_loss = epoch_hist_d['loss_valid'] if is_validation_epoch else None
        epoch_time_elapsed = time.time() - time_epoch_0
        epoch_min, epoch_sec = int(epoch_time_elapsed // 60), epoch_time_elapsed % 60
        out = f"time elapsed: {epoch_min}m, {epoch_sec:.2f}s" \
            + f"\nlosses:\n\ttrain: {train_loss:.6e}"
        if is_validation_epoch:
            out += f"\n\tvalid: {valid_loss:.6e}"
        _log_output(out)
        if verbosity > 0:
            print(out)

        # validation phases: early stopping and train history / best model saving steps
        if (phase == 'valid'):

            # reset 'new best metric score reached' flags
            new_best_score_reached, score_thresh_reached = False, False

            # grab current epoch's score for main metric
            epoch_main_metric_score = epoch_hist_d[args.MAIN_METRIC]

            # first validation phase only: set first main metric score as the 
            # score to beat in subsequent validation epochs
            if epoch_ctr.n == validate_every:
                epoch_ctr.set_best(args.MAIN_METRIC, epoch, epoch_main_metric_score)
            
            # check for new best validation loss; if not, increment the
            # 'no improvement' counter
            if valid_loss < epoch_ctr.best['_valid_loss']['score']:
                epoch_ctr.set_best('_valid_loss', epoch, valid_loss)
                num_epochs_no_vl_improv = 0
            else:
                num_epochs_no_vl_improv += validate_every # += 1

            # if in final desired epoch, or (burn-in period passed AND 'patience' num 
            # epochs w/o valid loss improvement reached): set 'last_epoch_flag=True', 
            # which will break the epochs' for-loop at end of the current epoch
            if (epoch == last_ctr_epoch) \
            or (
                (epoch > args.BURNIN_N_EPOCHS) \
                and (num_epochs_no_vl_improv >= args.NO_VALID_LOSS_IMPROVE_PATIENCE) \
                and (args.STOP_RULE is not None) \
                and ('no' in args.STOP_RULE) \
                and ('improv' in args.STOP_RULE)
            ):
                last_epoch_flag = True
                out = f'Validation loss did not improve for' \
                      + f' {num_epochs_no_vl_improv} epochs: stopping.'
                print(out)
                _log_output(out)

            # check for new best key validation score (by a margin)
            best_main_metric_score = epoch_ctr.best[args.MAIN_METRIC]['score']
            score_thresh = best_main_metric_score \
                if args.MAIN_METRIC_REL_IMPROV_THRESH is None \
                else (best_main_metric_score * args.MAIN_METRIC_REL_IMPROV_THRESH)
            if args.MAIN_METRIC_IS_BETTER == 'lower':
                score_thresh_reached = (epoch_main_metric_score < score_thresh)
            elif args.MAIN_METRIC_IS_BETTER == 'higher':
                score_thresh_reached = (epoch_main_metric_score > score_thresh)
    
            # if new best validation score threshold reached, record it
            if score_thresh_reached:
                new_best_score_reached = True
                best_main_metric_score = epoch_hist_d[args.MAIN_METRIC]
                epoch_ctr.set_best(args.MAIN_METRIC, epoch, best_main_metric_score)
                epoch_key = f"epoch_{epoch}"
                on_best_model_kwargs = {
                    'save_path': args.MODEL_SAVE_DIR,
                    'fold_i': fold_i
                }
                model.on_best_model(on_best_model_kwargs)

            # append this epoch's losses, metrics, and time elapsed to records
            # include epoch training time and reigning epoch with best validation score
            epoch_hist_d['sec_elapsed'] = epoch_time_elapsed
            epoch_hist_d['best_epoch'] = epoch_ctr.best[args.MAIN_METRIC]['epoch']
            records.append(epoch_hist_d)
    
            # if new best epoch (by main metric validation set score):
            if new_best_score_reached:
                # print msg
                score_str = f"{args.MAIN_METRIC}={epoch_hist_d[args.MAIN_METRIC]:.6e}"
                out = f"-> New best model!" # {score_str}
                _log_output(out)
                if verbosity > 0:
                    print(out)
                    
                if save_states:
                    print(f'Saving model...')
                    # save (overwrite) 'best' model and training logs to reach it
                    _save_snapshot('best')
                    u.pickle_obj(args.TRAIN_LOGS_SAVE_DIR, records)
                    if verbosity > 0:
                        _log_output(f'Model saved.')
                        
                if return_best:
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                # new main metric score wasn't the best seen
                prev_best_epoch = epoch_ctr.best[args.MAIN_METRIC]['epoch']
                out = f"[Current best epoch: {prev_best_epoch}]"
                _log_output(out)
                if verbosity > 0:
                    print(out)

            # last_epoch_flag has been set, break out of epochs' for-loop and
            # jump to POST-TRAINING section
            if last_epoch_flag:
                break

    """
    POST-TRAINING
    """
    # get total time elapsed
    t_min, t_sec = u.get_time_min_sec(time.time(), time_0)
    out = f'{epoch_ctr.n} epochs complete in {t_min:.0f}min, {t_sec:.1f}sec.'
    _log_output(out)
    print(out)

    # log final best validation score and epoch
    if epoch_ctr.n > args.BURNIN_N_EPOCHS:
        best_epoch = epoch_ctr.best[args.MAIN_METRIC]['epoch']
        out = f'Best {args.MAIN_METRIC}: {best_main_metric_score:.4f}' \
            + f' at epoch {best_epoch}'
        _log_output(out)

    # save final training log
    u.pickle_obj(args.TRAIN_LOGS_FILENAME, records, overwrite=False)
    print(f'Final training log saved.')

    # optional: save final epoch's model state
    if save_final_model_state:
        _save_snapshot('final')
        print(f'Last model state saved.')
        

    # optional: load best model weights and return tuple with history log
    if return_best:
        if best_model_wts is not None:
            out = f'Returning model with best weights (from epoch {best_epoch}).\n'
            _log_output(out)
            print(out)
            model.load_state_dict(best_model_wts)
            return (model, records, epoch_ctr)
        else:
            out = f'No best model found; no weights were saved!\n'
            _log_output(out)
            print(out)
            return (None, None, None)
    else:
        return (None, None, None)

