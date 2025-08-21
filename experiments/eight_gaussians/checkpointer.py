import optax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState


def new_train_state(rng_key, model, init_batch):
  variables = model.init(
    {"params": rng_key, "sample": rng_key},
    inputs=init_batch["inputs"],
    condition=init_batch["condition"],
  )
  tx = optax.adamw(0.0001)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=tx,
  )


def get_checkpointer_fns(outfolder):
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()

  def save_fn(ckpt):
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(outfolder, ckpt, save_args=save_args, force=True)

  def restore_fn():
    return checkpointer.restore(outfolder)

  return save_fn, restore_fn
