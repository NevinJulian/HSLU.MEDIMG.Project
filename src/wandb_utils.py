"""Optional Weights & Biases integration.

Provides a thin wrapper so notebooks run with or without wandb configured.
Handles the v1 API key format (86 chars) by setting the env var directly
instead of passing the key to wandb.login().
"""

import os


def setup_wandb():
    """Initialize wandb if available and configured. Returns True if active."""
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        print("[wandb] No WANDB_API_KEY found. Running without wandb tracking.")
        return False

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed. Running without tracking.")
        return False

    # The v1 API keys are 86 chars; wandb.login(key=...) rejects them.
    # Setting the env var and calling login without key= works for all formats.
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    try:
        wandb.login()
        print(f"[wandb] Logged in. Project: {os.getenv('WANDB_PROJECT', 'medimg-pneumonia-detection')}")
        return True
    except Exception as e:
        print(f"[wandb] Login failed: {e}. Running without tracking.")
        return False


def wandb_init(**kwargs):
    """Wrapper around wandb.init that returns None if wandb is not active."""
    try:
        import wandb
        if wandb.api.api_key is None:
            return None

        project = kwargs.pop("project", None) or os.getenv("WANDB_PROJECT", "medimg-pneumonia-detection")
        entity = kwargs.pop("entity", None) or os.getenv("WANDB_ENTITY", None)

        return wandb.init(project=project, entity=entity, **kwargs)
    except Exception as e:
        print(f"[wandb] init failed: {e}. Continuing without tracking.")
        return None


def wandb_log(data, run=None):
    """Log metrics. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        wandb.log(data)
    except Exception:
        pass


def wandb_summary(data, run=None):
    """Set summary values. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        for k, v in data.items():
            wandb.summary[k] = v
    except Exception:
        pass


def wandb_log_image(key, fig, run=None):
    """Log a matplotlib figure. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        wandb.log({key: wandb.Image(fig)})
    except Exception:
        pass


def wandb_log_artifact(name, artifact_type, filepath, description="", run=None):
    """Log a file as a wandb artifact. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(name=name, type=artifact_type, description=description)
        artifact.add_file(str(filepath))
        run.log_artifact(artifact)
    except Exception:
        pass


def wandb_log_table(key, columns, data, run=None):
    """Log a wandb Table. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})
    except Exception:
        pass


def wandb_finish(run=None):
    """Finish a run. No-op if run is None."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
