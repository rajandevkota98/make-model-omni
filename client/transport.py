"""Talk to the deployed Modal class. Auth is your local Modal token, so there is no public URL."""
from client.conversation import Respond


def connect(server_cfg: dict) -> Respond:
    import modal

    remote = modal.Cls.from_name(server_cfg["app_name"], server_cfg["class_name"])()
    return lambda messages: remote.respond.remote(messages)
