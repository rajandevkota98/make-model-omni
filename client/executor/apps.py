"""open_app, open_url, web_search."""
from urllib.parse import quote_plus

from client.executor.shell import run_first


def open_app(args: dict, env: dict) -> dict:
    return run_first(env["cfg"]["apps"][args["app"]], {}, env, "not_installed", "failed_to_start", detach=True)


def open_url(args: dict, env: dict) -> dict:
    return run_first(env["cfg"]["open_url"], {"url": args["url"]}, env, "not_installed", "failed", detach=True)


def web_search(args: dict, env: dict) -> dict:
    url = env["cfg"]["web_search_url"].format(query=quote_plus(args["query"]))
    return run_first(env["cfg"]["open_url"], {"url": url}, env, "not_installed", "failed", detach=True)
