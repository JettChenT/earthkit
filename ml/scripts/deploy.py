import os
import typer

app = typer.Typer()

APP_NAMES = [
    "endpoint",
    "geoclip_inference",
    "satellite",
    "streetview_dl",
    "vpr",
    "crossview",
    "satellite"
]

@app.command()
def deploy(apps: str, env_name: str = "dev", prod:bool=False):
    if prod:
        env_name = "main"
    app_list = apps.split(",") if apps else APP_NAMES

    for app_name in app_list:
        if app_name in APP_NAMES:
            os.system(f"modal deploy src.{app_name} --env {env_name}")
        else:
            typer.echo(f"App {app_name} not found in APP_NAMES")

if __name__ == "__main__":
    app()
