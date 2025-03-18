import zen_temple
import zen_temple.main
import zen_temple.config
import uvicorn
import argparse
import os
import webbrowser
from typing import Optional


def start(path: str = "./outputs", api_url: Optional[str] = None, port: int = 8000, app_name: str = ''):
    if api_url is None:
        api_url = f"http://127.0.0.1:{port}/api/"

    zen_temple.config.config.SOLUTION_FOLDER = path
    env_path = os.path.join(
        os.path.dirname(zen_temple.__file__), "explorer", "_app", "env.js"
    )

    try:
        os.remove(env_path)
    except FileNotFoundError:
        pass

    with open(env_path, "w") as f:
        f.write(f'export const env={{"PUBLIC_TEMPLE_URL":"{api_url}", "PUBLIC_APP_NAME":"{app_name}"}}')

    print(
        f"Starting visualization, looking for solutions in {path}. The frontend uses the API under {api_url}"
    )
    print(f"Open http://127.0.0.1:{port}/ to look at your solutions.")

    config = uvicorn.Config("zen_temple.main:app", port=int(port), log_level="info")
    server = uvicorn.Server(config)

    webbrowser.open(f"http://127.0.0.1:{port}/", new=2)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the ZEN Temple API Server for the visualizations.",
        add_help=True,
    )

    parser.add_argument(
        "-o",
        "--outputs-folder",
        required=False,
        type=str,
        default="./outputs",
        help="The folder where the outputs files are stored.",
    )

    parser.add_argument(
        "--api_url",
        required=False,
        type=str,
        default=None,
        help="Overwrite the URL for the API server. This means that the temple only serves as a webserver for the ZEN Explorer files.",
    )

    parser.add_argument(
        '--app_name',
        required=False,
        type=str,
        default=None,
        help="Set the name of the app that is shown in the browser tab.",
    )

    parser.add_argument(
        "-p",
        "--port",
        required=False,
        type=int,
        default=8000,
        help="Overwrite the port on which the server is running.",
    )

    args = parser.parse_args()

    start(path=args.outputs_folder, api_url=args.api_url, port=args.port, app_name=args.app_name)
