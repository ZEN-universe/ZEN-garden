import zen_temple
import zen_temple.main
import zen_temple.config
import uvicorn
import sys
import os
import webbrowser


def start(path="./outputs", api_url="http://localhost:8000/api/", port=8000):
    zen_temple.config.config.SOLUTION_FOLDER = path
    env_path = os.path.join(
        os.path.dirname(zen_temple.__file__), "explorer", "_app", "env.js"
    )

    try:
        os.remove(env_path)
    except FileNotFoundError:
        pass

    with open(env_path, "w") as f:
        f.write('export const env={"PUBLIC_TEMPLE_URL":"' + api_url + '"}')

    print(
        f"Starting visualization, looking for solutions in {path}. The frontend uses the API under {api_url}"
    )
    print("Open http://localhost:8000/ to look at your solutions.")

    config = uvicorn.Config("zen_temple.main:app", port=int(port), log_level="info")
    server = uvicorn.Server(config)

    webbrowser.open("http://localhost:8000/", new=2)
    server.run()


if __name__ == "__main__":
    args = {}
    if len(sys.argv) > 1:
        args["path"] = sys.argv[1]

    if len(sys.argv) > 2:
        args["api_url"] = sys.argv[2]

    if len(sys.argv) > 3:
        args["port"] = sys.argv[3]

    start(**args)
