import sys
from argparse import ArgumentParser, Namespace
from time import sleep

from azureml.core import Environment, Workspace
from docker.client import DockerClient
from docker.models.images import Image

ws = Workspace.from_config()


def parse_args() -> Namespace:
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--name", '-n', type=str, required=True,
                        help="The name of the registered environment to build")
    parser.add_argument("--version", '-v', type=int,
                        help="The version of the registered environment to build",
                        default=None)
    parser.add_argument("--async", "-q", dest="run_async", action="store_true",
                        help="Run asyncronously")
    parser.add_argument('--local', '-l', action="store_true",
                        help="Build the environment on the local compute context")
    parser.add_argument('--force', '-F', action="store_true",
                        help="Force image rebuild")
    return parser.parse_args()


def pull_image(img_name: str, registry_details: dict) -> None:
    """Pull the docker images locally and remove the registry info from the tag"""
    print(f"Pulling image from '{registry_details['registry']}'")
    client = DockerClient.from_env()
    client.login(**registry_details)
    img = client.images.pull(f"{registry_details['registry']}/{img_name}")
    if isinstance(img, Image):
        img.tag(img_name)
    else:
        img[0].tag(img_name)


def parse_image_details(environ: Environment) -> dict:
    """
    Pull out the image name and registry details from the environment object

    Since the docker.login method expects 'registry' instead of 'address', rename 'address' to 'registry'
    """
    docker_details = environ.get_image_details(ws)['dockerImage']
    return {"img_name": docker_details['name'],
            "registry_details": {"registry" if k == 'address' else k: v for
                                 k, v in docker_details['registry'].items()}}


def is_environment_built(environ: Environment) -> bool:
    """
    Checks if the environment is already built in Azure Machine Learning
    and stored in a container registry
    """
    env_img_details = environ.get_image_details(ws)
    return env_img_details['imageExistsInRegistry']


def main(name: str, version: int, run_async: bool, local: bool, force: bool):
    """If the environment is already built, either exit or pull it locally"""
    environ = Environment.get(ws, name=name, version=version)

    if is_environment_built(environ=environ) and not force:
        if local:
            print(
                f"'{name}' is already built - pulling locally. Use '--force' to force a rebuild on the local context")
            pull_image(**parse_image_details(environ))
            sys.exit(0)
        else:
            print(f"'{name}' is already built. Use '--force' to force a rebuild")
            sys.exit(0)
    else:
        if local:
            environ.build_local(ws)
        else:
            build = environ.build(ws)
            if not run_async:
                sleep(5)
                build.wait_for_completion(show_output=True)


if __name__ == "__main__":

    args = parse_args()
    main(name=args.name, version=args.version,
         run_async=args.run_async, local=args.local, force=args.force)
