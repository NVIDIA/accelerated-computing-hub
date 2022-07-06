# Binder environment

To run these notebooks we need to take the RAPIDS docker image and fix it up for Binder. The `Dockerfile` here sets things like expected `UID`, home directory, etc. It also overrides the default behaviour that starts Jupyter as we want to let Binder do this for us.

## Testing the container

You can test the container with [Container Canary](https://github.com/NVIDIA/container-canary).

### Build the image

```console
$ # From the top level of the repo.

$ docker build -t gpu-python-tutorial:dev -f binder/Dockerfile .
```

### Validate it with `canary`

```console
$ canary validate --file https://raw.githubusercontent.com/NVIDIA/container-canary/main/examples/binder.yaml gpu-python-tutorial:dev
Validating gpu-python-tutorial:dev against binder
 👩 User is jovyan                                   [passed]
 🏠 Home directory is /home/jovyan                   [passed]
 🖥 Has jupyter installed                            [passed]
 🆔 User ID is 1000                                  [passed]
 🌏 Starts Jupyter on port 8888                      [passed]
validation passed
```
