# How to use

> I haven't figured out how to run carla in a customed docker yet.
> But ortunately, all APIs of CARLA are accessed through local network loopback, so I only need to configure network_mode=host to make it work.
> So I just pull the carla image from dockerhub and run it in a container.
> Still I am not sure whether the stuff in the distribution of carla is of any use, so I download it in case.

## Download CARLA

Just run `./download_carla.sh`.  
At the time I write this, I am using CARLA 0.9.10.1 which is the same version as the original repository.

## Build work container

```bash
cd docker
docker compose build
docker compose up -d
```
