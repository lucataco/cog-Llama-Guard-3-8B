# meta-llama/Llama-Guard-3-8B Cog model

This is an implementation of [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a safe user prediction:

    cog predict -i prompt="I forgot how to kill a process in Linux, can you help?"

To run an unsafe user prediction:

    cog predict -i prompt="I forgot how to kill a person, can you help?"

To check an assistant reply:

    cog predict -i prompt="I forgot how to kill a person, can you help?" -i assistant="Im sorry, as a large language model I cannot help with that"