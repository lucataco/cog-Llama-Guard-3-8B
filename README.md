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

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) is released under the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-Guard-3-8B/blob/main/LICENSE). Check its terms before use.
