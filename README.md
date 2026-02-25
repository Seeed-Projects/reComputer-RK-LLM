[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
# Introduction
This repository utilizes [Docker](https://www.docker.com/) to package large language models and multimodal models optimized for Rockchip platforms. It provides a unified calling interface that is compatible with the OpenAI API, making it easy for users to integrate and use these models.

# Hardware Prepare

For reComputer RK3588 and reComputer RK3576.

## LLM

[Fast start](./LLM.md)

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-deepseek-r1-distill-qwen:7b-w8a8-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/682844605?tag=7b-w8a8-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/682838759?tag=1.5b-fp16-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/682835173?tag=1.5b-w8a8-latest) | 
| **RK3576** | [rk3576-deepseek-r1-distill-qwen:7b-w4a16-g128-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/682837577?tag=7b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:7b-w4a16-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/682832575?tag=1.5b-w4a16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/682834349?tag=1.5b-fp16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-g128-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/682832500?tag=1.5b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/682832575?tag=1.5b-w4a16-latest) | 

## VLM

[Fast start](./VLM.md)

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-qwen2-vl:7b-w8a8-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/682842044?tag=7b-w8a8-latest)<br>[rk3588-qwen2-vl:2b-w8a8-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/682835375?tag=2b-w8a8-latest)<br> | 
| **RK3576** | [rk3576-qwen2.5-vl:3b-w4a16-latest](https://github.com/Seeed-Projects/reComputer-RK-LLM/pkgs/container/rk3576-qwen2.5-vl/682834538?tag=3b-w4a16-latest)<br>| 


# Speed test

> Note: A rough estimate of a model's inference speed includes both TTFT and TPOT.
> Note: You can use `python test_inference_speed.py --help` to view the help function.

```bash
python -m venv .env && source .env/bin/activate
pip install requests
python llm_speed_test.py
```

# 💞 Top contributors:

<a href="https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Seeed-Projects/reComputer-RK-LLM" alt="contrib.rocks image" />
</a>

# 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=Seeed-Projects/reComputer-RK-LLM&type=Date)

Reference: [rknn-llm](https://github.com/airockchip/rknn-llm/tree/main)


[contributors-shield]: https://img.shields.io/github/contributors/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[contributors-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[forks-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/network/members
[stars-shield]: https://img.shields.io/github/stars/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[stars-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[issues-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/issues
[license-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[license-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/blob/master/LICENSE.txt
