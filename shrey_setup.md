conda create -n toolorchestra-local python=3.10 -y
conda activate toolorchestra-local
pip install -U pip setuptools wheel
export REPO_PATH="$PWD"
export HF_HOME="$HOME/.cache/huggingface"

# Installing lighter dependencies instead of flash attention, etc.
pip install requests openai pyyaml tqdm
pip install transformers anthropic