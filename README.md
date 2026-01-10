# Training MatrixGame on Footsies

Following the recipe of [paper](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/assets/pdf/report.pdf), we choose [SkyReels-V2-I2V-1.3B-540P-Diffusers](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers) as the base model.

## Training Workflow

- preprocess the dataset.
- To ensure compatibility with FastVideo, we first modify the model config to **use Wan model and pipeline**.
- Run the `remove_text_module.py` to strip text components from checkpoint.
- Run finetune stage 1 for a few steps, which ensures the model remains stable after the architectural changes.
- Modify the model config to **use MatrixGame model and pipeline**.
- Run finetune stage 2, train the model to learn action-conditioned.
