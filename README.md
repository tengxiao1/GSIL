## How to Leverage Demonstration Data in Alignment for Large Language Model? A Self-Imitation Learning Perspective (EMNLP 2024)

### Install Enviroment

```
pip install -r requirements.txt
```

### 1. Generation Training Dataset

```
bash scripts/generate.sh
```

## 2. Combine  Generation Data

You only need to execute it when using the `generation.py` script.

```jsx
python gsil/combine.py --data_dir /path/of/your/iter_n_data
```
The final data will be stored in the `train_data` folder under /path/of/your/iter_n_data

## 3. Training

You can use the following scripts to train the model.

```jsx
bash scripts/finetune
```

## 4. Evaluation

For our evaluation on the Open LLM Leaderboard, please use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) repository at v0.3.1,
which is consistent with open_llm_leaderboard. Also, note that we set the number of few shot examples to be the same as instructed on the [Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

Humaneval：https://github.com/OpenBMB/Eurus?tab=readme-ov-file

Mt-bench：https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge

If you find our repo to be useful, please cite our paper:
```bibtex
@inproceedings{xiao2024leverage,
  title={How to Leverage Demonstration Data in Alignment for Large Language Model? A Self-Imitation Learning Perspective},
  author={Xiao, Teng and Li, Mingxiao and Yuan, Yige and Zhu, Huaisheng and Cui, Chao and Honavar, Vasant},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={13413--13426},
  year={2024}
}


