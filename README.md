<p align="left">
    中文&nbsp ｜ &nbsp<a href="./README_en.md">English</a>
</p>
<div align="center">
  <img src="./images/通古logo.png" width="400"/>
</div>


# 通古大模型

## 介绍

通古大模型是华南理工大学深度学习与视觉计算实验室（SCUT-DLVCLab）开发的古籍多模态模型，具备较强大的古籍文字识别和理解能力。

## 开源清单

#### 数据集
CCS358K: 35.8万古籍多模态微调数据，涵盖古籍文字识别、阅读理解、文言文翻译等任务。

CCS358K数据集只能用于非商业研究目的。对于想要使用CCS358K数据集的学者或组织，请先填写此[申请表](./application-form/Application-Form-for-Using-CCS358K.docx)并通过电子邮件发送给我们。向我们提交申请表时，请列出或附上您近6年发表的论文1-2篇，以表明您（或您的团队）在古籍领域进行研究。 我们收到并批准您的申请后，将为您提供下载链接和解压密码。 所有用户必须遵守所有使用条件；否则，将撤销授权。

#### 模型
[**TongGu-VL-2B-Instruct**](https://huggingface.co/SCUT-DLVCLab/TongGu-VL-2B-Instruct): 2B古籍多模态模型，在35.8万古籍多模态语料上做指令微调得到，具备文字识别、书法赏析等功能。



# 新闻

- 2025/07/06 通古论文被ACM MM 2025接收。



# 推理

```python
import torch
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from qwen_vl_utils import process_vision_info


model_id = "/data3/cjh/models/tonggu_vl_models/Tonggu-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

def use_model(input_image, input_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image,
                },
                {"type": "text", "text": input_prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    guided_text = messages[0]["content"][1]["text"] + '<|vision_start|><|image_pad|><|vision_end|>'
    print(guided_text)
    inputs_ocr = processor(text=[guided_text], images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt")
    inputs["input_ids_ocr"] = inputs_ocr["input_ids"]
    inputs["attention_mask_ocr"] = inputs_ocr["attention_mask"]
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.8, top_p=0.95, top_k=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

image = "you image here"
prompt = "识别图中文字"

print(use_model(image, prompt))
```


# 引用

```
@inproceedings{cao2025tonggu,
  title={TongGu-VL: Advancing Visual-Language Understanding in Chinese Classical Studies through Parameter Sensitivity-Guided Instruction Tuning},
  author={Cao, Jiahuan and Liu, Yang and Zhang, Peirong and Shi, Yongxin and Ding, Kai and Jin, Lianwen},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={11111--11120},
  year={2025}
}
```

# 声明：

经过大规模数据的指令微调，TongGu-VL具备较强的古籍多模态理解能力，如文字识别、书法鉴赏等，然而受限于模型规模、自回归生成范式等，TongGu-VL仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎使用和注意甄别，请勿将TongGu-VL生成的有害内容传播至互联网。若产生不良后果，由传播者自负。
