import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

# ตรวจสอบว่า GPU พร้อมใช้งานหรือไม่
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.current_device()}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU.")

# โหลดโมเดลและ Tokenizer
model_name = './llama-3-typhoon-v1.5x-8b-instruct'  # โมเดลที่ต้องการใช้
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ย้ายโมเดลไปยัง GPU (หรือ CPU ถ้าไม่มี GPU)
model = model.to(device)

# ตัวอย่างข้อมูลที่ใช้ทดสอบ
input_text = "ประเทศไทยมีกี่ฤดู?"
input_tensor = tokenizer(input_text, return_tensors="pt").to(device)  # ย้ายข้อมูลไปที่ device

# ทดสอบโมเดล
with torch.no_grad():
    output = model.generate(input_tensor['input_ids'], max_length=50)
    
# แสดงผลลัพธ์
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)

# ตรวจสอบการใช้หน่วยความจำ GPU
print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
print(f"Memory Cached: {torch.cuda.memory_reserved()} bytes")
