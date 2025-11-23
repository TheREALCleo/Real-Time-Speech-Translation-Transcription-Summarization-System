import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import gc
import os
from dotenv import load_dotenv
load_dotenv()

class MeetingSummarizer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Instruct-2507"):
        print("Loading model and tokenizer...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="cuda",
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully with 4-bit quantization!")
    
    def parse_srt(self, srt_path):
        """Extract text content from SRT file"""
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove subtitle numbers and timestamps
        text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        # Remove empty lines and join
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        transcript = ' '.join(lines)
        return transcript
    
    def create_mom_prompt(self, transcript):
        """Create a detailed prompt for generating Minutes of Meeting in markdown"""
        prompt = f"""You are an expert meeting summarizer. Analyze the following meeting transcript and create comprehensive Minutes of the Meeting (MOM) in proper markdown format.

**Instructions:**
- Use proper markdown formatting with headers, lists, and emphasis
- Extract all key discussion points and decisions
- Identify action items with responsible parties (if mentioned)
- Note any deadlines or timelines discussed
- Highlight important agreements or conclusions
- Be concise but capture all critical information
- Use bullet points for lists, **bold** for emphasis

**Output Format (in markdown):**

# Minutes of the Meeting

## Meeting Information
- **Date:** [Extract if mentioned, otherwise Skip]
- **Time:** [Extract if mentioned, otherwise Skip]
- **Duration:** [If mentioned]
- **Location/Platform:** [If mentioned]

## Attendees
- **Present:** [List names if mentioned]
- **Absent:** [If mentioned]

## Agenda
[Brief overview of meeting purpose]

## Key Discussion Points
- [Main topic 1]
  - [Subtopic or detail]
- [Main topic 2]
  - [Subtopic or detail]

## Decisions Made
- **Decision 1:** [Clear statement of decision]
- **Decision 2:** [Clear statement of decision]


## Key Agreements
- [Important agreement 1]
- [Important agreement 2]

## Next Steps
- [Future action 1]
- [Future action 2]

## Next Meeting
- **Date:** [If mentioned]
- **Agenda:** [If mentioned]

## Additional Notes
- [Any other relevant information]

---

**Meeting Transcript:**
{transcript}

**Generate the Minutes of the Meeting in markdown format:**"""
        return prompt
    
    def generate_mom(self, srt_path, max_length=2048, temperature=0.3):
        """Generate MOM from SRT file"""
        transcript = self.parse_srt(srt_path)
        
        prompt = self.create_mom_prompt(transcript)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        print("Generating Minutes of the Meeting...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids 
            in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clear cache after generation
        del model_inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return response
    
    def save_mom(self, mom_text, output_path):
        """Save MOM to markdown file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<!--- Generated Minutes of the Meeting --->\n\n")
            f.write(mom_text)
        print(f"Minutes saved to: {output_path}")
    
    def cleanup(self):
        """Clean up model and free memory"""
        print("Cleaning up resources...")
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Resources cleared!")


if __name__ == "__main__":
    try:
        summarizer = MeetingSummarizer()
        
        srt_file = os.getenv('in_srt')
        output_file = os.getenv('out_sum')
        
        mom = summarizer.generate_mom(
            srt_file,
            max_length=3072,
            temperature=0.3
        )
        
        print("\n" + "="*60)
        print("GENERATED MINUTES OF THE MEETING")
        print("="*60 + "\n")
        print(mom)
        summarizer.save_mom(mom, output_file)      
    finally:
        if 'summarizer' in locals():
            summarizer.cleanup()
