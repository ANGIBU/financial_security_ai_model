# model_handler.py

import torch
import re
import time
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import warnings
warnings.filterwarnings("ignore")

@dataclass
class InferenceResult:
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0
    korean_quality: float = 0.0

class ModelHandler:
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = True, max_memory_gb: int = 22, verbose: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        if self.verbose:
            print(f"모델 로딩: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if self.verbose:
                print("Flash Attention 2 활성화")
        except ImportError:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_kwargs["attn_implementation"] = "sdpa"
                if self.verbose:
                    print("SDPA Attention 활성화")
            else:
                if self.verbose:
                    print("Standard Attention 사용")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                if self.verbose:
                    print("Torch Compile 활성화")
            except Exception:
                if self.verbose:
                    print("Torch Compile 실패, 기본 모델 사용")
        
        self._prepare_korean_optimization()
        
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = 500
        
        if self.verbose:
            print("모델 로딩 완료")
    
    def _prepare_korean_optimization(self):
        self.korean_tokens = []
        korean_chars = "가나다라마바사아자차카타파하개내대래매배새애재채캐태패해는을의에서과도"
        
        for char in korean_chars:
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            if tokens:
                self.korean_tokens.extend(tokens)
        
        self.korean_tokens = list(set(self.korean_tokens))
        
        self.bad_words_ids = []
        bad_patterns = [
            ["软", "件"], ["軟", "件"], ["金", "融"], ["電", "子"], ["個", "人"],
            ["資", "訊"], ["管", "理"], ["安", "全"], ["交", "易"]
        ]
        
        for pattern in bad_patterns:
            try:
                tokens = self.tokenizer.encode("".join(pattern), add_special_tokens=False)
                if tokens:
                    self.bad_words_ids.append(tokens)
            except:
                continue
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        if question_type == "multiple_choice":
            korean_prefix = "### 지시사항: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###\n\n"
            korean_suffix = "\n\n### 중요: 숫자만 답하세요 ###\n정답:"
        else:
            korean_prefix = "### 지시사항: 반드시 순수 한국어로만 답변하세요. 한자나 영어 절대 금지 ###\n\n"
            korean_suffix = "\n\n### 중요: 순수 한국어만 사용하세요 ###\n답변:"
        
        return korean_prefix + prompt + korean_suffix
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 3) -> InferenceResult:
        
        start_time = time.time()
        
        cache_key = hash(prompt[:100])
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        optimized_prompt = self._create_korean_optimized_prompt(prompt, question_type)
        
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            try:
                if question_type == "multiple_choice":
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.85,
                        top_k=25,
                        max_new_tokens=50,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids
                    )
                else:
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9,
                        top_k=50,
                        max_new_tokens=400,
                        repetition_penalty=1.03,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids
                    )
                
                inputs = self.tokenizer(
                    optimized_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.model.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config
                        )
                
                raw_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                cleaned_response = self._clean_korean_text_enhanced(raw_response)
                
                if question_type == "multiple_choice":
                    extracted_answer = self._extract_mc_answer_enhanced(cleaned_response)
                    
                    if extracted_answer and extracted_answer.isdigit() and 1 <= int(extracted_answer) <= 5:
                        result = InferenceResult(
                            response=extracted_answer,
                            confidence=0.9,
                            reasoning_quality=0.8,
                            analysis_depth=2,
                            korean_quality=1.0,
                            inference_time=time.time() - start_time
                        )
                        
                        if len(self.response_cache) >= self.max_cache_size:
                            oldest_key = next(iter(self.response_cache))
                            del self.response_cache[oldest_key]
                        self.response_cache[cache_key] = result
                        
                        return result
                    else:
                        continue
                
                else:
                    korean_quality = self._evaluate_korean_quality_enhanced(cleaned_response)
                    
                    if korean_quality > 0.3:
                        result = self._evaluate_response(cleaned_response, question_type)
                        result.korean_quality = korean_quality
                        result.inference_time = time.time() - start_time
                        
                        score = korean_quality * result.confidence
                        if score > best_score:
                            best_score = score
                            best_result = result
                    
                    if korean_quality > 0.6:
                        break
                        
            except Exception as e:
                if self.verbose:
                    print(f"생성 오류 (시도 {attempt+1}): {e}")
                continue
        
        if best_result is None:
            best_result = self._create_fallback_result(question_type)
            best_result.inference_time = time.time() - start_time
        
        return best_result
    
    def _extract_mc_answer_enhanced(self, text: str) -> str:
        patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'^([1-5])$',
            r'^([1-5])\s*$'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[0]
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    return answer
        
        numbers = re.findall(r'[1-5]', text)
        if numbers:
            return numbers[0]
        
        return ""
    
    def _clean_korean_text_enhanced(self, text: str) -> str:
        if not text:
            return ""
        
        chinese_to_korean = {
            r'軟[体體]件|软件': '소프트웨어',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'管理': '관리',
            r'個人|个人': '개인',
            r'資[讯訊]|资讯': '정보',
            r'電子|电子': '전자',
            r'系[统統]|系统': '시스템',
            r'保[护護]|保护': '보호',
            r'認[证證]|认证': '인증',
            r'加密': '암호화'
        }
        
        for pattern, replacement in chinese_to_korean.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3040-\u309f]+', '', text)
        text = re.sub(r'[\u30a0-\u30ff]+', '', text)
        
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
        
        return text.strip()
    
    def _evaluate_korean_quality_enhanced(self, text: str) -> float:
        if not text:
            return 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.1
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.1
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.85
        quality -= english_ratio * 0.15
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안', '시스템']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.08, 0.25)
        
        if 30 <= len(text) <= 500:
            quality += 0.12
        
        structure_markers = ['첫째', '둘째', '따라서', '그러므로']
        if any(marker in text for marker in structure_markers):
            quality += 0.08
        
        return max(0, min(1, quality))
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            return InferenceResult(
                response="2",
                confidence=0.5,
                reasoning_quality=0.4,
                analysis_depth=1,
                korean_quality=1.0
            )
        else:
            return InferenceResult(
                response="관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다.",
                confidence=0.7,
                reasoning_quality=0.6,
                analysis_depth=1,
                korean_quality=0.9
            )
    
    def _evaluate_response(self, response: str, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            confidence = 0.6
            reasoning = 0.6
            
            if re.match(r'^[1-5]$', response.strip()):
                confidence = 0.9
                reasoning = 0.8
            elif re.search(r'[1-5]', response):
                confidence = 0.75
                reasoning = 0.65
            
            return InferenceResult(
                response=response,
                confidence=confidence,
                reasoning_quality=reasoning,
                analysis_depth=2
            )
        
        else:
            confidence = 0.7
            reasoning = 0.7
            
            length = len(response)
            if 50 <= length <= 600:
                confidence += 0.15
            elif 30 <= length < 50:
                confidence += 0.08
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계', '방안']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 3:
                confidence += 0.15
                reasoning += 0.15
            elif keyword_count >= 2:
                confidence += 0.1
                reasoning += 0.1
            elif keyword_count >= 1:
                confidence += 0.05
                reasoning += 0.05
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3
            )
    
    def cleanup(self):
        if self.verbose:
            print(f"캐시 히트: {self.cache_hits}회")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        self.response_cache.clear()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache)
        }