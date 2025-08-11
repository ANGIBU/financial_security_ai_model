# model_handler.py

"""
모델 핸들러
- LLM 모델 로딩 및 관리
- 텍스트 생성 및 추론
- 한국어 최적화 설정
- 메모리 관리
"""

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
        self.max_cache_size = 400
        self.memory_cleanup_counter = 0
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "korean_quality_scores": [],
            "avg_inference_time": 0.0,
            "cache_efficiency": 0.0
        }
        
        if self.verbose:
            print("모델 로딩 완료")
    
    def _prepare_korean_optimization(self):
        self.bad_words_ids = []
        
        chinese_patterns = [
            "金融", "交易", "安全", "管理", "個人", "資訊", "電子", "系統",
            "保護", "認證", "加密", "網路"
        ]
        
        for pattern in chinese_patterns:
            try:
                tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
                if tokens and len(tokens) <= 3:
                    self.bad_words_ids.append(tokens)
            except:
                continue
        
        special_symbols = ["①", "②", "③", "④", "⑤", "➀", "➁", "❶", "❷", "❸"]
        for symbol in special_symbols:
            try:
                tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
                if tokens:
                    self.bad_words_ids.append(tokens)
            except:
                continue
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        if question_type == "multiple_choice":
            korean_prefix = "다음 문제의 정답 번호를 선택하세요.\n\n"
            korean_suffix = "\n\n정답 번호(1-5):"
        else:
            korean_prefix = "다음 질문에 한국어로 답변하세요.\n\n"
            korean_suffix = "\n\n답변:"
        
        return korean_prefix + prompt + korean_suffix
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 2) -> InferenceResult:
        
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
                        top_k=40,
                        max_new_tokens=20,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids[:5] if self.bad_words_ids else None
                    )
                else:
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        top_k=50,
                        max_new_tokens=300,
                        repetition_penalty=1.03,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids[:10] if self.bad_words_ids else None
                    )
                
                inputs = self.tokenizer(
                    optimized_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1000
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
                
                cleaned_response = self._clean_korean_text_safe(raw_response)
                
                if question_type == "multiple_choice":
                    extracted_answer = self._extract_mc_answer_simple(cleaned_response)
                    
                    if extracted_answer and extracted_answer.isdigit() and 1 <= int(extracted_answer) <= 5:
                        result = InferenceResult(
                            response=extracted_answer,
                            confidence=0.9,
                            reasoning_quality=0.8,
                            analysis_depth=2,
                            korean_quality=1.0,
                            inference_time=time.time() - start_time
                        )
                        
                        self._manage_cache()
                        self.response_cache[cache_key] = result
                        self._update_generation_stats(result, True)
                        
                        return result
                    else:
                        continue
                
                else:
                    korean_quality = self._evaluate_korean_quality_simple(cleaned_response)
                    
                    if korean_quality > 0.4 and len(cleaned_response) > 20:
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
            self._update_generation_stats(best_result, False)
        else:
            self._update_generation_stats(best_result, True)
        
        self._perform_memory_cleanup()
        
        return best_result
    
    def _extract_mc_answer_simple(self, text: str) -> str:
        if re.match(r'^[1-5]$', text.strip()):
            return text.strip()
        
        patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'^([1-5])',
            r'([1-5])번'
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
    
    def _clean_korean_text_safe(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        simple_replacements = {
            '金融': '금융',
            '交易': '거래',
            '安全': '안전',
            '管理': '관리',
            '個人': '개인',
            '資訊': '정보',
            '電子': '전자',
            '系統': '시스템'
        }
        
        for chinese, korean in simple_replacements.items():
            text = text.replace(chinese, korean)
        
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]', '', text)
        text = re.sub(r'[а-яё]', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'[①②③④⑤➀➁❶❷❸❹❺]', '', text)
        text = re.sub(r'bo+', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-]', ' ', text)
        
        return text.strip()
    
    def _evaluate_korean_quality_simple(self, text: str) -> float:
        if not text:
            return 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.1
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.8 - english_ratio * 0.2
        
        if 30 <= len(text) <= 400:
            quality += 0.1
        
        professional_terms = ['법', '규정', '관리', '보안', '조치', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.05, 0.15)
        
        return max(0, min(1, quality))
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            return InferenceResult(
                response="3",
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
            if 50 <= length <= 400:
                confidence += 0.1
            elif length > 400:
                confidence -= 0.1
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 2:
                confidence += 0.1
                reasoning += 0.1
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3
            )
    
    def _manage_cache(self):
        if len(self.response_cache) >= self.max_cache_size:
            keys_to_remove = list(self.response_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _perform_memory_cleanup(self):
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % 30 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.memory_cleanup_counter % 60 == 0:
            if len(self.response_cache) > self.max_cache_size // 2:
                self._manage_cache()
    
    def _update_generation_stats(self, result: InferenceResult, success: bool):
        self.generation_stats["total_generations"] += 1
        if success:
            self.generation_stats["successful_generations"] += 1
        
        self.generation_stats["korean_quality_scores"].append(result.korean_quality)
        if len(self.generation_stats["korean_quality_scores"]) > 50:
            self.generation_stats["korean_quality_scores"] = self.generation_stats["korean_quality_scores"][-50:]
        
        total_time = self.generation_stats["avg_inference_time"] * (self.generation_stats["total_generations"] - 1)
        total_time += result.inference_time
        self.generation_stats["avg_inference_time"] = total_time / self.generation_stats["total_generations"]
        
        total_requests = self.generation_stats["total_generations"]
        self.generation_stats["cache_efficiency"] = self.cache_hits / max(total_requests, 1)
    
    def get_performance_stats(self) -> Dict:
        avg_korean_quality = 0.0
        if self.generation_stats["korean_quality_scores"]:
            avg_korean_quality = sum(self.generation_stats["korean_quality_scores"]) / len(self.generation_stats["korean_quality_scores"])
        
        success_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            success_rate = self.generation_stats["successful_generations"] / self.generation_stats["total_generations"]
        
        return {
            "model_name": self.model_name,
            "total_generations": self.generation_stats["total_generations"],
            "success_rate": success_rate,
            "avg_korean_quality": avg_korean_quality,
            "avg_inference_time": self.generation_stats["avg_inference_time"],
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "cache_efficiency": self.generation_stats["cache_efficiency"],
            "memory_cleanups": self.memory_cleanup_counter
        }
    
    def cleanup(self):
        if self.verbose:
            stats = self.get_performance_stats()
            print(f"모델 통계: 생성 성공률 {stats['success_rate']:.1%}, 한국어 품질 {stats['avg_korean_quality']:.2f}")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        self.response_cache.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache)
        }